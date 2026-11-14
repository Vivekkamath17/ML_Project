"""
Common data preparation, preprocessing pipelines, fold generation, and metrics helpers for Model Zoo.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.models.data import load_split
from src.models.harness import learner_group_cv


ASSESSMENT_FEATURES = [
    "mean_score_submitted",
    "last_score",
    "score_slope",
    "mean_lateness",
    "banked_score_mean",
]


class ImputedZooPreprocessor(BaseEstimator, TransformerMixin):
    """
    Train-only fitted preprocessor for non-tree and neural models.
    - Imputes missing continuous features with the training median.
    - Creates explicit binary missingness indicator columns for assessment features
      (preserving the critical 'no submission yet' behavioral signal).
    - StandardScales continuous columns.
    - Preserves missingness indicators as unscaled binary values.
    """

    def __init__(self, assessment_cols: Optional[List[str]] = None):
        self.assessment_cols = assessment_cols or ASSESSMENT_FEATURES
        self.medians_: Dict[str, float] = {}
        self.scaler_ = StandardScaler()
        self.feature_names_in_: List[str] = []
        self.output_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "ImputedZooPreprocessor":
        self.feature_names_in_ = list(X.columns)

        # 1. Compute medians strictly on training rows
        for col in self.feature_names_in_:
            med = float(X[col].median())
            # Fallback if entirely null (none should be, but safe)
            if np.isnan(med):
                med = 0.0
            self.medians_[col] = med

        # 2. Impute continuous columns temporarily to fit scaler
        X_imputed = X.copy()
        for col in self.feature_names_in_:
            X_imputed[col] = X_imputed[col].fillna(self.medians_[col])

        self.scaler_.fit(X_imputed[self.feature_names_in_])

        # 3. Output feature names: scaled continuous + binary indicators
        indicator_cols = [f"{col}_is_missing" for col in self.assessment_cols if col in self.feature_names_in_]
        self.output_feature_names_ = self.feature_names_in_ + indicator_cols
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_in_:
            raise ValueError("Preprocessor has not been fitted.")

        X_df = X.copy()

        # Generate missing indicators for assessment columns
        indicators = []
        for col in self.assessment_cols:
            if col in self.feature_names_in_:
                ind = X_df[col].isna().astype(np.float32).to_numpy().reshape(-1, 1)
                indicators.append(ind)

        # Impute medians
        for col in self.feature_names_in_:
            X_df[col] = X_df[col].fillna(self.medians_[col])

        # Scale continuous features
        scaled_continuous = self.scaler_.transform(X_df[self.feature_names_in_]).astype(np.float32)

        if indicators:
            indicators_mat = np.hstack(indicators)
            return np.hstack([scaled_continuous, indicators_mat])
        return scaled_continuous

    def get_feature_names(self) -> List[str]:
        return self.output_feature_names_


def load_zoo_data(
    target_col: Optional[str] = None,
    k: int = 3,
    min_week: int = 2,
    drop_never_active: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, List[str]]:
    """
    Load Train (2013B + 2013J) and Validation (2014B) splits with strict point-in-time filtering
    and restriction to the 38 v3 sanctioned features (excluding n_skipped_resources).

    Returns:
    --------
    X_tr, y_tr, meta_tr, X_va, y_va, meta_va, v3_features
    """
    X_tr_raw, y_tr, meta_tr = load_split(
        "train", k=k, min_week=min_week, drop_never_active=drop_never_active, target_col=target_col
    )
    X_va_raw, y_va, meta_va = load_split(
        "val", k=k, min_week=min_week, drop_never_active=drop_never_active, target_col=target_col
    )

    # 38 v3 features (all 39 sanctioned minus n_skipped_resources)
    v3_features = [col for col in X_tr_raw.columns if col != "n_skipped_resources"]

    X_tr = X_tr_raw[v3_features].copy()
    X_va = X_va_raw[v3_features].copy()

    return X_tr, y_tr, meta_tr, X_va, y_va, meta_va, v3_features


def get_imputed_pipeline(
    X_tr: pd.DataFrame,
    save_path: Union[str, Path] = "models/zoo/preprocess_imputed.joblib",
    force_refit: bool = False,
) -> ImputedZooPreprocessor:
    """
    Get or fit the imputed preprocessor strictly on training data.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not force_refit:
        preprocessor = joblib.load(save_path)
    else:
        preprocessor = ImputedZooPreprocessor()
        preprocessor.fit(X_tr)
        joblib.dump(preprocessor, save_path)

    return preprocessor


def get_learner_subsample_mask(
    meta_df: pd.DataFrame,
    fraction: float = 0.25,
    seed: int = 42,
) -> np.ndarray:
    """
    Deterministically subsample a fraction of unique students (id_student)
    for faster Optuna tuning in Random Forest and ExtraTrees.
    Returns a boolean mask of length len(meta_df).
    """
    unique_students = np.sort(meta_df["id_student"].unique())
    rng = np.random.default_rng(seed)
    n_sampled = max(1, int(len(unique_students) * fraction))
    sampled_students = set(rng.choice(unique_students, size=n_sampled, replace=False))

    mask = meta_df["id_student"].isin(sampled_students).to_numpy()
    return mask


def get_model_size_mb(path: Union[str, Path]) -> float:
    """Return model file size on disk in megabytes."""
    path = Path(path)
    if not path.exists():
        return 0.0
    return float(path.stat().st_size / (1024 * 1024))


def measure_inference_latency_ms_per_1k(
    predict_fn: Callable[[Any], np.ndarray],
    X_sample: Any,
    n_runs: int = 5,
    n_rows: Optional[int] = None,
) -> float:
    """
    Measure average inference latency in milliseconds per 1,000 rows.
    """
    if n_rows is None:
        if hasattr(X_sample, "num_row"):
            n_rows = X_sample.num_row()
        elif hasattr(X_sample, "shape"):
            n_rows = X_sample.shape[0]
        else:
            n_rows = len(X_sample)

    if n_rows == 0:
        return 0.0

    # Warmup
    _ = predict_fn(X_sample)

    durations = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = predict_fn(X_sample)
        durations.append(time.perf_counter() - t0)

    mean_sec = float(np.mean(durations))
    latency_ms_per_1k = (mean_sec / n_rows) * 1000 * 1000
    return latency_ms_per_1k


def compute_median_inner_iterations(fold_best_iterations: List[int]) -> int:
    """
    Derive the final tree/iteration count for the full training fit
    as the median of the best iteration numbers across the 5 inner CV folds.
    """
    if not fold_best_iterations:
        return 100
    return int(np.round(np.median(fold_best_iterations)))
