"""
Single Data Loader module for Phase 2 Modeling.
All filtering (enrolled, non-censored, min_week, population flags) lives here.
Strict test-split guard: raises ValueError if split == 'test' unless allow_test=True.
Uses ONLY the 39 sanctioned features from config/features.yaml.
"""
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import polars as pl
import yaml


def load_feature_manifest(manifest_path: str = "config/features.yaml", scaled_only: bool = True) -> List[str]:
    """
    Load the sanctioned model feature names.
    If scaled_only is True (default), returns the 39 continuous numeric features.
    If scaled_only is False, returns all 42 features (39 continuous + 3 boolean flags).
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    
    if scaled_only:
        features = list(manifest["scaled_features"])
        assert len(features) == 39, f"Expected 39 scaled features, found {len(features)}"
        return features
    else:
        features = []
        for grp in ["static", "behavior", "temporal", "assessment", "curriculum"]:
            features.extend(manifest["features"][grp])
        return features


def load_split(
    split: str,
    k: int = 3,
    min_week: int = 2,
    drop_never_active: bool = True,
    drop_proxy: bool = False,
    non_withdrawn_only: bool = False,
    target_col: Optional[str] = None,
    allow_test: bool = False,
    data_path: str = "data/processed/features_weekly.parquet",
    manifest_path: str = "config/features.yaml",
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load a single split with strict point-in-time filtering.

    Parameters:
    -----------
    split: "train", "val", or "test"
    k: prediction horizon (default 3)
    min_week: minimum course week (default 2)
    drop_never_active: exclude never_active_learner rows (config default True for train)
    drop_proxy: exclude label_is_proxy rows
    non_withdrawn_only: include only non-withdrawn learners (for forecasting)
    target_col: custom target column (e.g. 'label_inactive_k3', 'final_score', 'grade_band')
    allow_test: explicit security guard. If False and split == 'test', raises ValueError.

    Returns:
    --------
    X: pandas DataFrame containing ONLY the 39 sanctioned features.
    y: numpy 1D array of labels/targets.
    meta: pandas DataFrame with metadata (identifiers, week, module, flags).
    """
    if split == "test" and not allow_test:
        raise ValueError(
            "CRITICAL SECURITY GUARD: The test split (2014J) must NOT be loaded or used in Phase 2! "
            "allow_test=True is strictly forbidden."
        )

    if split not in ["train", "val", "test"]:
        raise ValueError(f"Unknown split: '{split}'. Must be 'train', 'val', or 'test'.")

    sanctioned_features = load_feature_manifest(manifest_path)

    # Scan parquet lazily for efficiency
    lf = pl.scan_parquet(data_path)

    # Resolve target and filtering
    actual_target = target_col if target_col is not None else f"label_dropout_k{k}"
    censored_col = f"is_censored_k{k}"

    filter_expr = (
        (pl.col("split") == split)
        & (pl.col("is_enrolled_at_week_t"))
        & (pl.col("week") >= min_week)
    )

    if actual_target.startswith("label_dropout_"):
        filter_expr = filter_expr & (~pl.col(censored_col))
    elif actual_target.startswith("label_inactive_"):
        filter_expr = filter_expr & pl.col(actual_target).is_not_null()

    if non_withdrawn_only:
        filter_expr = filter_expr & (pl.col("final_result") != "Withdrawn")

    if drop_never_active:
        filter_expr = filter_expr & (~pl.col("never_active_learner"))

    if drop_proxy:
        filter_expr = filter_expr & (~pl.col("label_is_proxy"))

    df = lf.filter(filter_expr).collect()

    # Extract metadata columns for slicing, bootstrap, and analysis
    meta_cols = [
        "code_module",
        "code_presentation",
        "id_student",
        "week",
        "split",
        "label_is_proxy",
        "never_active_learner",
        "is_zero_click_withdrawal",
        "is_post_assessment_period",
        "final_result",
        "final_score",
        "grade_band",
    ]

    # Target y
    if actual_target == "grade_band":
        y = df[actual_target].to_numpy()
    elif actual_target == "final_score":
        y = df[actual_target].to_numpy().astype(np.float64)
    else:
        y = df[actual_target].to_numpy().astype(np.int32)

    # Features X
    X = df.select(sanctioned_features).to_pandas()

    # Meta dataframe
    meta = df.select(meta_cols).to_pandas()

    # Print summary statistics
    learner_key = ["code_module", "code_presentation", "id_student"]
    n_learners = df.select(learner_key).unique().height
    try:
        pos_rate = float(y.mean())
        pos_str = f"Positive rate: {pos_rate*100:.3f}%"
    except Exception:
        pos_str = f"Classes/distribution: {pd.Series(y).value_counts().to_dict()}"

    print(
        f"Loaded split='{split}' (k={k}, min_week={min_week}, drop_never_active={drop_never_active}): "
        f"{len(X):,} rows across {n_learners:,} learners. {pos_str}"
    )

    return X, y, meta


if __name__ == "__main__":
    X_tr, y_tr, m_tr = load_split("train")
    X_va, y_va, m_va = load_split("val")
