"""
Sequence Builder module for Phase 4 Sequence Models.
Constructs windowed sequence tensors, validity masks, static vectors, and targets.
Enforces:
- Strict point-in-time safety: no information after week t.
- Left-padding with zeros; right-aligned real steps.
- Train-only fitted StandardScaler with training-median imputation.
- Padded steps forced to exact 0.0 with mask False.
- Caching to npz + JSON manifest for deterministic, reproducible loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler

from src.models.data import load_feature_manifest, load_split
from src.seq.config import (
    CACHE_DIR,
    DATA_PATH,
    STATIC_FEATURES,
    TARGET_MAP,
)


class SequenceScaler:
    """
    Train-only scaler for sequence modeling.
    Computes medians and standard scaling parameters strictly on train rows.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.medians: Dict[str, float] = {}
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, df_train: pd.DataFrame) -> "SequenceScaler":
        self.medians = {}
        means = []
        scales = []
        for col in self.feature_names:
            series = df_train[col]
            med = float(series.median())
            if np.isnan(med):
                med = 0.0
            self.medians[col] = med
            filled = series.fillna(med).to_numpy(dtype=np.float64)
            m = float(filled.mean())
            s = float(filled.std())
            if s < 1e-7:
                s = 1.0
            means.append(m)
            scales.append(s)

        self.mean_ = np.array(means, dtype=np.float32)
        self.scale_ = np.array(scales, dtype=np.float32)
        return self

    def transform(self, array_or_df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(array_or_df, pd.DataFrame):
            arr = np.empty((len(array_or_df), len(self.feature_names)), dtype=np.float32)
            for j, col in enumerate(self.feature_names):
                arr[:, j] = array_or_df[col].fillna(self.medians[col]).to_numpy(dtype=np.float32)
        else:
            arr = array_or_df.copy().astype(np.float32)
            # Impute NaNs with medians along feature columns
            for j, col in enumerate(self.feature_names):
                nan_mask = np.isnan(arr[:, j])
                if nan_mask.any():
                    arr[nan_mask, j] = self.medians[col]

        scaled = (arr - self.mean_) / self.scale_
        return scaled.astype(np.float32)


def get_v3_feature_names() -> List[str]:
    """Return the official 38-feature subset (excluding n_skipped_resources)."""
    manifest = load_feature_manifest()
    return [f for f in manifest if f != "n_skipped_resources"]


def get_or_fit_scaler(data_path: Union[str, Path] = DATA_PATH) -> SequenceScaler:
    """Load or fit the SequenceScaler strictly on training rows."""
    scaler_path = CACHE_DIR / "train_scaler.joblib"
    features = get_v3_feature_names()

    if scaler_path.is_file():
        scaler = joblib.load(scaler_path)
        if scaler.feature_names == features:
            return scaler

    print("Fitting SequenceScaler strictly on train split...")
    df_raw = pl.scan_parquet(data_path).filter(pl.col("split") == "train").select(features).collect()
    scaler = SequenceScaler(features)
    scaler.fit(df_raw.to_pandas())
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted scaler to {scaler_path}")
    return scaler


def build_sequences(
    split: str,
    window: int = 8,
    target: str = "dropout",
    force_rebuild: bool = False,
    data_path: Union[str, Path] = DATA_PATH,
) -> Dict[str, Any]:
    """
    Build sequence tensors for a given split and window.

    Parameters:
    -----------
    split: 'train' or 'val'
    window: int, window length W in {4, 8}
    target: 'dropout' (default) or 'onset'
    force_rebuild: bool, whether to bypass disk cache

    Returns:
    --------
    dict with keys:
      'X': np.ndarray, shape (N, W, 38), float32
      'M': np.ndarray, shape (N, W), bool
      'S': np.ndarray, shape (N, 4), float32
      'y_dropout': np.ndarray, shape (N,), int32
      'y_onset': np.ndarray, shape (N,), int32
      'onset_mask': np.ndarray, shape (N,), bool
      'final_score': np.ndarray, shape (N,), float32
      'score_mask': np.ndarray, shape (N,), bool
      'meta': pd.DataFrame, N rows
      'feature_names': list of 38 feature names
      'static_feature_names': list of 4 static feature names
    """
    cache_file = CACHE_DIR / f"{split}_w{window}_{target}.npz"
    manifest_file = CACHE_DIR / f"{split}_w{window}_{target}_manifest.json"
    meta_cache = CACHE_DIR / f"{split}_w{window}_{target}_meta.parquet"

    features = get_v3_feature_names()
    static_feats = [f for f in STATIC_FEATURES if f in features]

    if not force_rebuild and cache_file.is_file() and manifest_file.is_file() and meta_cache.is_file():
        loaded = np.load(cache_file)
        df_meta = pd.read_parquet(meta_cache)
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        return {
            "X": loaded["X"],
            "M": loaded["M"],
            "S": loaded["S"],
            "y_dropout": loaded["y_dropout"],
            "y_onset": loaded["y_onset"],
            "onset_mask": loaded["onset_mask"],
            "final_score": loaded["final_score"],
            "score_mask": loaded["score_mask"],
            "meta": df_meta,
            "feature_names": manifest["feature_names"],
            "static_feature_names": manifest["static_feature_names"],
        }

    # 1. Load target rows from load_split to guarantee 100% row alignment
    target_col = TARGET_MAP.get(target, target)
    allow_t = (split == "test")
    if target == "dropout":
        X_df, y_arr, meta_df = load_split(split, k=3, min_week=2, drop_never_active=True, allow_test=allow_t)
    elif target == "onset":
        X_df, y_arr, meta_df = load_split(split, k=3, min_week=2, target_col="label_inactive_k3", allow_test=allow_t)
    else:
        raise ValueError(f"Unknown target: {target}")

    meta_df["learner_key"] = (
        meta_df["code_module"] + "_" + meta_df["code_presentation"] + "_" + meta_df["id_student"].astype(str)
    )

    N = len(meta_df)
    target_learners = meta_df["learner_key"].to_numpy()
    target_weeks = meta_df["week"].to_numpy()

    # 2. Load the full contiguous grid from features_weekly.parquet
    print(f"Loading full features_weekly grid for {split}...")
    full_df = pl.read_parquet(data_path)
    full_df = full_df.with_columns(
        (pl.col("code_module") + "_" + pl.col("code_presentation") + "_" + pl.col("id_student").cast(pl.Utf8)).alias("learner_key")
    ).sort(["learner_key", "week"])

    # 3. Fit or retrieve scaler
    scaler = get_or_fit_scaler(data_path)

    # 4. Extract and scale all features in the sorted grid
    print("Scaling full feature grid...")
    feat_pd = full_df.select(features).to_pandas()
    scaled_grid = scaler.transform(feat_pd)  # shape (total_rows, 38)

    # Learner start index lookup
    grid_learners = full_df["learner_key"].to_numpy()
    unique_learners, first_indices = np.unique(grid_learners, return_index=True)
    learner_to_start = dict(zip(unique_learners, first_indices))

    # 5. Build sequence tensor X and mask M
    print(f"Building sequence tensor: N={N}, W={window}, F={len(features)}...")
    X = np.zeros((N, window, len(features)), dtype=np.float32)
    M = np.zeros((N, window), dtype=bool)

    learner_starts = np.array([learner_to_start[l] for l in target_learners], dtype=np.int64)

    for step in range(window):
        rel_week = target_weeks - window + 1 + step
        valid_mask = rel_week >= 0
        valid_indices = np.where(valid_mask)[0]
        row_indices = learner_starts[valid_indices] + rel_week[valid_indices]

        X[valid_indices, step, :] = scaled_grid[row_indices]
        M[valid_indices, step] = True

    # 6. Build static tensor S (scaled static features)
    static_indices = [features.index(f) for f in static_feats]
    # S takes the static features from the last step (which is identical across real steps)
    S = X[:, -1, static_indices].copy()

    # 7. Build multi-task labels and validity masks
    y_dropout = np.zeros(N, dtype=np.int32)
    y_onset = np.zeros(N, dtype=np.int32)
    onset_mask = np.zeros(N, dtype=bool)
    final_score = np.zeros(N, dtype=np.float32)
    score_mask = np.zeros(N, dtype=bool)

    # Join target columns from full_df at week t
    target_cols_to_fetch = ["label_dropout_k3", "label_inactive_k3", "final_score", "final_result"]
    meta_df_keys = pd.DataFrame({"learner_key": target_learners, "week": target_weeks})

    sub_targets = (
        full_df.select(["learner_key", "week"] + target_cols_to_fetch)
        .to_pandas()
        .merge(meta_df_keys, on=["learner_key", "week"], how="right")
    )

    # Dropout label
    d_vals = sub_targets["label_dropout_k3"].to_numpy()
    y_dropout[:] = np.where(np.isnan(d_vals), 0, d_vals).astype(np.int32)

    # Onset label
    o_vals = sub_targets["label_inactive_k3"].to_numpy()
    valid_onset = ~np.isnan(o_vals)
    y_onset[valid_onset] = o_vals[valid_onset].astype(np.int32)
    onset_mask[:] = valid_onset

    # Final score
    s_vals = sub_targets["final_score"].to_numpy()
    res_vals = sub_targets["final_result"].to_numpy()
    valid_score = (~np.isnan(s_vals)) & (res_vals != "Withdrawn")
    final_score[valid_score] = s_vals[valid_score].astype(np.float32)
    score_mask[:] = valid_score

    # 8. Cache outputs
    print(f"Caching sequence dataset to {cache_file}...")
    np.savez_compressed(
        cache_file,
        X=X,
        M=M,
        S=S,
        y_dropout=y_dropout,
        y_onset=y_onset,
        onset_mask=onset_mask,
        final_score=final_score,
        score_mask=score_mask,
    )

    meta_df.to_parquet(meta_cache)

    manifest_data = {
        "split": split,
        "window": window,
        "target": target,
        "N": int(N),
        "W": int(window),
        "F": len(features),
        "feature_names": features,
        "static_feature_names": static_feats,
        "cache_file": str(cache_file),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Done! Dataset ready: X={X.shape}, M={M.shape}, S={S.shape}")
    return {
        "X": X,
        "M": M,
        "S": S,
        "y_dropout": y_dropout,
        "y_onset": y_onset,
        "onset_mask": onset_mask,
        "final_score": final_score,
        "score_mask": score_mask,
        "meta": meta_df,
        "feature_names": features,
        "static_feature_names": static_feats,
    }
