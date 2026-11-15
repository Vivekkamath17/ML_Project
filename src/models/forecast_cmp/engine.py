"""Forecasting comparison implementation for final_score regression and grade_band classification.

Trains models strictly on train split (2013B + 2013J), tunes with 15 Optuna trials using
5-fold GroupKFold by id_student, evaluates on val (2014B) at weeks 4, 8, 12.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

from src.models.predictions import write_predictions

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEEDS = [42, 101, 2024, 7, 999]
SNAPSHOT_WEEKS = [4, 8, 12]
GRADE_MAP = {"Fail": 0, "Pass": 1, "Distinction": 2}
REV_GRADE_MAP = {0: "Fail", 1: "Pass", 2: "Distinction"}


def load_v3_feature_list(features_yaml_path: str = "config/features.yaml") -> List[str]:
    with open(features_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    v3 = [x for x in cfg["scaled_features"] if x != "n_skipped_resources"]
    assert len(v3) == 38, f"Expected 38 v3 features, got {len(v3)}"
    forbidden = [
        "final_result", "final_score", "grade_band", "never_active_learner",
        "is_zero_click_withdrawal", "label_is_proxy", "code_module", "code_presentation",
        "gender", "region", "highest_education", "imd_band", "age_band", "disability"
    ]
    leaks = set(v3).intersection(forbidden)
    assert len(leaks) == 0, f"Forbidden features found: {leaks}"
    return v3


def load_forecasting_data(
    features_path: str = "data/processed/features_weekly.parquet",
    labels_path: str = "data/processed/labels.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    v3_features = load_v3_feature_list()
    
    meta_cols = [
        "id_student", "code_module", "code_presentation", "week", "split",
        "final_result", "never_active_learner"
    ]
    
    # Read features
    df_feat = pd.read_parquet(features_path, columns=meta_cols + v3_features)
    # Read labels
    df_lbl = pd.read_parquet(labels_path, columns=["id_student", "code_module", "code_presentation", "week", "final_score", "grade_band"])
    
    # Merge strictly on grid keys
    merged = df_feat.merge(df_lbl, on=["id_student", "code_module", "code_presentation", "week"], how="inner")
    
    # Strict quarantine check: ensure no test presentation 2014J is used
    assert (merged["code_presentation"] != "2014J").all() or (merged[merged["code_presentation"] == "2014J"]["split"] == "test").any()
    
    # Cohort filtering: non-withdrawn, never-active excluded
    cohort = merged[(merged["final_result"] != "Withdrawn") & (~merged["never_active_learner"])].copy()
    
    train_df = cohort[(cohort["split"] == "train") & (cohort["week"].isin(SNAPSHOT_WEEKS))].copy()
    val_df = cohort[(cohort["split"] == "val") & (cohort["week"].isin(SNAPSHOT_WEEKS))].copy()
    
    # Check learner counts
    val_w4_learners = set(zip(val_df[val_df["week"] == 4]["code_module"], val_df[val_df["week"] == 4]["id_student"]))
    val_w8_learners = set(zip(val_df[val_df["week"] == 8]["code_module"], val_df[val_df["week"] == 8]["id_student"]))
    val_w12_learners = set(zip(val_df[val_df["week"] == 12]["code_module"], val_df[val_df["week"] == 12]["id_student"]))
    
    val_common = val_w4_learners & val_w8_learners & val_w12_learners
    assert len(val_common) == 5079, f"Expected 5,079 val learners, got {len(val_common)}"
    assert len(val_w4_learners) == 5079 and len(val_w8_learners) == 5079 and len(val_w12_learners) == 5079
    
    # Filter to exact common learners
    val_keys = set(val_common)
    val_df["learner_key"] = list(zip(val_df["code_module"], val_df["id_student"]))
    val_df = val_df[val_df["learner_key"].isin(val_keys)].drop(columns=["learner_key"])
    
    train_w4_learners = set(zip(train_df[train_df["week"] == 4]["code_module"], train_df[train_df["week"] == 4]["id_student"]))
    train_w8_learners = set(zip(train_df[train_df["week"] == 8]["code_module"], train_df[train_df["week"] == 8]["id_student"]))
    train_w12_learners = set(zip(train_df[train_df["week"] == 12]["code_module"], train_df[train_df["week"] == 12]["id_student"]))
    train_common = train_w4_learners & train_w8_learners & train_w12_learners
    
    train_df["learner_key"] = list(zip(train_df["code_module"], train_df["id_student"]))
    train_df = train_df[train_df["learner_key"].isin(train_common)].drop(columns=["learner_key"])
    
    print(f"Loaded Train: {len(train_common):,} learners ({len(train_df):,} rows), Val: {len(val_common):,} learners ({len(val_df):,} rows)")
    print(f"Train mean final_score: {train_df[train_df['week'] == 4]['final_score'].mean():.4f}")
    print(f"Val mean final_score: {val_df[val_df['week'] == 4]['final_score'].mean():.4f}")
    
    return train_df, val_df, v3_features


class LinearPreprocessor:
    """Median imputation + missing indicators fitted on train only."""
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.imputer = SimpleImputer(strategy="median")
        self.indicator = MissingIndicator()
        
    def fit(self, X: pd.DataFrame):
        X_mat = X[self.feature_names].values
        self.imputer.fit(X_mat)
        self.indicator.fit(X_mat)
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = X[self.feature_names].values
        imputed = self.imputer.transform(X_mat)
        ind = self.indicator.transform(X_mat)
        if ind.shape[1] > 0:
            return np.hstack([imputed, ind.astype(float)])
        return imputed


def cluster_bootstrap_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_ids: np.ndarray,
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    mae_pt = float(mean_absolute_error(y_true, y_pred))
    rmse_pt = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_pt = float(r2_score(y_true, y_pred))
    
    unique_c, inv_c = np.unique(cluster_ids, return_inverse=True)
    n_c = len(unique_c)
    
    cluster_rows = [[] for _ in range(n_c)]
    for r_idx, c_idx in enumerate(inv_c):
        cluster_rows[c_idx].append(r_idx)
    cluster_rows_arr = [np.array(r, dtype=np.int64) for r in cluster_rows]
    
    rng = np.random.default_rng(seed)
    maes = np.empty(n_resamples)
    rmses = np.empty(n_resamples)
    r2s = np.empty(n_resamples)
    
    for b in range(n_resamples):
        sampled = rng.integers(0, n_c, size=n_c)
        rows = np.concatenate([cluster_rows_arr[c] for c in sampled])
        yt = y_true[rows]
        yp = y_pred[rows]
        maes[b] = mean_absolute_error(yt, yp)
        rmses[b] = np.sqrt(mean_squared_error(yt, yp))
        r2s[b] = r2_score(yt, yp)
        
    def get_stat(pt, arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return (float(pt), float(lo), float(hi))
        
    return {
        "mae": get_stat(mae_pt, maes),
        "rmse": get_stat(rmse_pt, rmses),
        "r2": get_stat(r2_pt, r2s),
    }


def paired_bootstrap_delta_regression(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    cluster_ids: np.ndarray,
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float, bool]]:
    delta_mae_pt = float(mean_absolute_error(y_true, y_pred_model) - mean_absolute_error(y_true, y_pred_baseline))
    delta_r2_pt = float(r2_score(y_true, y_pred_model) - r2_score(y_true, y_pred_baseline))
    
    unique_c, inv_c = np.unique(cluster_ids, return_inverse=True)
    n_c = len(unique_c)
    
    cluster_rows = [[] for _ in range(n_c)]
    for r_idx, c_idx in enumerate(inv_c):
        cluster_rows[c_idx].append(r_idx)
    cluster_rows_arr = [np.array(r, dtype=np.int64) for r in cluster_rows]
    
    rng = np.random.default_rng(seed)
    d_maes = np.empty(n_resamples)
    d_r2s = np.empty(n_resamples)
    
    for b in range(n_resamples):
        sampled = rng.integers(0, n_c, size=n_c)
        rows = np.concatenate([cluster_rows_arr[c] for c in sampled])
        yt = y_true[rows]
        ym = y_pred_model[rows]
        yb = y_pred_baseline[rows]
        d_maes[b] = mean_absolute_error(yt, ym) - mean_absolute_error(yt, yb)
        d_r2s[b] = r2_score(yt, ym) - r2_score(yt, yb)
        
    def parse_ci(pt, arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        includes_zero = (lo <= 0.0 <= hi)
        return (float(pt), float(lo), float(hi), includes_zero)
        
    return {
        "delta_mae": parse_ci(delta_mae_pt, d_maes),
        "delta_r2": parse_ci(delta_r2_pt, d_r2s),
    }


def cluster_bootstrap_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_ids: np.ndarray,
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    f1_pt = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk_pt = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    recalls_pt = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    unique_c, inv_c = np.unique(cluster_ids, return_inverse=True)
    n_c = len(unique_c)
    
    cluster_rows = [[] for _ in range(n_c)]
    for r_idx, c_idx in enumerate(inv_c):
        cluster_rows[c_idx].append(r_idx)
    cluster_rows_arr = [np.array(r, dtype=np.int64) for r in cluster_rows]
    
    rng = np.random.default_rng(seed)
    f1s = np.empty(n_resamples)
    qwks = np.empty(n_resamples)
    r_fails = np.empty(n_resamples)
    r_passes = np.empty(n_resamples)
    r_dists = np.empty(n_resamples)
    
    for b in range(n_resamples):
        sampled = rng.integers(0, n_c, size=n_c)
        rows = np.concatenate([cluster_rows_arr[c] for c in sampled])
        yt = y_true[rows]
        yp = y_pred[rows]
        f1s[b] = f1_score(yt, yp, average="macro", zero_division=0)
        qwks[b] = cohen_kappa_score(yt, yp, weights="quadratic")
        rc = recall_score(yt, yp, average=None, zero_division=0, labels=[0, 1, 2])
        r_fails[b] = rc[0]
        r_passes[b] = rc[1]
        r_dists[b] = rc[2]
        
    def get_stat(pt, arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return (float(pt), float(lo), float(hi))
        
    return {
        "macro_f1": get_stat(f1_pt, f1s),
        "qwk": get_stat(qwk_pt, qwks),
        "recall_fail": get_stat(recalls_pt[0], r_fails),
        "recall_pass": get_stat(recalls_pt[1], r_passes),
        "recall_dist": get_stat(recalls_pt[2], r_dists),
    }


def paired_bootstrap_delta_classification(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    cluster_ids: np.ndarray,
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float, bool]]:
    d_f1_pt = float(f1_score(y_true, y_pred_model, average="macro", zero_division=0) - f1_score(y_true, y_pred_baseline, average="macro", zero_division=0))
    d_qwk_pt = float(cohen_kappa_score(y_true, y_pred_model, weights="quadratic") - cohen_kappa_score(y_true, y_pred_baseline, weights="quadratic"))
    
    unique_c, inv_c = np.unique(cluster_ids, return_inverse=True)
    n_c = len(unique_c)
    
    cluster_rows = [[] for _ in range(n_c)]
    for r_idx, c_idx in enumerate(inv_c):
        cluster_rows[c_idx].append(r_idx)
    cluster_rows_arr = [np.array(r, dtype=np.int64) for r in cluster_rows]
    
    rng = np.random.default_rng(seed)
    d_f1s = np.empty(n_resamples)
    d_qwks = np.empty(n_resamples)
    
    for b in range(n_resamples):
        sampled = rng.integers(0, n_c, size=n_c)
        rows = np.concatenate([cluster_rows_arr[c] for c in sampled])
        yt = y_true[rows]
        ym = y_pred_model[rows]
        yb = y_pred_baseline[rows]
        d_f1s[b] = f1_score(yt, ym, average="macro", zero_division=0) - f1_score(yt, yb, average="macro", zero_division=0)
        d_qwks[b] = cohen_kappa_score(yt, ym, weights="quadratic") - cohen_kappa_score(yt, yb, weights="quadratic")
        
    def parse_ci(pt, arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        includes_zero = (lo <= 0.0 <= hi)
        return (float(pt), float(lo), float(hi), includes_zero)
        
    return {
        "delta_macro_f1": parse_ci(d_f1_pt, d_f1s),
        "delta_qwk": parse_ci(d_qwk_pt, d_qwks),
    }
