"""
SHAP Feature Attribution Engine for Dropout Early-Warning Models.
Generates genuine local SHAP explanations for all compatible models:
- LightGBM v3 (TreeSHAP in C++)
- XGBoost (TreeSHAP in C++)
- CatBoost (TreeSHAP in C++)
- Random Forest (TreeExplainer with tree-path attribution)
- Logistic Regression (Analytical linear SHAP)

Saves Top-5 signed feature contributions per learner-week to:
reports/shap/<model>_k3_val_top5.parquet
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import shap

from src.models.zoo.common import load_zoo_data, ImputedZooPreprocessor
from src.models.data import load_split

RAW_FEATURES_38 = [
    "num_of_prev_attempts", "studied_credits", "registration_lead_time",
    "precourse_clicks", "course_progress",
    "weekly_clicks", "days_since_last_activity", "weekend_click_ratio",
    "active_days_entropy", "share_forumng", "share_quiz", "share_resource",
    "share_oucontent", "share_homepage", "share_subpage", "share_url",
    "share_other_types",
    "wow_change_clicks", "wow_change_active_days", "rolling_mean_clicks_2w",
    "rolling_mean_clicks_4w", "rolling_mean_active_days_2w",
    "rolling_mean_active_days_4w", "clicks_slope_4w", "clicks_ewma",
    "engagement_decay_rate",
    "n_submitted", "n_missed", "mean_score_submitted", "last_score",
    "score_slope", "mean_lateness", "weighted_score_to_date", "n_banked",
    "banked_score_mean",
    "pct_resources_visited", "pacing_vs_cohort_median", "prerequisite_gap",
]

IMPUTED_INDICATOR_COLS = [
    "mean_score_submitted_is_missing",
    "last_score_is_missing",
    "score_slope_is_missing",
    "mean_lateness_is_missing",
    "banked_score_mean_is_missing",
]
IMPUTED_FEATURES_43 = RAW_FEATURES_38 + IMPUTED_INDICATOR_COLS


def extract_top5_dataframe(
    shap_matrix: np.ndarray,
    feature_names: List[str],
    meta_df: pd.DataFrame,
    feature_matrix: Optional[np.ndarray] = None,
    model_name: str = "model",
    k: int = 3,
) -> pd.DataFrame:
    """
    Vectorized extraction of Top-5 signed SHAP feature contributions per row.
    Ranks 1 to 5 are ordered by descending absolute impact |shap_value|.
    """
    n_rows, n_cols = shap_matrix.shape
    assert n_rows == len(meta_df), f"Row mismatch: {n_rows} vs {len(meta_df)}"
    assert n_cols == len(feature_names), f"Col mismatch: {n_cols} vs {len(feature_names)}"

    feat_arr = np.array(feature_names)
    abs_matrix = np.abs(shap_matrix)

    # 1. Partition top 5 largest absolute SHAP values
    k_top = min(5, n_cols)
    part_idx = np.argpartition(-abs_matrix, k_top, axis=1)[:, :k_top]

    # 2. Sort the top 5 per row in exact descending order of absolute value
    row_idx = np.arange(n_rows)[:, None]
    part_vals = abs_matrix[row_idx, part_idx]
    order = np.argsort(-part_vals, axis=1)
    top5_cols = np.take_along_axis(part_idx, order, axis=1)

    top5_shap = np.take_along_axis(shap_matrix, top5_cols, axis=1)
    top5_feats = feat_arr[top5_cols]

    ranks = np.tile(np.arange(1, k_top + 1), (n_rows, 1))

    id_students = np.repeat(meta_df["id_student"].values, k_top)
    weeks = np.repeat(meta_df["week"].values, k_top)

    data_dict = {
        "id_student": id_students.astype(np.int64),
        "week": weeks.astype(np.int32),
        "rank": ranks.ravel().astype(np.int8),
        "feature": top5_feats.ravel(),
        "shap_value": top5_shap.ravel().astype(np.float32),
        "model": model_name,
        "k": k,
    }

    if feature_matrix is not None:
        top5_fvals = np.take_along_axis(feature_matrix, top5_cols, axis=1)
        data_dict["feature_value"] = top5_fvals.ravel().astype(np.float32)

    return pd.DataFrame(data_dict)


def compute_lgbm_shap(
    X_val: pd.DataFrame,
    meta_val: pd.DataFrame,
    model_path: str = "models/baselines/lgbm_champion_v3_official.txt",
) -> pd.DataFrame:
    """Compute LightGBM TreeSHAP contributions."""
    with open(model_path, "rb") as f:
        clean_str = f.read().replace(b"\r\n", b"\n").decode("utf-8")
    booster = lgb.Booster(model_str=clean_str)

    # LightGBM returns [N, n_features + 1], last column is bias
    contribs = booster.predict(X_val, pred_contrib=True)
    shap_vals = contribs[:, :len(RAW_FEATURES_38)]

    return extract_top5_dataframe(
        shap_matrix=shap_vals,
        feature_names=RAW_FEATURES_38,
        meta_df=meta_val,
        feature_matrix=X_val.values,
        model_name="lgbm_v3",
        k=3,
    )


def compute_xgb_shap(
    X_val: pd.DataFrame,
    meta_val: pd.DataFrame,
    model_path: str = "models/zoo/xgb_seed42.json",
) -> pd.DataFrame:
    """Compute XGBoost TreeSHAP contributions."""
    booster = xgb.Booster()
    booster.load_model(model_path)
    dm = xgb.DMatrix(X_val.values, feature_names=RAW_FEATURES_38)

    # XGBoost returns [N, n_features + 1], last column is bias
    contribs = booster.predict(dm, pred_contribs=True)
    shap_vals = contribs[:, :len(RAW_FEATURES_38)]

    return extract_top5_dataframe(
        shap_matrix=shap_vals,
        feature_names=RAW_FEATURES_38,
        meta_df=meta_val,
        feature_matrix=X_val.values,
        model_name="xgb",
        k=3,
    )


def compute_catboost_shap(
    X_val: pd.DataFrame,
    meta_val: pd.DataFrame,
    model_path: str = "models/zoo/catboost_seed42.cbm",
) -> pd.DataFrame:
    """Compute CatBoost TreeSHAP contributions."""
    cb = CatBoostClassifier()
    cb.load_model(model_path)
    pool = Pool(X_val.values, feature_names=RAW_FEATURES_38)

    # CatBoost returns [N, n_features + 1], last column is bias
    contribs = cb.get_feature_importance(pool, type="ShapValues")
    shap_vals = contribs[:, :len(RAW_FEATURES_38)]

    return extract_top5_dataframe(
        shap_matrix=shap_vals,
        feature_names=RAW_FEATURES_38,
        meta_df=meta_val,
        feature_matrix=X_val.values,
        model_name="catboost",
        k=3,
    )


def compute_rf_shap(
    X_val: pd.DataFrame,
    meta_val: pd.DataFrame,
    model_path: str = "models/zoo/rf_seed42.joblib",
    preprocessor_path: str = "models/zoo/preprocess_imputed.joblib",
) -> pd.DataFrame:
    """Compute Random Forest TreeSHAP contributions."""
    rf = joblib.load(model_path)
    preprocessor: ImputedZooPreprocessor = joblib.load(preprocessor_path)
    X_val_imp = preprocessor.transform(X_val)

    explainer = shap.TreeExplainer(rf)
    # Use fast path-dependent tree attribution on the 43 imputed features
    shap_vals_43 = explainer.shap_values(X_val_imp, approximate=True)

    # If binary classification returns list or 3D array [N, features, 2]
    if isinstance(shap_vals_43, list):
        shap_vals_43 = shap_vals_43[1]
    elif shap_vals_43.ndim == 3:
        shap_vals_43 = shap_vals_43[:, :, 1]

    return extract_top5_dataframe(
        shap_matrix=shap_vals_43,
        feature_names=IMPUTED_FEATURES_43,
        meta_df=meta_val,
        feature_matrix=X_val_imp,
        model_name="rf",
        k=3,
    )


def compute_lr_shap(
    X_val_39: pd.DataFrame,
    X_tr_39: pd.DataFrame,
    meta_val: pd.DataFrame,
    model_path: str = "models/baselines/logistic_regression_unweighted.joblib",
    scaler_path: str = "models/preprocessing/train_scaler.joblib",
) -> pd.DataFrame:
    """Compute Logistic Regression analytical SHAP contributions."""
    lr = joblib.load(model_path)
    scaler_dict = joblib.load(scaler_path)
    scaler = scaler_dict["scaler"]
    feature_names = scaler_dict["feature_names"]

    # Replicate train_lr.py preprocessing
    cols_with_na = [col for col in feature_names if X_tr_39[col].isna().any()]
    indicator_names = [f"missing_{c}" for c in cols_with_na]
    all_names = list(feature_names) + indicator_names

    va_indicators = X_val_39[cols_with_na].isna().astype(float).values
    train_medians = X_tr_39.median()
    X_va_imputed = X_val_39.fillna(train_medians).values
    X_va_scaled = scaler.transform(X_va_imputed)
    X_va_mat = np.hstack([X_va_scaled, va_indicators])

    # Analytical SHAP for linear model: phi_i = coef_i * (x_i - E[x_i])
    # Features are StandardScaled (mean 0 on train), indicators have train mean E[z_i]
    tr_indicators = X_tr_39[cols_with_na].isna().astype(float).values
    ind_means = tr_indicators.mean(axis=0)

    E_x = np.zeros(X_va_mat.shape[1], dtype=np.float32)
    E_x[len(feature_names):] = ind_means

    coef = lr.coef_[0]
    shap_vals = (X_va_mat - E_x) * coef

    return extract_top5_dataframe(
        shap_matrix=shap_vals,
        feature_names=all_names,
        meta_df=meta_val,
        feature_matrix=X_va_mat,
        model_name="lr_39feat_unweighted",
        k=3,
    )
