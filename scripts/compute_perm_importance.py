"""
Compute permutation feature importance for Phase 3 models on a fixed
5,000-row validation subsample (seed=42, 3 repeats, metric=pr_auc).

Models covered: LightGBM v3, XGBoost, CatBoost, Random Forest
Output: reports/phase3/permutation_importance.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml

from src.models.zoo.common import load_zoo_data, ImputedZooPreprocessor
from src.models.zoo.importance import (
    compute_permutation_importance,
    compute_importance_rank_correlation,
    find_consensus_features,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_SIZE = 5_000
SEED = 42
N_REPEATS = 3
METRIC = "pr_auc"
OUT_PATH = Path("reports/phase3/permutation_importance.json")

# Feature groups (from config/features.yaml, minus n_skipped_resources)
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

FEATURE_GROUPS = {
    "static":     ["num_of_prev_attempts", "studied_credits", "registration_lead_time",
                   "precourse_clicks", "course_progress"],
    "behavior":   ["weekly_clicks", "days_since_last_activity", "weekend_click_ratio",
                   "active_days_entropy", "share_forumng", "share_quiz", "share_resource",
                   "share_oucontent", "share_homepage", "share_subpage", "share_url",
                   "share_other_types"],
    "temporal":   ["wow_change_clicks", "wow_change_active_days", "rolling_mean_clicks_2w",
                   "rolling_mean_clicks_4w", "rolling_mean_active_days_2w",
                   "rolling_mean_active_days_4w", "clicks_slope_4w", "clicks_ewma",
                   "engagement_decay_rate"],
    "assessment": ["n_submitted", "n_missed", "mean_score_submitted", "last_score",
                   "score_slope", "mean_lateness", "weighted_score_to_date", "n_banked",
                   "banked_score_mean"],
    "curriculum": ["pct_resources_visited", "pacing_vs_cohort_median", "prerequisite_gap"],
}

# Imputed feature names (38 raw + 5 missing indicators)
IMPUTED_INDICATOR_COLS = [
    "mean_score_submitted_is_missing",
    "last_score_is_missing",
    "score_slope_is_missing",
    "mean_lateness_is_missing",
    "banked_score_mean_is_missing",
]
IMPUTED_FEATURES_43 = RAW_FEATURES_38 + IMPUTED_INDICATOR_COLS


def group_importances(importance_dict: Dict[str, float]) -> Dict[str, float]:
    """Sum importances by feature group (only over raw 38 features)."""
    group_imp: Dict[str, float] = {}
    for group_name, feats in FEATURE_GROUPS.items():
        vals = [importance_dict.get(f, 0.0) for f in feats]
        group_imp[group_name] = float(np.sum(vals))
    return group_imp


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading zoo data …")
X_tr_raw, y_tr, meta_tr, X_va_raw, y_va, meta_va, feat_names = load_zoo_data(
    k=3, min_week=2
)
assert feat_names == RAW_FEATURES_38, f"Feature mismatch: {feat_names}"

# Fixed 5k val subsample (reproducible, sorted by id_student then index)
rng = np.random.default_rng(SEED)
sub_idx = rng.choice(len(y_va), size=min(SAMPLE_SIZE, len(y_va)), replace=False)
sub_idx = np.sort(sub_idx)

# X_va_raw is a DataFrame from load_zoo_data; use iloc for row indexing
X_va_raw_df_sub = X_va_raw.iloc[sub_idx].reset_index(drop=True)   # DataFrame with feature names (for LightGBM)
X_va_raw_sub = X_va_raw_df_sub.to_numpy()                           # numpy array (for XGBoost, CatBoost)
y_va_sub = y_va[sub_idx]
print(f"Val subsample: {len(y_va_sub)} rows, positive rate = {y_va_sub.mean():.4f}")

# Imputed subsample (for RF and MLP — 43 features)
preprocessor: ImputedZooPreprocessor = joblib.load("models/zoo/preprocess_imputed.joblib")
X_va_imp_all = preprocessor.transform(X_va_raw)
X_va_imp_sub = X_va_imp_all[sub_idx]

results: Dict[str, Any] = {}

# ── 1. LightGBM v3 ────────────────────────────────────────────────────────────
print("\n[1/5] LightGBM v3 permutation importance …")
import lightgbm as lgb

lgbm_model = lgb.Booster(model_file="models/baselines/lgbm_champion_v3_official.txt")

def lgbm_predict(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=RAW_FEATURES_38)
    return lgbm_model.predict(X)

t0 = time.perf_counter()
lgbm_imp = compute_permutation_importance(
    lgbm_predict, X_va_raw_df_sub, y_va_sub, RAW_FEATURES_38, n_repeats=N_REPEATS, seed=SEED, metric=METRIC
)
print(f"  Done in {time.perf_counter()-t0:.1f}s")
print(f"  Top-5: {sorted(lgbm_imp.items(), key=lambda x: x[1], reverse=True)[:5]}")
results["lgbm_v3"] = {
    "feature_importance": lgbm_imp,
    "group_importance": group_importances(lgbm_imp),
}

# ── 2. XGBoost ────────────────────────────────────────────────────────────────
print("\n[2/5] XGBoost permutation importance …")
import xgboost as xgb

xgb_model = xgb.Booster()
xgb_model.load_model("models/zoo/xgb_seed42.json")
xgb_model.set_param("n_jobs", -1)

def xgb_predict(X: np.ndarray) -> np.ndarray:
    dm = xgb.DMatrix(X, feature_names=RAW_FEATURES_38)
    return xgb_model.predict(dm)

t0 = time.perf_counter()
xgb_imp = compute_permutation_importance(
    xgb_predict, X_va_raw_sub, y_va_sub, RAW_FEATURES_38, n_repeats=N_REPEATS, seed=SEED, metric=METRIC
)
print(f"  Done in {time.perf_counter()-t0:.1f}s")
print(f"  Top-5: {sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:5]}")
results["xgb"] = {
    "feature_importance": xgb_imp,
    "group_importance": group_importances(xgb_imp),
}

# ── 3. CatBoost ───────────────────────────────────────────────────────────────
print("\n[3/5] CatBoost permutation importance …")
from catboost import CatBoostClassifier

cb_model = CatBoostClassifier()
cb_model.load_model("models/zoo/catboost_seed42.cbm")

def cb_predict(X: np.ndarray) -> np.ndarray:
    return cb_model.predict_proba(X)[:, 1]

t0 = time.perf_counter()
cb_imp = compute_permutation_importance(
    cb_predict, X_va_raw_sub, y_va_sub, RAW_FEATURES_38, n_repeats=N_REPEATS, seed=SEED, metric=METRIC
)
print(f"  Done in {time.perf_counter()-t0:.1f}s")
print(f"  Top-5: {sorted(cb_imp.items(), key=lambda x: x[1], reverse=True)[:5]}")
results["catboost"] = {
    "feature_importance": cb_imp,
    "group_importance": group_importances(cb_imp),
}

# ── 4. Random Forest ──────────────────────────────────────────────────────────
print("\n[4/4] Random Forest permutation importance (43 imputed features) …")
rf_model = joblib.load("models/zoo/rf_seed42.joblib")

def rf_predict(X: np.ndarray) -> np.ndarray:
    return rf_model.predict_proba(X)[:, 1]

t0 = time.perf_counter()
rf_imp_43 = compute_permutation_importance(
    rf_predict, X_va_imp_sub, y_va_sub, IMPUTED_FEATURES_43, n_repeats=N_REPEATS, seed=SEED, metric=METRIC
)
print(f"  Done in {time.perf_counter()-t0:.1f}s")

# Collapse back to 38 raw features for comparison (zero out indicator-only entries)
rf_imp = {f: rf_imp_43.get(f, 0.0) for f in RAW_FEATURES_38}
print(f"  Top-5 (raw 38): {sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:5]}")
results["rf"] = {
    "feature_importance": rf_imp,
    "feature_importance_43": rf_imp_43,
    "group_importance": group_importances(rf_imp),
}

# ── Spearman rank correlations ─────────────────────────────────────────────────
print("\n--- Importance Spearman rank correlations ---")
model_keys = [k for k, v in results.items() if v is not None]
spearman_matrix: Dict[str, Dict[str, float]] = {}

for m1 in model_keys:
    spearman_matrix[m1] = {}
    for m2 in model_keys:
        if m1 == m2:
            spearman_matrix[m1][m2] = 1.0
        else:
            imp_a = results[m1]["feature_importance"]
            imp_b = results[m2]["feature_importance"]
            corr, pval = compute_importance_rank_correlation(imp_a, imp_b)
            spearman_matrix[m1][m2] = round(corr, 4)
            if m1 < m2:
                print(f"  {m1} vs {m2}: ρ={corr:.4f} (p={pval:.4f})")

# ── Consensus features ─────────────────────────────────────────────────────────
print("\n--- Consensus top-10 features ---")
all_imps = {k: results[k]["feature_importance"] for k in model_keys}
consensus = find_consensus_features(all_imps, top_n=10)
for feat, avg_rank in consensus:
    print(f"  {feat}: avg_normalized_rank={avg_rank:.3f}")

# ── Save output ────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
output = {
    "config": {
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "n_repeats": N_REPEATS,
        "metric": METRIC,
        "note": "Permutation importance on fixed 5k-row val subsample (seed=42). RF and MLP use 43 imputed features; trees use raw 38 features. Importance values collapsed to 38 raw features for cross-model comparison.",
    },
    "model_importances": {k: results[k] for k in model_keys},
    "spearman_rank_correlation": spearman_matrix,
    "consensus_features_top10": [
        {"feature": f, "avg_normalized_rank": r} for f, r in consensus
    ],
}
OUT_PATH.write_text(json.dumps(output, indent=2))
print(f"\nSaved to {OUT_PATH}")
