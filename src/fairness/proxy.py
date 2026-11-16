"""
Demographic Proxy Analysis Module for OULAD.

Evaluates:
1. How well the model's input features and behavioral indicators predict each sensitive demographic attribute
   (5-fold learner-grouped cross-validation).
2. Spearman rank correlation between model predicted risk scores and IMD deprivation bands.
Provides clear evidence on whether the model inadvertently captures demographic proxies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.fairness.audit import load_audit_data, load_config

logger = logging.getLogger(__name__)


def build_learner_feature_summary(
    df_audit: pd.DataFrame,
    info_parquet_path: str = "data/interim/studentInfo.parquet",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct learner-level feature matrix combining sanctioned static features
    and aggregated temporal behavior / risk signals.
    """
    # 1. Behavioral summary from prediction trajectories
    learner_keys = ["code_module", "code_presentation", "id_student"]
    behav_summary = df_audit.groupby(learner_keys).agg(
        mean_risk=("y_prob", "mean"),
        max_risk=("y_prob", "max"),
        min_risk=("y_prob", "min"),
        std_risk=("y_prob", "std"),
        tenure_weeks=("week", "count"),
        first_week=("week", "min"),
        last_week=("week", "max"),
    ).reset_index()
    behav_summary["std_risk"] = behav_summary["std_risk"].fillna(0.0)

    # 2. Join static features from studentInfo if available
    info_path = Path(info_parquet_path)
    if info_path.exists():
        df_info = pq.read_table(info_path).to_pandas()
        static_cols = ["code_module", "code_presentation", "id_student", "studied_credits", "num_of_prev_attempts"]
        static_df = df_info[[c for c in static_cols if c in df_info.columns]].drop_duplicates(subset=learner_keys)
        merged = behav_summary.merge(static_df, on=learner_keys, how="left")
    else:
        merged = behav_summary

    # One-hot encode course module
    if "code_module" in merged.columns:
        module_dummies = pd.get_dummies(merged["code_module"], prefix="module", dtype=float)
        merged = pd.concat([merged, module_dummies], axis=1)

    feature_cols = [c for c in merged.columns if c not in learner_keys and c != "code_module"]
    return merged, feature_cols


def run_proxy_prediction(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate predictability of demographic attribute y using GroupKFold.
    Compares against a naive majority-class dummy classifier.
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    # Drop rows with null target
    valid_mask = y.notna()
    X_val = X[valid_mask].copy()
    y_val = y[valid_mask].copy()
    groups_val = groups[valid_mask].copy()

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y_val.astype(str))
    n_classes = len(le.classes_)

    if n_classes < 2:
        return {
            "cv_macro_f1": 0.0,
            "cv_balanced_acc": 1.0,
            "dummy_macro_f1": 0.0,
            "dummy_balanced_acc": 1.0,
            "f1_lift_over_dummy": 1.0,
        }

    preds_model = np.zeros(len(y_enc), dtype=int)
    preds_dummy = np.zeros(len(y_enc), dtype=int)

    for train_idx, test_idx in gkf.split(X_val, y_enc, groups=groups_val):
        X_tr, X_te = X_val.iloc[train_idx], X_val.iloc[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr.fillna(0.0))
        X_te_sc = scaler.transform(X_te.fillna(0.0))

        # Model: Linear Logistic Regression
        clf = LogisticRegression(max_iter=500, random_state=seed, class_weight="balanced")
        clf.fit(X_tr_sc, y_tr)
        preds_model[test_idx] = clf.predict(X_te_sc)

        # Baseline: Dummy stratified / majority
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_tr_sc, y_tr)
        preds_dummy[test_idx] = dummy.predict(X_te_sc)

    f1_m = float(f1_score(y_enc, preds_model, average="macro", zero_division=0))
    bal_acc_m = float(balanced_accuracy_score(y_enc, preds_model))

    f1_d = float(f1_score(y_enc, preds_dummy, average="macro", zero_division=0))
    bal_acc_d = float(balanced_accuracy_score(y_enc, preds_dummy))

    lift = float(f1_m / f1_d) if f1_d > 0 else 1.0

    return {
        "cv_macro_f1": f1_m,
        "cv_balanced_acc": bal_acc_m,
        "dummy_macro_f1": f1_d,
        "dummy_balanced_acc": bal_acc_d,
        "f1_lift_over_dummy": lift,
    }


def compute_imd_correlation(
    df_audit: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute Spearman rank correlation of model predicted risk with IMD deprivation band.
    """
    imd_rank_map = {
        "0-10%": 1,
        "10-20": 2,
        "10-20%": 2,
        "20-30%": 3,
        "30-40%": 4,
        "40-50%": 5,
        "50-60%": 6,
        "60-70%": 7,
        "70-80%": 8,
        "80-90%": 9,
        "90-100%": 10,
    }

    df_valid = df_audit.dropna(subset=["imd_band"]).copy()
    df_valid["imd_rank"] = df_valid["imd_band"].map(imd_rank_map)

    # Filter known ranks
    df_ranked = df_valid[df_valid["imd_rank"].notna()]
    if len(df_ranked) == 0:
        return {"spearman_rho": 0.0, "p_value": 1.0, "n_obs": 0}

    rho, pval = spearmanr(df_ranked["imd_rank"], df_ranked["y_prob"])
    
    # Also compute learner-level correlation
    learner_df = df_ranked.groupby("id_student").agg({
        "imd_rank": "first",
        "y_prob": "mean",
    })
    learner_rho, learner_pval = spearmanr(learner_df["imd_rank"], learner_df["y_prob"])

    return {
        "row_spearman_rho": float(rho),
        "row_p_value": float(pval),
        "row_n_obs": len(df_ranked),
        "learner_spearman_rho": float(learner_rho),
        "learner_p_value": float(learner_pval),
        "learner_n_obs": len(learner_df),
    }


def run_proxy_analysis(
    pred_path: Union[str, Path] = "predictions/dropout/k3/lgbm_v3_val.parquet",
    demographics_path: str = "data/processed/audit_demographics.parquet",
    output_dir: str = "reports/fairness",
    config_path: str = "config/app.yaml",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run complete proxy analysis and save reports/fairness/proxy_analysis.csv.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_audit = load_audit_data(pred_path, demographics_path=demographics_path)
    model_name = str(df_audit["model"].iloc[0])

    # 1. IMD Correlation
    imd_stats = compute_imd_correlation(df_audit)

    # 2. Predictability of each demographic attribute
    learner_features_df, feature_cols = build_learner_feature_summary(df_audit)

    # Merge demographics on learner level
    learner_keys = ["code_module", "code_presentation", "id_student"]
    demo_df = df_audit[learner_keys + ["gender", "age_band", "highest_education", "imd_band", "disability", "region"]].drop_duplicates(subset=learner_keys)
    full_learner_df = learner_features_df.merge(demo_df, on=learner_keys, how="inner")

    X = full_learner_df[feature_cols]
    groups = full_learner_df["id_student"]

    results = []
    attributes = ["gender", "age_band", "highest_education", "imd_band", "disability", "region"]

    for attr in attributes:
        if attr not in full_learner_df.columns:
            continue
        y = full_learner_df[attr]
        res = run_proxy_prediction(X, y, groups)
        res["model"] = model_name
        res["attribute"] = attr
        res["feature_count"] = len(feature_cols)
        res["features_used"] = ", ".join(feature_cols)
        results.append(res)

    results_df = pd.DataFrame(results)
    
    # Append IMD stats summary
    results_df["imd_spearman_rho"] = imd_stats["learner_spearman_rho"]
    results_df["imd_spearman_pval"] = imd_stats["learner_p_value"]

    out_file = out_dir / "proxy_analysis.csv"
    results_df.to_csv(out_file, index=False)
    logger.info(f"Saved proxy analysis to {out_file}")

    return results_df, imd_stats
