"""
Simple Baselines module for Phase 2:
1. Majority / prior baseline (predicts train empirical base rate).
2. Recency-only Logistic Regression (days_since_last_activity).
3. One-feature baselines for key feature groups.
Logs all results to MLflow under experiment 'dropout'.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.data import load_split
from src.models.metrics import evaluate_predictions
from src.models.tracking import log_model_experiment


def run_simple_baselines(
    k: int = 3,
    min_week: int = 2,
    drop_never_active: bool = True,
    n_resamples: int = 500,
) -> Dict[str, Any]:
    """Execute all simple baselines on validation split."""
    print("=" * 60)
    print("RUNNING STEP 2: SIMPLE BASELINES")
    print("=" * 60)

    # 1. Load Train and Val splits
    X_train, y_train, meta_train = load_split("train", k=k, min_week=min_week, drop_never_active=drop_never_active)
    X_val, y_val, meta_val = load_split("val", k=k, min_week=min_week, drop_never_active=drop_never_active)

    # Learner cluster keys for bootstrap
    learner_keys_val = meta_val["code_module"] + "_" + meta_val["code_presentation"] + "_" + meta_val["id_student"].astype(str)

    results = {}

    # BASELINE 1: Majority / Prior Baseline
    print("\n[1/3] Evaluating Majority / Prior Baseline...")
    train_prior = float(y_train.mean())
    prior_probs_val = np.full_like(y_val, fill_value=train_prior, dtype=float)
    prior_eval = evaluate_predictions(y_val, prior_probs_val, learner_keys=learner_keys_val, n_resamples=n_resamples)

    results["majority_prior"] = {
        "name": "Majority Prior",
        "point": prior_eval["point"],
        "ci": prior_eval["ci"],
    }
    log_model_experiment(
        experiment_name="dropout",
        run_name="baseline_majority_prior",
        params={"model": "prior", "train_prior": train_prior, "k": k, "min_week": min_week},
        metrics={
            "roc_auc": prior_eval["point"]["roc_auc"],
            "pr_auc": prior_eval["point"]["pr_auc"],
            "brier": prior_eval["point"]["brier"],
            "mcc": prior_eval["point"]["mcc"],
        },
        feature_list=[],
    )

    # BASELINE 2: Recency-Only Logistic Regression
    print("\n[2/3] Evaluating Recency-Only Logistic Regression (days_since_last_activity)...")
    rec_feat = "days_since_last_activity"
    X_tr_rec = X_train[[rec_feat]].values
    X_va_rec = X_val[[rec_feat]].values

    clf_rec = LogisticRegression(random_state=42, max_iter=1000)
    clf_rec.fit(X_tr_rec, y_train)
    rec_probs_val = clf_rec.predict_proba(X_va_rec)[:, 1]

    rec_eval = evaluate_predictions(y_val, rec_probs_val, learner_keys=learner_keys_val, n_resamples=n_resamples)
    results["recency_only_lr"] = {
        "name": "Recency-Only LR",
        "point": rec_eval["point"],
        "ci": rec_eval["ci"],
    }
    log_model_experiment(
        experiment_name="dropout",
        run_name="baseline_recency_only_lr",
        params={"model": "logistic_regression", "feature": rec_feat, "k": k, "min_week": min_week},
        metrics={
            "roc_auc": rec_eval["point"]["roc_auc"],
            "pr_auc": rec_eval["point"]["pr_auc"],
            "pr_auc_mult": rec_eval["point"]["pr_auc_mult"],
            "brier": rec_eval["point"]["brier"],
            "mcc": rec_eval["point"]["mcc"],
        },
        feature_list=[rec_feat],
    )

    # BASELINE 3: One-Feature Baselines across feature groups
    print("\n[3/3] Evaluating One-Feature Baselines...")
    one_features = {
        "behavior_weekly_clicks": "weekly_clicks",
        "assessment_weighted_score": "weighted_score_to_date",
        "curriculum_pct_resources": "pct_resources_visited",
        "temporal_wow_clicks": "wow_change_clicks",
        "static_prev_attempts": "num_of_prev_attempts",
    }

    for name, feat in one_features.items():
        # Impute single train median for missing values (e.g. assessments before submission)
        med = float(X_train[feat].median())
        tr_vec = X_train[feat].fillna(med).values.reshape(-1, 1)
        va_vec = X_val[feat].fillna(med).values.reshape(-1, 1)

        clf_single = LogisticRegression(random_state=42, max_iter=1000)
        clf_single.fit(tr_vec, y_train)
        probs = clf_single.predict_proba(va_vec)[:, 1]

        ev = evaluate_predictions(y_val, probs, learner_keys=learner_keys_val, n_resamples=n_resamples)
        results[name] = {
            "name": f"Single: {feat}",
            "point": ev["point"],
            "ci": ev["ci"],
        }
        log_model_experiment(
            experiment_name="dropout",
            run_name=f"baseline_one_feat_{feat}",
            params={"model": "logistic_regression", "feature": feat, "k": k, "min_week": min_week},
            metrics={
                "roc_auc": ev["point"]["roc_auc"],
                "pr_auc": ev["point"]["pr_auc"],
                "pr_auc_mult": ev["point"]["pr_auc_mult"],
                "brier": ev["point"]["brier"],
                "mcc": ev["point"]["mcc"],
            },
            feature_list=[feat],
        )

    return results


if __name__ == "__main__":
    res = run_simple_baselines(n_resamples=100)
    for k, v in res.items():
        p = v["point"]
        c = v["ci"]
        print(f"{v['name']:<28} | ROC-AUC: {p['roc_auc']:.4f} [{c['roc_auc'][0]:.4f}, {c['roc_auc'][1]:.4f}] | PR-AUC: {p['pr_auc']:.4f} ({p['pr_auc_mult']:.2f}x) | Brier: {p['brier']:.4f}")
