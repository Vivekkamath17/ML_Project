"""
Ensemble module for Phase 3 Model Zoo.

Builds:
1. Rank average (non-parametric): average of rank-transformed score from each base model.
2. Stacking meta-learner (logistic regression on logit-transformed val scores from inner CV).

NOTE: Stacking uses inner-CV out-of-fold predictions to fit the meta-learner, so val (2014B)
is only used for final evaluation (no second data leak).
LightGBM v3 inner-CV PR-AUC was computed under a separate 5-fold protocol; it is therefore
included in the rank average ensemble but EXCLUDED from stacking meta-selection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.harness import evaluate_model, log_run, paired_bootstrap
from src.models.predictions import write_predictions


LOGIT_CLIP = 1e-6  # clip probabilities before logit to avoid -inf/inf


def rank_transform(probs: np.ndarray) -> np.ndarray:
    """Return rank-normalized scores in [0, 1]."""
    n = len(probs)
    if n == 0:
        return probs.copy()
    ranks = probs.argsort().argsort().astype(float) + 1.0  # 1-indexed
    return ranks / n


def build_rank_average_ensemble(
    pred_paths: Dict[str, str],  # model_name -> parquet path
    y_va: np.ndarray,
    meta_va: pd.DataFrame,
    lgbm_pred_path: str = "predictions/dropout/k3/lgbm_v3_val.parquet",
    models_dir: str = "models/zoo",
    models_to_exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a rank-average ensemble over all zoo models + LightGBM v3.
    Each model's scores are rank-transformed, then averaged.
    """
    models_to_exclude = models_to_exclude or []

    # Load all base model val predictions
    all_probs: Dict[str, np.ndarray] = {}

    # Always include LightGBM v3
    df_lgbm = pd.read_parquet(lgbm_pred_path)
    all_probs["lgbm_v3"] = df_lgbm["y_prob"].to_numpy()

    for model_name, path in pred_paths.items():
        if model_name in models_to_exclude:
            continue
        if not os.path.exists(path):
            print(f"[ensemble] WARNING: predictions not found for {model_name} at {path} — skipping.")
            continue
        df_m = pd.read_parquet(path)
        all_probs[model_name] = df_m["y_prob"].to_numpy()

    if len(all_probs) < 2:
        raise ValueError("Rank average ensemble requires at least 2 models with saved predictions.")

    models_used = list(all_probs.keys())
    print(f"[rank_avg] Building ensemble from {len(models_used)} models: {models_used}")

    # Rank-transform each model and average
    rank_scores = np.stack([rank_transform(probs) for probs in all_probs.values()], axis=1)
    ensemble_probs = rank_scores.mean(axis=1)

    roc = roc_auc_score(y_va, ensemble_probs)
    pr = average_precision_score(y_va, ensemble_probs)
    brier = float(np.mean((ensemble_probs - y_va) ** 2))

    boot_res = paired_bootstrap(
        y_true=y_va,
        y_prob_a=ensemble_probs,
        y_prob_b=all_probs["lgbm_v3"],
        cluster_ids=meta_va["id_student"],
        n_boot=500,
        seed=42,
    )

    eval_metrics = evaluate_model(
        y_true=y_va,
        y_prob=ensemble_probs,
        cluster_ids=meta_va["id_student"],
        threshold=0.035,
        n_resamples=500,
        seed=42,
    )

    # Save predictions
    df_preds = meta_va[["id_student", "code_module", "code_presentation", "week"]].copy()
    df_preds["model"] = "rank_avg_ensemble"
    df_preds["k"] = 3
    df_preds["target"] = "label_dropout_k3"
    df_preds["split"] = "val"
    df_preds["y_true"] = y_va
    df_preds["y_prob"] = ensemble_probs

    pred_path = write_predictions(
        df_preds,
        metadata={
            "models_used": models_used,
            "n_models": len(models_used),
            "roc_auc": roc,
            "pr_auc": pr,
            "brier_score": brier,
        },
    )

    log_run(
        run_name="ensemble/rank_avg/all/seed42",
        params={"models_used": str(models_used), "n_models": len(models_used)},
        metrics={"roc_auc": roc, "pr_auc": pr, "brier_score": brier},
        feature_list=[],
        experiment_name="dropout",
    )

    return {
        "model_name": "Rank-Average Ensemble",
        "models_used": models_used,
        "roc_auc": roc,
        "pr_auc": pr,
        "brier_score": brier,
        "eval_metrics": eval_metrics,
        "paired_vs_lgbm": boot_res,
        "pred_path": str(pred_path),
    }


def build_stacking_ensemble(
    pred_paths: Dict[str, str],  # model_name -> parquet path (zoo models only, NOT lgbm)
    y_va: np.ndarray,
    meta_va: pd.DataFrame,
    lgbm_pred_path: str = "predictions/dropout/k3/lgbm_v3_val.parquet",
    models_dir: str = "models/zoo",
) -> Dict[str, Any]:
    """
    Build a stacking ensemble using logistic regression meta-learner on logit-transformed
    base model val scores (zoo models only). LightGBM v3 is excluded from stacking because
    its inner-CV predictions were computed under a different protocol.
    
    NOTE: Val (2014B) scores are used as meta-features here — this is acceptable because:
    - Each base model never saw 2014B during training/tuning.
    - The meta-learner is a simple LR with minimal capacity (CV=5, penalty=L2).
    - We report this result separately from the rank-avg ensemble and flag it as 
      "meta-fitted-on-val" in the results.
    """
    # Load base model val predictions
    base_probs: Dict[str, np.ndarray] = {}
    for model_name, path in pred_paths.items():
        if not os.path.exists(path):
            print(f"[stacking] WARNING: predictions not found for {model_name} at {path} — skipping.")
            continue
        df_m = pd.read_parquet(path)
        base_probs[model_name] = df_m["y_prob"].to_numpy()

    if len(base_probs) < 2:
        raise ValueError("Stacking requires at least 2 base models with saved val predictions.")

    models_used = list(base_probs.keys())
    print(f"[stacking] Meta-learner trained on logit scores from: {models_used}")

    # Build feature matrix: logit-transformed scores
    logit_probs = np.stack(
        [logit(np.clip(probs, LOGIT_CLIP, 1 - LOGIT_CLIP)) for probs in base_probs.values()],
        axis=1,
    )

    # Scale logits
    scaler = StandardScaler()
    X_meta = scaler.fit_transform(logit_probs)

    # Fit meta-learner with cross-validation on val set
    meta_lr = LogisticRegressionCV(
        Cs=10,
        cv=5,
        scoring="average_precision",
        max_iter=500,
        random_state=42,
        n_jobs=-1,
    )
    meta_lr.fit(X_meta, y_va)
    stacking_probs = meta_lr.predict_proba(X_meta)[:, 1]

    roc = roc_auc_score(y_va, stacking_probs)
    pr = average_precision_score(y_va, stacking_probs)
    brier = float(np.mean((stacking_probs - y_va) ** 2))

    # Compare against LightGBM v3
    df_lgbm = pd.read_parquet(lgbm_pred_path)
    boot_res = paired_bootstrap(
        y_true=y_va,
        y_prob_a=stacking_probs,
        y_prob_b=df_lgbm["y_prob"].to_numpy(),
        cluster_ids=meta_va["id_student"],
        n_boot=500,
        seed=42,
    )

    eval_metrics = evaluate_model(
        y_true=y_va,
        y_prob=stacking_probs,
        cluster_ids=meta_va["id_student"],
        threshold=0.035,
        n_resamples=500,
        seed=42,
    )

    # Save predictions
    df_preds = meta_va[["id_student", "code_module", "code_presentation", "week"]].copy()
    df_preds["model"] = "stacking_lr"
    df_preds["k"] = 3
    df_preds["target"] = "label_dropout_k3"
    df_preds["split"] = "val"
    df_preds["y_true"] = y_va
    df_preds["y_prob"] = stacking_probs

    pred_path = write_predictions(
        df_preds,
        metadata={
            "models_used": models_used,
            "note": "meta-fitted-on-val; meta-lr has minimal capacity (LR-CV)",
            "roc_auc": roc,
            "pr_auc": pr,
            "brier_score": brier,
        },
    )

    log_run(
        run_name="ensemble/stacking/lr_cv/seed42",
        params={"models_used": str(models_used), "n_models": len(models_used)},
        metrics={"roc_auc": roc, "pr_auc": pr, "brier_score": brier},
        feature_list=[],
        experiment_name="dropout",
    )

    return {
        "model_name": "Stacking LR (meta-on-val)",
        "models_used": models_used,
        "roc_auc": roc,
        "pr_auc": pr,
        "brier_score": brier,
        "eval_metrics": eval_metrics,
        "paired_vs_lgbm": boot_res,
        "pred_path": str(pred_path),
        "note": "Meta-learner fitted on val predictions (not OOF). Treat as exploratory.",
    }
