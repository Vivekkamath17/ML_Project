"""
Random Forest model for Phase 3 Model Zoo.
Uses 25% learner-grouped subsampling for Optuna tuning,
and trains across 5 seeds on full Train for final evaluation on Val.
Also fits a default-configuration Random Forest on full Train to measure tuning impact.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.harness import evaluate_model, learner_group_cv, log_run, paired_bootstrap
from src.models.predictions import write_predictions
from src.models.zoo.common import (
    get_imputed_pipeline,
    get_learner_subsample_mask,
    get_model_size_mb,
    measure_inference_latency_ms_per_1k,
)


def get_rf_estimator(
    params: Dict[str, Any],
    seed: int,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    cw = {0: 1.0, 1: 5.0} if params.get("use_class_weight", False) else None

    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=cw,
        random_state=seed,
        n_jobs=n_jobs,
    )


def tune_bagging(
    model_type: str = "rf",
    X_tr_imp: np.ndarray = None,
    y_tr: np.ndarray = None,
    meta_tr: pd.DataFrame = None,
    n_trials: int = 30,
    seed: int = 42,
    subsample_ratio: float = 0.25,
    study_save_path: str | None = None,
) -> Tuple[Dict[str, Any], float, optuna.Study]:
    """
    Tune Random Forest hyperparameters with Optuna over 30 trials using 5-fold inner GroupKFold
    on a 25% learner-grouped subsample of Train.
    """
    if study_save_path is None:
        study_save_path = "models/zoo/rf_optuna_study.joblib"

    # Deterministic learner-grouped 25% subsample
    sub_mask = get_learner_subsample_mask(meta_tr, fraction=subsample_ratio, seed=seed)
    X_sub = X_tr_imp[sub_mask]
    y_sub = y_tr[sub_mask]
    meta_sub = meta_tr.iloc[sub_mask].reset_index(drop=True)

    groups = meta_sub["id_student"].to_numpy()
    folds = list(learner_group_cv(pd.DataFrame(X_sub), y_sub, groups=groups, n_splits=5))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
            "max_features": trial.suggest_float("max_features", 0.3, 0.8),
            "use_class_weight": trial.suggest_categorical("use_class_weight", [False, True]),
        }

        fold_prs = []
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            clf = get_rf_estimator(params, seed=seed, n_jobs=-1)
            clf.fit(X_sub[tr_idx], y_sub[tr_idx])
            preds = clf.predict_proba(X_sub[val_idx])[:, 1]
            pr = average_precision_score(y_sub[val_idx], preds)
            fold_prs.append(pr)

        return float(np.mean(fold_prs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    Path(study_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, study_save_path)

    best_params = study.best_params.copy()
    best_params["inner_cv_pr_auc"] = study.best_value
    best_params["subsample_ratio"] = subsample_ratio

    return best_params, study.best_value, study


def fit_and_evaluate_bagging(
    model_type: str = "rf",
    X_tr_imp: np.ndarray = None,
    y_tr: np.ndarray = None,
    meta_tr: pd.DataFrame = None,
    X_va_imp: np.ndarray = None,
    y_va: np.ndarray = None,
    meta_va: pd.DataFrame = None,
    feature_names: List[str] = None,
    best_params: Dict[str, Any] = None,
    seeds: List[int] = None,
    models_dir: str = "models/zoo",
) -> Dict[str, Any]:
    """
    Fit Random Forest on FULL Train across 5 seeds using the chosen parameters.
    Predict on Val (2014B), evaluate with evaluate_model, and run paired bootstrap vs LightGBM v3.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    model_label = "Random Forest"

    seed_val_probs: List[np.ndarray] = []
    seed_rocs: List[float] = []
    seed_prs: List[float] = []
    seed_briers: List[float] = []
    seed_train_times: List[float] = []
    model_files: List[Path] = []
    first_model = None

    for s in seeds:
        t0 = time.perf_counter()
        clf = get_rf_estimator(best_params, seed=s, n_jobs=-1)
        clf.fit(X_tr_imp, y_tr)
        t_fit = time.perf_counter() - t0
        seed_train_times.append(t_fit)

        m_file = models_path / f"rf_seed{s}.joblib"
        joblib.dump(clf, m_file)
        model_files.append(m_file)

        if first_model is None:
            first_model = clf

        probs = clf.predict_proba(X_va_imp)[:, 1]
        seed_val_probs.append(probs)

        roc = roc_auc_score(y_va, probs)
        pr = average_precision_score(y_va, probs)
        brier = float(np.mean((probs - y_va) ** 2))
        seed_rocs.append(roc)
        seed_prs.append(pr)
        seed_briers.append(brier)

        log_run(
            run_name=f"bagging/rf/default/seed{s}",
            params={
                **best_params,
                "seed": s,
                "k": 3,
                "target": "label_dropout_k3",
            },
            metrics={"roc_auc": roc, "pr_auc": pr, "brier_score": brier, "fit_time_sec": t_fit},
            feature_list=feature_names,
            experiment_name="dropout",
        )

    # Headline seed mean predictions
    headline_probs = np.mean(seed_val_probs, axis=0)

    # Measure latency on 1000 rows
    sample_data = X_va_imp[:1000]
    latency_ms_per_1k = measure_inference_latency_ms_per_1k(
        lambda data: first_model.predict_proba(data)[:, 1], sample_data
    )
    model_size_mb = get_model_size_mb(model_files[0])

    # Save validation predictions
    df_preds = meta_va[["id_student", "code_module", "code_presentation", "week"]].copy()
    df_preds["model"] = "rf"
    df_preds["k"] = 3
    df_preds["target"] = "label_dropout_k3"
    df_preds["split"] = "val"
    df_preds["y_true"] = y_va
    df_preds["y_prob"] = headline_probs

    pred_path = write_predictions(
        df_preds,
        metadata={
            "best_params": best_params,
            "seed_mean_roc": float(np.mean(seed_rocs)),
            "seed_mean_pr": float(np.mean(seed_prs)),
            "seed_std_pr": float(np.std(seed_prs)),
            "seed_mean_brier": float(np.mean(seed_briers)),
        },
    )

    # Feature importance (MDI) from first model
    importances = first_model.feature_importances_
    total_imp = sum(importances)
    norm_importances = {
        col: imp / total_imp if total_imp > 0 else 0.0
        for col, imp in zip(feature_names, importances)
    }
    sorted_importances = sorted(norm_importances.items(), key=lambda x: x[1], reverse=True)

    # Paired bootstrap vs LightGBM v3
    df_lgbm = pd.read_parquet("predictions/dropout/k3/lgbm_v3_val.parquet")
    boot_res = paired_bootstrap(
        y_true=y_va,
        y_prob_a=headline_probs,
        y_prob_b=df_lgbm["y_prob"].to_numpy(),
        cluster_ids=meta_va["id_student"],
        n_boot=500,
        seed=42,
    )

    eval_metrics = evaluate_model(
        y_true=y_va,
        y_prob=headline_probs,
        cluster_ids=meta_va["id_student"],
        threshold=0.035,
        n_resamples=500,
        seed=42,
    )

    return {
        "model_name": model_label,
        "best_params": best_params,
        "inner_cv_pr_auc": best_params.get("inner_cv_pr_auc", 0.0),
        "seed_rocs": seed_rocs,
        "seed_prs": seed_prs,
        "seed_briers": seed_briers,
        "seed_mean_roc": float(np.mean(seed_rocs)),
        "seed_std_roc": float(np.std(seed_rocs)),
        "seed_mean_pr": float(np.mean(seed_prs)),
        "seed_std_pr": float(np.std(seed_prs)),
        "seed_mean_brier": float(np.mean(seed_briers)),
        "train_time_sec": float(np.mean(seed_train_times)),
        "latency_ms_per_1k": latency_ms_per_1k,
        "model_size_mb": model_size_mb,
        "eval_metrics": eval_metrics,
        "paired_vs_lgbm": boot_res,
        "top_features": sorted_importances[:10],
        "pred_path": str(pred_path),
    }


def fit_default_rf_comparison(
    X_tr_imp: np.ndarray,
    y_tr: np.ndarray,
    meta_tr: pd.DataFrame,
    X_va_imp: np.ndarray,
    y_va: np.ndarray,
    meta_va: pd.DataFrame,
    feature_names: List[str],
    seeds: List[int],
) -> Dict[str, Any]:
    """
    Fit default-configuration Random Forest (min_samples_leaf=100, max_depth=12, n_estimators=200, max_features=0.5)
    on full Train across 5 seeds to show whether 25% subsample tuning helped or hurt.
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 100,
        "max_features": 0.5,
        "use_class_weight": False,
    }

    seed_prs: List[float] = []
    seed_rocs: List[float] = []
    seed_probs: List[np.ndarray] = []

    for s in seeds:
        clf = get_rf_estimator(default_params, seed=s, n_jobs=-1)
        clf.fit(X_tr_imp, y_tr)
        probs = clf.predict_proba(X_va_imp)[:, 1]
        seed_probs.append(probs)
        seed_prs.append(float(average_precision_score(y_va, probs)))
        seed_rocs.append(float(roc_auc_score(y_va, probs)))

    headline_probs = np.mean(seed_probs, axis=0)

    return {
        "default_params": default_params,
        "seed_prs": seed_prs,
        "seed_rocs": seed_rocs,
        "seed_mean_pr": float(np.mean(seed_prs)),
        "seed_std_pr": float(np.std(seed_prs)),
        "seed_mean_roc": float(np.mean(seed_rocs)),
        "headline_pr": float(average_precision_score(y_va, headline_probs)),
        "headline_roc": float(roc_auc_score(y_va, headline_probs)),
    }

