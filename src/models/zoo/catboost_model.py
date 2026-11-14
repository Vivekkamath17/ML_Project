"""
CatBoost model wrapper, Optuna tuning, and multi-seed training for Phase 3 Model Zoo.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import catboost as cb
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.harness import evaluate_model, learner_group_cv, log_run, paired_bootstrap
from src.models.predictions import write_predictions
from src.models.zoo.common import (
    compute_median_inner_iterations,
    get_model_size_mb,
    measure_inference_latency_ms_per_1k,
)


def tune_catboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    meta_tr: pd.DataFrame,
    n_trials: int = 30,
    seed: int = 42,
    moderate_pos_weight: float = 5.0,
    study_save_path: str = "models/zoo/catboost_optuna_study.joblib",
    use_gpu: bool = True,
) -> Tuple[Dict[str, Any], int, float, optuna.Study]:
    """
    Tune CatBoost hyperparameters with Optuna over 30 trials using 5-fold inner GroupKFold on Train.
    Early stopping inside inner folds selects optimal iteration count.
    """
    groups = meta_tr["id_student"].to_numpy()
    folds = list(learner_group_cv(X_tr, y_tr, groups=groups, n_splits=5))

    trial_fold_iterations: Dict[int, List[int]] = {}

    task_type = "GPU" if use_gpu else "CPU"

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("depth", 3, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 20.0)
        random_strength = trial.suggest_float("random_strength", 0.1, 10.0)
        use_cw = trial.suggest_categorical("use_class_weight", [False, True])

        class_weights = [1.0, moderate_pos_weight] if use_cw else None

        params: Dict[str, Any] = {
            "iterations": 800,
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "random_strength": random_strength,
            "eval_metric": "PRAUC",
            "early_stopping_rounds": 50,
            "random_seed": seed,
            "verbose": 0,
            "task_type": task_type,
        }
        if class_weights is not None:
            params["class_weights"] = class_weights
        if not use_gpu:
            params["thread_count"] = -1

        fold_prs = []
        best_iters = []

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            X_f_tr, y_f_tr = X_tr.iloc[tr_idx], y_tr[tr_idx]
            X_f_va, y_f_val = X_tr.iloc[val_idx], y_tr[val_idx]

            model = cb.CatBoostClassifier(**params)
            model.fit(
                X_f_tr,
                y_f_tr,
                eval_set=(X_f_va, y_f_val),
                verbose=False,
            )

            preds = model.predict_proba(X_f_va)[:, 1]
            pr = average_precision_score(y_f_val, preds)
            fold_prs.append(pr)
            best_iters.append(int(model.get_best_iteration()) + 1)

        trial_fold_iterations[trial.number] = best_iters
        return float(np.mean(fold_prs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_trial_num = study.best_trial.number
    best_fold_iters = trial_fold_iterations.get(best_trial_num, [100, 100, 100, 100, 100])
    chosen_n_estimators = compute_median_inner_iterations(best_fold_iters)

    Path(study_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, study_save_path)

    best_params = study.best_params.copy()
    best_params["n_estimators"] = chosen_n_estimators
    best_params["inner_cv_pr_auc"] = study.best_value

    return best_params, chosen_n_estimators, study.best_value, study


def fit_and_evaluate_catboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    meta_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    meta_va: pd.DataFrame,
    best_params: Dict[str, Any],
    chosen_n_estimators: int,
    seeds: List[int],
    models_dir: str = "models/zoo",
    moderate_pos_weight: float = 5.0,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Fit CatBoost on full Train across 5 seeds using chosen parameters and fixed iteration count.
    Predict on Val (2014B), evaluate with evaluate_model, and run paired bootstrap vs LightGBM v3.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    use_cw = best_params.get("use_class_weight", False)
    class_weights = [1.0, moderate_pos_weight] if use_cw else None
    task_type = "GPU" if use_gpu else "CPU"

    base_params: Dict[str, Any] = {
        "iterations": chosen_n_estimators,
        "depth": best_params["depth"],
        "learning_rate": best_params["learning_rate"],
        "l2_leaf_reg": best_params["l2_leaf_reg"],
        "random_strength": best_params["random_strength"],
        "eval_metric": "PRAUC",
        "verbose": 0,
        "task_type": task_type,
    }
    if class_weights is not None:
        base_params["class_weights"] = class_weights
    if not use_gpu:
        base_params["thread_count"] = -1

    seed_val_probs: List[np.ndarray] = []
    seed_rocs: List[float] = []
    seed_prs: List[float] = []
    seed_briers: List[float] = []
    seed_train_times: List[float] = []
    model_files: List[Path] = []
    first_model = None

    for s in seeds:
        params_s = base_params.copy()
        params_s["random_seed"] = s

        t0 = time.perf_counter()
        cb_m = cb.CatBoostClassifier(**params_s)
        cb_m.fit(X_tr, y_tr, verbose=False)
        t_fit = time.perf_counter() - t0
        seed_train_times.append(t_fit)

        m_file = models_path / f"catboost_seed{s}.cbm"
        cb_m.save_model(str(m_file))
        model_files.append(m_file)

        if first_model is None:
            first_model = cb_m

        probs = cb_m.predict_proba(X_va)[:, 1]
        seed_val_probs.append(probs)

        roc = roc_auc_score(y_va, probs)
        pr = average_precision_score(y_va, probs)
        brier = float(np.mean((probs - y_va) ** 2))
        seed_rocs.append(roc)
        seed_prs.append(pr)
        seed_briers.append(brier)

        log_run(
            run_name=f"tree/catboost/default/seed{s}",
            params={
                **best_params,
                "n_estimators": chosen_n_estimators,
                "seed": s,
                "k": 3,
                "target": "label_dropout_k3",
            },
            metrics={"roc_auc": roc, "pr_auc": pr, "brier_score": brier, "fit_time_sec": t_fit},
            feature_list=list(X_tr.columns),
            experiment_name="dropout",
        )

    # Headline seed mean predictions
    headline_probs = np.mean(seed_val_probs, axis=0)

    # Measure inference latency on 1000 rows
    sample_df = X_va.iloc[:1000]
    latency_ms_per_1k = measure_inference_latency_ms_per_1k(
        lambda data: first_model.predict_proba(data)[:, 1], sample_df
    )
    model_size_mb = get_model_size_mb(model_files[0])

    # Save validation predictions
    df_preds = meta_va[["id_student", "code_module", "code_presentation", "week"]].copy()
    df_preds["model"] = "catboost"
    df_preds["k"] = 3
    df_preds["target"] = "label_dropout_k3"
    df_preds["split"] = "val"
    df_preds["y_true"] = y_va
    df_preds["y_prob"] = headline_probs

    pred_path = write_predictions(
        df_preds,
        metadata={
            "best_params": best_params,
            "n_estimators": chosen_n_estimators,
            "seed_mean_roc": float(np.mean(seed_rocs)),
            "seed_mean_pr": float(np.mean(seed_prs)),
            "seed_std_pr": float(np.std(seed_prs)),
            "seed_mean_brier": float(np.mean(seed_briers)),
        },
    )

    # Feature importance from first seed model
    importances = first_model.get_feature_importance()
    total_imp = sum(importances)
    norm_importances = {
        col: imp / total_imp if total_imp > 0 else 0.0
        for col, imp in zip(X_tr.columns, importances)
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

    # Full evaluation metrics with cluster CIs
    eval_metrics = evaluate_model(
        y_true=y_va,
        y_prob=headline_probs,
        cluster_ids=meta_va["id_student"],
        threshold=0.035,
        n_resamples=500,
        seed=42,
    )

    return {
        "model_name": "CatBoost",
        "best_params": best_params,
        "chosen_n_estimators": chosen_n_estimators,
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
