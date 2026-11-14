"""
Shared Model Evaluation Harness for OULAD Pipeline.

Exposes standard evaluation, clustered paired bootstrap, learner-grouped cross-validation,
Optuna hyperparameter tuning, and MLflow run logging for team use.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from src.models.tracking import log_model_experiment, setup_mlflow


def evaluate_model(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute standard model evaluation metrics.

    Parameters:
    -----------
    y_true: array-like of binary ground truth labels (0 or 1).
    y_prob: array-like of predicted probabilities in [0.0, 1.0].
    threshold: optional float decision threshold. If provided, computes classification metrics.

    Returns:
    --------
    Dict[str, float] with 'roc_auc', 'pr_auc', 'brier_score',
    and optionally 'precision', 'recall', 'f1', 'mcc'.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)}, len(y_prob)={len(y_p)}")

    roc = float(roc_auc_score(y_t, y_p))
    pr = float(average_precision_score(y_t, y_p))
    brier = float(brier_score_loss(y_t, y_p))

    metrics = {
        "roc_auc": roc,
        "pr_auc": pr,
        "brier_score": brier,
    }

    if threshold is not None:
        y_pred = (y_p >= threshold).astype(int)
        metrics["precision"] = float(precision_score(y_t, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_t, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_t, y_pred, zero_division=0))
        metrics["mcc"] = float(matthews_corrcoef(y_t, y_pred))

    return metrics


def paired_bootstrap(
    y_true: Union[np.ndarray, pd.Series],
    y_prob_a: Union[np.ndarray, pd.Series],
    y_prob_b: Union[np.ndarray, pd.Series],
    cluster_ids: Union[np.ndarray, pd.Series, list],
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute paired differences (Model A minus Model B) with 95% bootstrap confidence intervals
    clustered by student identifier (cluster_ids must be id_student alone).

    Parameters:
    -----------
    y_true: binary labels.
    y_prob_a: predictions from Model A.
    y_prob_b: predictions from Model B.
    cluster_ids: cluster identifiers (id_student) for each row.
    n_boot: number of bootstrap resamples (default 500).
    seed: random seed for reproducibility.

    Returns:
    --------
    Dict containing:
      'delta_roc': float,
      'delta_roc_ci': (float, float),
      'delta_pr': float,
      'delta_pr_ci': (float, float),
      'n_boot': int,
      'n_clusters': int
    """
    y_t = np.asarray(y_true).astype(int)
    p_a = np.asarray(y_prob_a).astype(float)
    p_b = np.asarray(y_prob_b).astype(float)
    clusters = np.asarray(cluster_ids)

    n_rows = len(y_t)
    if not (len(p_a) == len(p_b) == len(clusters) == n_rows):
        raise ValueError("All inputs must have identical length.")

    # Base point estimates
    point_roc_a = roc_auc_score(y_t, p_a)
    point_roc_b = roc_auc_score(y_t, p_b)
    point_delta_roc = float(point_roc_a - point_roc_b)

    point_pr_a = average_precision_score(y_t, p_a)
    point_pr_b = average_precision_score(y_t, p_b)
    point_delta_pr = float(point_pr_a - point_pr_b)

    # Index rows by cluster
    unique_clusters, inv_indices = np.unique(clusters, return_inverse=True)
    n_clusters = len(unique_clusters)

    cluster_to_rows: List[np.ndarray] = [[] for _ in range(n_clusters)]
    for row_idx, c_idx in enumerate(inv_indices):
        cluster_to_rows[c_idx].append(row_idx)
    cluster_to_rows_arr = [np.array(r, dtype=np.int64) for r in cluster_to_rows]

    rng = np.random.default_rng(seed)
    boot_deltas_roc = np.empty(n_boot, dtype=np.float64)
    boot_deltas_pr = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_c_indices = rng.integers(0, n_clusters, size=n_clusters)
        sampled_rows = np.concatenate([cluster_to_rows_arr[c] for c in sampled_c_indices])

        y_boot = y_t[sampled_rows]
        pa_boot = p_a[sampled_rows]
        pb_boot = p_b[sampled_rows]

        # In case all labels in sample are identical
        if len(np.unique(y_boot)) < 2:
            boot_deltas_roc[b] = point_delta_roc
            boot_deltas_pr[b] = point_delta_pr
        else:
            roc_a = roc_auc_score(y_boot, pa_boot)
            roc_b = roc_auc_score(y_boot, pb_boot)
            boot_deltas_roc[b] = roc_a - roc_b

            pr_a = average_precision_score(y_boot, pa_boot)
            pr_b = average_precision_score(y_boot, pb_boot)
            boot_deltas_pr[b] = pr_a - pr_b

    ci_low_roc, ci_high_roc = np.percentile(boot_deltas_roc, [2.5, 97.5])
    ci_low_pr, ci_high_pr = np.percentile(boot_deltas_pr, [2.5, 97.5])

    return {
        "delta_roc": point_delta_roc,
        "delta_roc_ci": (float(ci_low_roc), float(ci_high_roc)),
        "delta_pr": point_delta_pr,
        "delta_pr_ci": (float(ci_low_pr), float(ci_high_pr)),
        "n_boot": n_boot,
        "n_clusters": n_clusters,
    }


def learner_group_cv(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[np.ndarray, pd.Series],
    groups: Union[np.ndarray, pd.Series, list],
    n_splits: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator yielding train/val index splits grouped strictly by learner identifier (id_student).
    Guarantees zero student overlap between folds.

    Parameters:
    -----------
    X: feature matrix
    y: target array
    groups: learner IDs (id_student)
    n_splits: number of folds (default 5)

    Yields:
    -------
    Tuple[np.ndarray, np.ndarray]: (train_indices, val_indices)
    """
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(X, y, groups=groups)


def tune_with_optuna(
    objective_fn: Callable[[optuna.Trial], float],
    n_trials: int = 30,
    seed: int = 42,
    direction: str = "maximize",
) -> optuna.Study:
    """
    Standard wrapper for hyperparameter tuning with Optuna.
    Configures deterministic TPESampler with the specified seed.

    Parameters:
    -----------
    objective_fn: callable taking an optuna.Trial and returning the optimization metric.
    n_trials: number of search trials (default 30).
    seed: random seed for the TPESampler.
    direction: 'maximize' or 'minimize'.

    Returns:
    --------
    optuna.Study object.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials)
    return study


def log_run(
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_list: Optional[List[str]] = None,
    artifacts: Optional[Dict[str, str]] = None,
    experiment_name: str = "oulad_baselines",
) -> str:
    """
    Standardized MLflow logger logging parameters, metrics, and artifacts to SQLite database.

    Parameters:
    -----------
    model_name: descriptive name for the model run.
    params: dictionary of hyperparameter or setup values.
    metrics: dictionary of evaluation metrics (roc_auc, pr_auc, etc.).
    feature_list: list of feature names used by the model.
    artifacts: optional dict of artifact names to file paths.
    experiment_name: MLflow experiment name (default 'oulad_baselines').

    Returns:
    --------
    str: MLflow run_id.
    """
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    setup_mlflow()
    feats = feature_list or list(params.keys())
    return log_model_experiment(
        experiment_name=experiment_name,
        run_name=model_name,
        params=params,
        metrics=metrics,
        feature_list=feats,
        artifacts=artifacts,
    )
