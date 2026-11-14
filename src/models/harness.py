"""
Shared Model Evaluation Harness for OULAD Modeling Pipeline.

Exposes:
- evaluate_model: wraps metrics.py for point metrics, top-k lift, and learner-bootstrap 95% CIs.
- paired_bootstrap: computes delta ROC and delta PR with 95% bootstrap CIs clustered by student ID.
- learner_group_cv: 5-fold cross-validation generator strictly grouped by id_student.
- tune_with_optuna: Optuna hyperparameter search wrapper with deterministic seed.
- log_run: MLflow logger enforcing naming schema <family>/<model>/<variant>/seed<N> and experiment validation.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.models.metrics import compute_bootstrap_ci, compute_point_metrics
from src.models.tracking import log_model_experiment, setup_mlflow


ALLOWED_EXPERIMENTS = {"dropout", "onset", "score", "grade_band", "calibration"}
RUN_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+/seed\d+$")


def evaluate_model(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    cluster_ids: Optional[Union[np.ndarray, pd.Series, list]] = None,
    threshold: float = 0.035,
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute full evaluation suite for binary classification: point metrics, lift@top-k,
    and optional learner-bootstrap 95% confidence intervals.

    Parameters:
    -----------
    y_true: binary ground truth labels (0 or 1).
    y_prob: predicted probabilities in [0.0, 1.0].
    cluster_ids: cluster key for bootstrap resampling. Must be id_student (or composite learner key).
                 If provided, computes 95% cluster-bootstrap CIs.
    threshold: decision threshold (default 0.035 for MCC-optimal baseline).
    n_resamples: bootstrap resamples (default 500).
    seed: random seed for bootstrap.

    Returns:
    --------
    Dict[str, Any] containing point metrics and (if cluster_ids provided) 'ci' dictionary.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_prob).astype(float)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)}, len(y_prob)={len(y_p)}")

    point = compute_point_metrics(y_t, y_p, threshold=threshold)

    result = {
        "roc_auc": point["roc_auc"],
        "pr_auc": point["pr_auc"],
        "brier_score": point["brier"],
        "base_rate": point["base_rate"],
        "pr_multiplier": point.get("pr_auc_mult", point["pr_auc"] / point["base_rate"] if point["base_rate"] > 0 else 1.0),
        "precision": point["precision"],
        "recall": point["recall"],
        "f1": point["f1"],
        "mcc": point["mcc"],
        "prec_at_top5": point.get("precision_at_5pct", 0.0),
        "recall_at_top5": point.get("recall_at_5pct", 0.0),
        "prec_at_top10": point.get("precision_at_10pct", 0.0),
        "recall_at_top10": point.get("recall_at_10pct", 0.0),
        "lift_at_top10": point.get("lift_at_10pct", 0.0),
    }

    if cluster_ids is not None:
        c_series = pd.Series(cluster_ids).astype(str)
        cis = compute_bootstrap_ci(
            y_t,
            y_p,
            learner_keys=c_series,
            threshold=threshold,
            n_resamples=n_resamples,
            random_seed=seed,
        )
        result["ci"] = cis

    return result


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
    clustered strictly by student identifier (cluster_ids = id_student).

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
    Dict containing delta_roc, delta_roc_ci, delta_pr, delta_pr_ci, n_boot, n_clusters.
    """
    y_t = np.asarray(y_true).astype(int)
    p_a = np.asarray(y_prob_a).astype(float)
    p_b = np.asarray(y_prob_b).astype(float)
    clusters = np.asarray(cluster_ids)

    n_rows = len(y_t)
    if not (len(p_a) == len(p_b) == len(clusters) == n_rows):
        raise ValueError("All inputs must have identical length.")

    from sklearn.metrics import average_precision_score, roc_auc_score

    point_roc_a = roc_auc_score(y_t, p_a)
    point_roc_b = roc_auc_score(y_t, p_b)
    point_delta_roc = float(point_roc_a - point_roc_b)

    point_pr_a = average_precision_score(y_t, p_a)
    point_pr_b = average_precision_score(y_t, p_b)
    point_delta_pr = float(point_pr_a - point_pr_b)

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
    Generator yielding 5-fold train/val splits strictly grouped by student identifier (id_student).
    Guarantees zero student overlap between folds.

    Parameters:
    -----------
    X: feature matrix
    y: target array
    groups: student IDs (id_student)
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
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials)
    return study


def log_run(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    tags: Optional[Dict[str, str]] = None,
    feature_list: Optional[List[str]] = None,
    artifacts: Optional[Dict[str, str]] = None,
    experiment_name: str = "dropout",
    tracking_uri: str = "sqlite:///mlruns.db",
) -> str:
    """
    Enforce naming standard and log run to MLflow SQLite backend.

    Naming convention:
      run_name must match <family>/<model>/<variant>/seed<N>
      Example: 'tree/lgbm/v3/seed42', 'sequence/lstm/bilstm/seed1'

    Experiment name must be one of:
      {'dropout', 'onset', 'score', 'grade_band', 'calibration'}

    Required tags:
      family, model, variant, seed, target, horizon
    """
    # 1. Enforce run name format
    if not RUN_NAME_REGEX.match(run_name):
        raise ValueError(
            f"Invalid run_name: '{run_name}'. "
            "Must match <family>/<model>/<variant>/seed<N> (e.g., 'tree/lgbm/v3/seed42')."
        )

    parts = run_name.split("/")
    family, model_val, variant, seed_str = parts[0], parts[1], parts[2], parts[3]
    seed_num = int(seed_str.replace("seed", ""))

    # 2. Enforce authorized experiment
    if experiment_name not in ALLOWED_EXPERIMENTS:
        raise ValueError(
            f"Invalid experiment_name: '{experiment_name}'. "
            f"Must be one of {ALLOWED_EXPERIMENTS}."
        )

    # 3. Required tags
    combined_tags = tags.copy() if tags else {}
    combined_tags.setdefault("family", family)
    combined_tags.setdefault("model", model_val)
    combined_tags.setdefault("variant", variant)
    combined_tags.setdefault("seed", str(seed_num))
    combined_tags.setdefault("target", str(params.get("target", "label_dropout_k3")))
    combined_tags.setdefault("horizon", str(params.get("k", 3)))

    for req_tag in ["family", "model", "variant", "seed", "target", "horizon"]:
        if req_tag not in combined_tags:
            raise ValueError(f"Missing required tag: '{req_tag}'")

    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    setup_mlflow(tracking_uri)
    feats = feature_list or list(params.keys())

    return log_model_experiment(
        experiment_name=experiment_name,
        run_name=run_name,
        params=params,
        metrics=metrics,
        feature_list=feats,
        artifacts=artifacts,
        tags=combined_tags,
    )
