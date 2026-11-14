"""
Comprehensive Metrics and Cluster-Bootstrap module for OULAD Modeling.
Computes:
- ROC-AUC, PR-AUC, base rate, PR-AUC multiple of base rate
- Precision, recall, F1, MCC, confusion matrix (TP, FP, FN, TN)
- Brier score
- Top-k% metrics: precision@top-5%, recall@top-5%, precision@top-10%, recall@top-10%, lift@top-10%
- Fast learner-cluster bootstrap 95% CIs via joblib parallel threads.
"""
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
)
from joblib import Parallel, delayed


def compute_point_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute point metrics for binary classification.
    Handles edge cases (all single class, NaN predictions, uniform predictions).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    base_rate = float(np.mean(y_true))
    n_pos = int(np.sum(y_true))
    n_total = len(y_true)

    # ROC-AUC & PR-AUC
    if n_pos == 0 or n_pos == n_total:
        roc_auc = 0.5
        pr_auc = base_rate
    else:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = base_rate

    pr_auc_mult = float(pr_auc / base_rate) if base_rate > 0 else 0.0

    # Binary classification metrics at threshold
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Top-k% capacity metrics
    def top_k_metrics(k_pct: float) -> Tuple[float, float, float]:
        n_top = max(1, int(np.ceil(n_total * k_pct)))
        top_idx = np.argsort(-y_prob)[:n_top]
        n_pos_in_top = int(np.sum(y_true[top_idx]))
        prec_at_k = float(n_pos_in_top / n_top)
        rec_at_k = float(n_pos_in_top / n_pos) if n_pos > 0 else 0.0
        lift_at_k = float(prec_at_k / base_rate) if base_rate > 0 else 0.0
        return prec_at_k, rec_at_k, lift_at_k

    p_top5, r_top5, _ = top_k_metrics(0.05)
    p_top10, r_top10, l_top10 = top_k_metrics(0.10)

    return {
        "base_rate": base_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "pr_auc_mult": pr_auc_mult,
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "brier": brier,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision_at_5pct": p_top5,
        "recall_at_5pct": r_top5,
        "precision_at_10pct": p_top10,
        "recall_at_10pct": r_top10,
        "lift_at_10pct": l_top10,
    }


def _one_resample(
    b: int,
    all_sampled: np.ndarray,
    codes: np.ndarray,
    n_clusters: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    metrics_to_ci: List[str],
    sorted_y_desc: np.ndarray,
    sorted_codes_desc: np.ndarray,
) -> Optional[Dict[str, float]]:
    """Single bootstrap resample worker (called in parallel)."""
    counts = np.bincount(all_sampled[b], minlength=n_clusters)
    w = counts[codes]
    sw_pos = float(np.dot(w.astype(np.float64), y_true.astype(np.float64)))
    sw_tot = float(w.sum())
    if sw_pos == 0.0 or sw_pos == sw_tot:
        return None

    out: Dict[str, float] = {}

    if "roc_auc" in metrics_to_ci:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob, sample_weight=w))
        except Exception:
            out["roc_auc"] = 0.5

    if "pr_auc" in metrics_to_ci:
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_prob, sample_weight=w))
        except Exception:
            out["pr_auc"] = 0.0

    if "brier" in metrics_to_ci:
        out["brier"] = float(brier_score_loss(y_true, y_prob, sample_weight=w))

    if "mcc" in metrics_to_ci:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred, sample_weight=w))

    if "precision_at_10pct" in metrics_to_ci or "recall_at_10pct" in metrics_to_ci:
        sorted_w = counts[sorted_codes_desc].astype(np.float64)
        cum_w    = np.cumsum(sorted_w)
        cutoff   = int(np.searchsorted(cum_w, cum_w[-1] * 0.10, side="left"))
        w_top    = float(cum_w[cutoff])
        pos_top  = float(np.dot(sorted_y_desc[: cutoff + 1].astype(np.float64), sorted_w[: cutoff + 1]))
        prec_10  = pos_top / w_top  if w_top  > 0 else 0.0
        rec_10   = pos_top / sw_pos if sw_pos > 0 else 0.0
        if "precision_at_10pct" in metrics_to_ci:
            out["precision_at_10pct"] = prec_10
        if "recall_at_10pct" in metrics_to_ci:
            out["recall_at_10pct"] = rec_10

    return out


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    learner_keys: pd.Series,
    threshold: float = 0.5,
    n_resamples: int = 500,
    random_seed: int = 42,
    metrics_to_ci: Optional[List[str]] = None,
    n_jobs: int = 4,
) -> Dict[str, Tuple[float, float]]:
    """
    Cluster-bootstrap 95% CIs resampled by learner key.
    Uses joblib parallel threads for ~6x speedup over sequential weighted calls.
    """
    if metrics_to_ci is None:
        metrics_to_ci = ["roc_auc", "pr_auc", "mcc", "brier", "precision_at_10pct", "recall_at_10pct"]

    if n_resamples <= 0:
        return {m: (0.0, 0.0) for m in metrics_to_ci}

    t0 = time.time()
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    codes, uniques = pd.factorize(learner_keys)
    n_clusters = len(uniques)

    # Pre-sort by descending prob for weighted top-k metrics
    desc_idx          = np.argsort(-y_prob)
    sorted_y_desc     = y_true[desc_idx]
    sorted_codes_desc = codes[desc_idx]

    rng = np.random.default_rng(random_seed)
    all_sampled = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_resample)(
            b, all_sampled, codes, n_clusters,
            y_true, y_prob, y_pred,
            metrics_to_ci,
            sorted_y_desc, sorted_codes_desc,
        )
        for b in range(n_resamples)
    )
    results = [r for r in results if r is not None]

    ci_results: Dict[str, Tuple[float, float]] = {}
    for m in metrics_to_ci:
        vals = np.array([r[m] for r in results if m in r])
        if len(vals) == 0:
            ci_results[m] = (0.0, 0.0)
        else:
            ci_results[m] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    elapsed = time.time() - t0
    ci_results["_runtime_sec"] = (elapsed, elapsed)
    return ci_results


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    learner_keys: pd.Series,
    threshold: float = 0.5,
    n_resamples: int = 500,
    random_seed: int = 42,
    n_jobs: int = 4,
) -> Dict[str, Any]:
    """
    Combined evaluation returning point metrics + 95% cluster-bootstrap CIs.
    """
    point = compute_point_metrics(y_true, y_prob, threshold=threshold)
    cis = compute_bootstrap_ci(
        y_true,
        y_prob,
        learner_keys=learner_keys,
        threshold=threshold,
        n_resamples=n_resamples,
        random_seed=random_seed,
        n_jobs=n_jobs,
    )
    return {
        "point": point,
        "ci": cis,
    }
