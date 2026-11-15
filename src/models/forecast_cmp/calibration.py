"""Post-hoc probability calibration module for OULAD dropout prediction.

Calibrators:
  - Platt scaling (logistic regression on logits)
  - Isotonic regression
  - Temperature scaling (optimizing T via cross-entropy loss)

Protocol:
  - 5-fold GroupKFold by id_student strictly inside validation set for honest OOF evaluation.
  - Final calibrator per model/method fitted on all validation rows (saved for honest test split).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

from src.models.predictions import read_predictions, write_predictions


class TemperatureScaling:
    """Temperature scaling calibrator optimizing scale T on logits."""

    def __init__(self, eps: float = 1e-7) -> None:
        self.temperature: float = 1.0
        self.eps = eps

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        p = np.clip(y_prob, self.eps, 1.0 - self.eps)
        logits = np.log(p / (1.0 - p))

        def nll(t_arr: np.ndarray) -> float:
            t = t_arr[0]
            scaled = logits / max(t, 1e-3)
            # log-sigmoid
            p_adj = 1.0 / (1.0 + np.exp(-scaled))
            p_adj = np.clip(p_adj, self.eps, 1.0 - self.eps)
            return float(-np.mean(y_true * np.log(p_adj) + (1.0 - y_true) * np.log(1.0 - p_adj)))

        res = minimize(nll, x0=np.array([1.0]), bounds=[(0.05, 20.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        p = np.clip(y_prob, self.eps, 1.0 - self.eps)
        logits = np.log(p / (1.0 - p))
        scaled = logits / max(self.temperature, 1e-3)
        return 1.0 / (1.0 + np.exp(-scaled))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error using equal-frequency (quantile) bins."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_prob, quantiles)
    bin_edges[-1] += 1e-7

    ece = 0.0
    n = len(y_prob)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high) if i < n_bins - 1 else (y_prob >= low) & (y_prob <= high)
        n_bin = np.sum(mask)
        if n_bin > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (n_bin / n) * abs(bin_acc - bin_conf)
    return float(ece)


def learner_bootstrap_brier_delta(
    y_true: np.ndarray,
    y_before: np.ndarray,
    y_after: np.ndarray,
    cluster_ids: np.ndarray,
    n_resamples: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute paired Brier score change (after - before) with 95% learner-cluster bootstrap CIs."""
    delta_point = float(brier_score_loss(y_true, y_after) - brier_score_loss(y_true, y_before))
    unique_clusters, inv_indices = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)

    cluster_to_rows: List[np.ndarray] = [[] for _ in range(n_clusters)]
    for row_idx, c_idx in enumerate(inv_indices):
        cluster_to_rows[c_idx].append(row_idx)
    cluster_to_rows_arr = [np.array(r, dtype=np.int64) for r in cluster_to_rows]

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        rows = np.concatenate([cluster_to_rows_arr[c] for c in sampled])
        b_after = brier_score_loss(y_true[rows], y_after[rows])
        b_before = brier_score_loss(y_true[rows], y_before[rows])
        deltas[b] = b_after - b_before

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return delta_point, float(ci_low), float(ci_high)
