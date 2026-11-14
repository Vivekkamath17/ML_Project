"""
Feature importance agreement and permutation importance for Phase 3 Model Zoo.
Permutation importance is evaluated on a fixed validation sample (5,000 rows, seed=42)
to ensure strictly comparable importance rankings across tree, bagging, linear, and neural models.
Computes Spearman rank correlation across all model pairs and identifies consensus features.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_permutation_importance(
    predict_fn: Callable[[np.ndarray | pd.DataFrame], np.ndarray],
    X_val_sample: np.ndarray | pd.DataFrame,
    y_val_sample: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 3,
    seed: int = 42,
    metric: str = "pr_auc",
) -> Dict[str, float]:
    """
    Compute permutation feature importance on a fixed validation sample.
    Metric: drop in PR-AUC (or ROC-AUC) when feature is randomly shuffled.
    """
    rng = np.random.default_rng(seed)
    is_df = isinstance(X_val_sample, pd.DataFrame)

    # Baseline score
    baseline_preds = predict_fn(X_val_sample)
    if metric == "pr_auc":
        baseline_score = average_precision_score(y_val_sample, baseline_preds)
    else:
        baseline_score = roc_auc_score(y_val_sample, baseline_preds)

    importances: Dict[str, float] = {}

    for col_idx, col_name in enumerate(feature_names):
        drops = []
        for r in range(n_repeats):
            if is_df:
                X_perm = X_val_sample.copy()
                perm_idx = rng.permutation(len(X_perm))
                X_perm[col_name] = X_perm[col_name].iloc[perm_idx].values
            else:
                X_perm = X_val_sample.copy()
                perm_idx = rng.permutation(len(X_perm))
                X_perm[:, col_idx] = X_perm[perm_idx, col_idx]

            perm_preds = predict_fn(X_perm)
            if metric == "pr_auc":
                score = average_precision_score(y_val_sample, perm_preds)
            else:
                score = roc_auc_score(y_val_sample, perm_preds)

            drops.append(baseline_score - score)

        importances[col_name] = float(np.mean(drops))

    return importances


def compute_importance_rank_correlation(
    importance_dict_a: Dict[str, float],
    importance_dict_b: Dict[str, float],
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation between two feature importance dictionaries
    over shared features.
    """
    shared_features = sorted(set(importance_dict_a.keys()).intersection(set(importance_dict_b.keys())))
    if len(shared_features) < 3:
        return 0.0, 1.0

    ranks_a = [importance_dict_a[f] for f in shared_features]
    ranks_b = [importance_dict_b[f] for f in shared_features]

    corr, pval = spearmanr(ranks_a, ranks_b)
    return float(corr), float(pval)


def find_consensus_features(
    model_importances: Dict[str, Dict[str, float]],
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Identify consensus features across all models by computing average normalized rank.
    """
    feature_ranks: Dict[str, List[float]] = {}

    for model_name, imp_map in model_importances.items():
        sorted_feats = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)
        n = len(sorted_feats)
        for rank, (feat, _) in enumerate(sorted_feats):
            normalized_rank = 1.0 - (rank / n)  # 1.0 = top feature, 0.0 = lowest
            if feat not in feature_ranks:
                feature_ranks[feat] = []
            feature_ranks[feat].append(normalized_rank)

    mean_ranks = {feat: float(np.mean(ranks)) for feat, ranks in feature_ranks.items()}
    sorted_consensus = sorted(mean_ranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_consensus[:top_n]
