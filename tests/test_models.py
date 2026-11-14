"""
Unit tests for Phase 2 Modeling:
1. Metrics correctness against scikit-learn on toy data.
2. Data loader filters: no censored rows, no week < min_week, no never-active rows when excluded.
3. Strict security guard: load_split('test') raises ValueError unless allow_test=True.
4. Feature cleanliness: X contains exactly the 39 sanctioned columns and zero forbidden columns.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.models.data import load_split, load_feature_manifest
from src.models.metrics import compute_point_metrics, compute_bootstrap_ci


def test_metrics_against_sklearn():
    """Verify compute_point_metrics matches sklearn on synthetic data."""
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.1, size=500)
    y_prob = rng.uniform(0, 1, size=500)

    res = compute_point_metrics(y_true, y_prob, threshold=0.5)

    sk_roc = roc_auc_score(y_true, y_prob)
    sk_pr = average_precision_score(y_true, y_prob)
    sk_brier = brier_score_loss(y_true, y_prob)

    assert abs(res["roc_auc"] - sk_roc) < 1e-6
    assert abs(res["pr_auc"] - sk_pr) < 1e-6
    assert abs(res["brier"] - sk_brier) < 1e-6
    assert abs(res["base_rate"] - y_true.mean()) < 1e-6
    assert abs(res["pr_auc_mult"] - (sk_pr / y_true.mean())) < 1e-6


def test_test_split_security_guard():
    """Verify that requesting split='test' without allow_test=True raises ValueError."""
    with pytest.raises(ValueError, match="CRITICAL SECURITY GUARD"):
        load_split("test")

    with pytest.raises(ValueError, match="CRITICAL SECURITY GUARD"):
        load_split("test", allow_test=False)


def test_data_loader_filters_and_sanctioned_features():
    """
    Verify that load_split respects min_week, non-censored condition,
    drop_never_active, and returns ONLY the 39 sanctioned columns.
    """
    X_val, y_val, meta_val = load_split("val", k=3, min_week=2, drop_never_active=True)

    # 1. Feature counts & columns
    manifest_39 = load_feature_manifest()
    assert list(X_val.columns) == manifest_39
    assert len(X_val.columns) == 39

    # Forbidden columns check
    forbidden = [
        "code_module",
        "gender",
        "final_result",
        "final_score",
        "grade_band",
        "label_dropout_k3",
        "never_active_learner",
        "is_zero_click_withdrawal",
        "label_is_proxy",
        "last_vle_date",
    ]
    for col in forbidden:
        assert col not in X_val.columns

    # 2. Row filtering checks
    assert (meta_val["week"] >= 2).all(), "Found week < min_week!"
    assert not meta_val["never_active_learner"].any(), "Found never_active_learner when drop_never_active=True!"
    assert len(X_val) == len(y_val)
    assert len(X_val) == len(meta_val)
    assert set(np.unique(y_val)).issubset({0, 1})


def test_bootstrap_cluster_ci_bounds():
    """Verify cluster bootstrap produces valid confidence intervals."""
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.1, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    learners = pd.Series([f"L_{i // 4}" for i in range(200)])

    cis = compute_bootstrap_ci(y_true, y_prob, learners, n_resamples=50, random_seed=42)

    assert "roc_auc" in cis
    assert "pr_auc" in cis
    low_roc, high_roc = cis["roc_auc"]
    assert 0.0 <= low_roc <= high_roc <= 1.0
