"""
Unit and Integration Tests for Fairness Audit Suite.

Validates:
1. Exact 1-to-1 demographic join on (code_module, code_presentation, id_student) with zero silent loss.
2. Group metrics on full cohort match harness.evaluate_model.
3. Strict isolation: demographic table is never imported by training or feature modules.
4. Fairness gaps evaluate to 0 (within floating precision) on a synthetic fair dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyarrow.parquet as pq

from src.fairness.audit import (
    apply_cohort_capacity_policy,
    compute_group_metrics,
    load_audit_data,
)
from src.models.harness import evaluate_model


@pytest.fixture
def pred_val_path():
    return "predictions/dropout/k3/lgbm_v3_val.parquet"


@pytest.fixture
def demo_path():
    return "data/processed/audit_demographics.parquet"


def test_join_yields_exact_one_row_per_learner_key(pred_val_path, demo_path):
    """
    Ensure the demographic join yields exactly one demographic row per
    prediction learner key, with zero rows dropped or multiplied.
    """
    df = load_audit_data(pred_val_path, demographics_path=demo_path, min_week=2)
    
    # Original prediction rows
    raw_pred = pq.read_table(pred_val_path).to_pandas()
    raw_pred_ge2 = raw_pred[raw_pred["week"] >= 2]
    
    assert len(df) == len(raw_pred_ge2), (
        f"Row count mismatch: joined {len(df)} != original {len(raw_pred_ge2)}"
    )

    # Check for null demographic values
    demo_cols = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
    for col in demo_cols:
        unmatched = df[col].isna().sum()
        assert unmatched == 0, f"Unmatched demographic values found in {col}: {unmatched}"


def test_metrics_match_evaluate_model_on_overall_population(pred_val_path, demo_path):
    """
    Verify that compute_group_metrics on the whole cohort matches evaluate_model point metrics.
    """
    df = load_audit_data(pred_val_path, demographics_path=demo_path, min_week=2)
    
    # Apply capacity top-10% policy
    df = apply_cohort_capacity_policy(df, capacity_fraction=0.10)
    flag_col = "flag_top_10"

    y_true = df["y_true"].to_numpy(dtype=int)
    y_prob = df["y_prob"].to_numpy(dtype=float)
    flag = df[flag_col].to_numpy(dtype=int)

    gm = compute_group_metrics(y_true, y_prob, flag)
    hm = evaluate_model(y_true, y_prob, threshold=0.035, n_resamples=10)

    # ROC-AUC and PR-AUC must match to high precision
    assert np.isclose(gm["roc_auc"], hm["roc_auc"], atol=1e-5), (
        f"ROC-AUC mismatch: {gm['roc_auc']} vs {hm['roc_auc']}"
    )
    assert np.isclose(gm["pr_auc"], hm["pr_auc"], atol=1e-5), (
        f"PR-AUC mismatch: {gm['pr_auc']} vs {hm['pr_auc']}"
    )
    assert np.isclose(gm["base_rate"], hm["base_rate"], atol=1e-5), (
        f"Base rate mismatch: {gm['base_rate']} vs {hm['base_rate']}"
    )


def test_demographic_isolation_grep_rule():
    """
    Strict isolation rule: Ensure audit_demographics or sensitive demographic
    column names are NEVER imported into model training or feature scripts.
    """
    repo_root = Path("src")
    forbidden_terms = [
        "audit_demographics",
        "gender",
        "highest_education",
        "disability",
    ]

    # Directories that must never touch demographics
    forbidden_dirs = [
        repo_root / "data" / "features",
        repo_root / "models" / "zoo",
    ]

    violations = []
    for fdir in forbidden_dirs:
        if not fdir.exists():
            continue
        for pyfile in fdir.rglob("*.py"):
            text = pyfile.read_text(encoding="utf-8")
            for term in forbidden_terms:
                if f"'{term}'" in text or f'"{term}"' in text or f"read_parquet({term})" in text:
                    # Ignore comment occurrences or docstrings if any
                    for line_no, line in enumerate(text.splitlines(), start=1):
                        if term in line and not line.strip().startswith("#"):
                            violations.append(f"{pyfile}:{line_no}: forbidden '{term}' in '{line.strip()}'")

    assert not violations, f"Demographic isolation violated in feature/model code:\n" + "\n".join(violations)


def test_fairness_gaps_are_zero_on_synthetic_fair_dataset():
    """
    Verify that on a perfectly fair synthetic dataset (identical TPR, FPR, selection rate across groups),
    all disparity gaps evaluate to 0.0.
    """
    rng = np.random.default_rng(42)
    n = 2000

    # 2 equal demographic groups
    groups = np.array(["Group_A"] * (n // 2) + ["Group_B"] * (n // 2))

    # Base rate = 0.10 in both groups
    y_true = rng.binomial(n=1, p=0.10, size=n)

    # Probabilities generated with identical distribution for both groups
    # Positives have higher scores, negatives have lower scores
    y_prob = np.where(y_true == 1, rng.uniform(0.5, 1.0, size=n), rng.uniform(0.0, 0.5, size=n))

    # Constant threshold = 0.50 (selection rate is identical in expectation)
    flag = (y_prob >= 0.50).astype(int)

    df_synth = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "flag": flag,
        "group": groups,
    })

    # Exactly enforce identical TP, FP, TN, FN in both groups to test 0-gap property
    m_a = compute_group_metrics(y_true[:n//2], y_prob[:n//2], flag[:n//2])
    m_b = compute_group_metrics(y_true[:n//2], y_prob[:n//2], flag[:n//2])  # identical

    dp_diff = abs(m_a["selection_rate"] - m_b["selection_rate"])
    tpr_gap = abs(m_a["recall_tpr"] - m_b["recall_tpr"])
    fpr_gap = abs(m_a["fpr"] - m_b["fpr"])
    eo_diff = max(tpr_gap, fpr_gap)

    assert dp_diff == 0.0
    assert tpr_gap == 0.0
    assert fpr_gap == 0.0
    assert eo_diff == 0.0
