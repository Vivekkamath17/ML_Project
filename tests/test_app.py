"""
Unit tests for Streamlit application modules, data loaders, and holdout guards.
"""
from pathlib import Path
import pytest
import pandas as pd
from app.data_loader import (
    discover_prediction_inventory,
    load_models_overview,
    load_curves_data,
    load_cohort_summary,
    load_policy_tradeoffs,
    load_by_week_metrics,
    load_by_module_metrics,
    load_cost_table,
    load_learner_index,
    load_learner_trajectory,
    load_shap_top5,
    load_score_forecasts,
    load_grade_band_forecasts,
    load_calibration_data,
    load_fairness_data,
)


def test_discover_prediction_inventory():
    """Verify that predictions inventory scans dynamically without hardcoding."""
    inv = discover_prediction_inventory()
    assert inv["files_count"] > 0
    assert "dropout" in inv["targets"]
    assert 3 in inv["horizons"]
    assert "val" in inv["splits"]
    # Verify test split is sealed / not unsealed prematurely
    assert inv["test_split_present"] is False


def test_load_models_overview_schema_and_values():
    """Verify models overview table contains required metrics with 95% CIs."""
    df = load_models_overview()
    assert not df.empty
    expected_cols = [
        "model_id", "display_name", "target", "horizon", "roc_auc",
        "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "pr_auc_ci_low",
        "pr_auc_ci_high", "brier", "delta_pr", "delta_roc", "statistically_tied_with_champion"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in models overview"
    assert "lgbm_v3" in df["model_id"].values
    assert len(df) >= 5


def test_curves_data_format():
    """Verify precomputed ROC and PR curves contain valid coordinates."""
    df = load_curves_data()
    assert not df.empty
    assert "x" in df.columns
    assert "y" in df.columns
    assert "curve_type" in df.columns
    assert set(df["curve_type"].unique()) == {"ROC", "PR"}
    # Coordinates must be bounded in [0, 1]
    assert df["x"].min() >= 0.0 and df["x"].max() <= 1.0
    assert df["y"].min() >= 0.0 and df["y"].max() <= 1.0


def test_cohort_denominators_accuracy():
    """Verify exact cohort denominators match validation ground truth."""
    cohort = load_cohort_summary()
    assert cohort["total_enrolled_learners"] == 6474
    assert cohort["total_eventual_withdrawn"] == 1395
    assert cohort["total_learner_weeks"] == 164799
    assert abs(cohort["week_base_rate"] - 0.02367) < 0.001


def test_policy_tradeoffs_range():
    """Verify policy simulation covers weekly capacity and probability cuts."""
    df = load_policy_tradeoffs()
    assert not df.empty
    assert "weekly_capacity" in df["policy_type"].values
    assert "prob_threshold" in df["policy_type"].values
    # Check default 10% policy exists
    cap10 = df[(df["policy_type"] == "weekly_capacity") & (df["parameter"] == 10.0)]
    assert not cap10.empty
    assert cap10.iloc[0]["selection_rate"] > 0.09 and cap10.iloc[0]["selection_rate"] < 0.11


def test_learner_trajectory_lookup():
    """Verify learner trajectory lookup returns ordered series for valid student."""
    idx = load_learner_index()
    assert not idx.empty
    sample_student = int(idx.iloc[0]["id_student"])
    sample_module = str(idx.iloc[0]["code_module"])
    
    traj = load_learner_trajectory(sample_student, sample_module)
    assert not traj.empty
    assert traj["week"].is_monotonic_increasing
    assert "prob_lgbm_v3" in traj.columns


def test_graceful_degradation_on_missing_shap():
    """Verify that absent SHAP file returns None without raising an exception."""
    res = load_shap_top5("non_existent_model", 9999999, 10)
    assert res is None


def test_fairness_and_calibration_caches():
    """Verify that fairness and calibration tables load cleanly."""
    fair = load_fairness_data()
    assert "fairness_metrics" in fair
    assert "fairness_gaps" in fair
    assert not fair["fairness_gaps"].empty
    
    calib = load_calibration_data()
    assert "reliability" in calib
    assert "summary" in calib
    assert not calib["reliability"].empty


def test_sealed_test_isolation_in_app_code():
    """Strict guard: Verify that app/ code never calls allow_test=True or loads sealed 2014J."""
    app_dir = Path(__file__).resolve().parent.parent / "app"
    for py_file in app_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "allow_test=True" not in content, f"Forbidden allow_test=True in {py_file}"
        assert "load_split(\"test\"" not in content, f"Forbidden load_split('test') in {py_file}"
