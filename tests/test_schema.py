"""
Comprehensive Schema and Fairness tests for OULAD data preparation pipeline.
Validates:
1. No forbidden targets, demographics, or population flags in feature list (using config/features.yaml).
2. Exactly one row per learner in audit_demographics.parquet.
3. No duplicate (code_module, code_presentation, id_student, week) in features_weekly.
4. Contiguous week grid starting at week 0 for every learner.
5. All labeled rows have is_enrolled_at_week_t == True and labels in {0, 1}.
6. Scaled features (39) are an exact subset of the feature manifest.
7. Verification that label_inactive_k4 <= label_inactive_k3 <= label_inactive_k2 on rows eligible for all three.
"""
from pathlib import Path
import polars as pl
import pytest
import yaml


@pytest.fixture(scope="module")
def manifest():
    with open("config/features.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def features_weekly():
    return pl.read_parquet("data/processed/features_weekly.parquet")


@pytest.fixture(scope="module")
def audit_table():
    return pl.read_parquet("data/processed/audit_demographics.parquet")


def test_feature_manifest_cleanliness(manifest):
    """
    Assert that forbidden targets, population flags, and demographics
    never appear in the feature manifest.
    """
    all_features = []
    for grp in ["static", "behavior", "temporal", "assessment", "curriculum"]:
        all_features.extend(manifest["features"][grp])

    forbidden = [
        "final_result",
        "date_unregistration",
        "effective_date_unregistration",
        "final_score",
        "grade_band",
        "gender",
        "age_band",
        "region",
        "highest_education",
        "imd_band",
        "disability",
        "never_active_learner",
        "is_zero_click_withdrawal",
        "label_is_proxy",
        "last_vle_date",
        "label_dropout_k2",
        "label_dropout_k3",
        "label_dropout_k4",
        "label_inactive_k2",
        "label_inactive_k3",
        "label_inactive_k4",
        "label_inactive_persistent_k2",
        "label_inactive_persistent_k3",
        "label_inactive_persistent_k4",
    ]

    for col in forbidden:
        assert col not in all_features, f"Forbidden column '{col}' found in feature manifest!"


def test_scaled_features_subset_of_manifest(manifest):
    """Assert that the 39 scaled features are an exact subset of the feature manifest."""
    all_features = set()
    for grp in ["static", "behavior", "temporal", "assessment", "curriculum"]:
        all_features.update(manifest["features"][grp])

    scaled = set(manifest["scaled_features"])
    assert len(scaled) == 39
    assert scaled.issubset(all_features), f"Scaled features not a subset: {scaled - all_features}"


def test_no_duplicate_keys(features_weekly):
    """Assert no duplicate (learner_key, week) rows exist in features_weekly."""
    key = ["code_module", "code_presentation", "id_student", "week"]
    n_total = len(features_weekly)
    n_unique = len(features_weekly.select(key).unique())
    assert n_total == n_unique, f"Duplicate keys detected: {n_total} total vs {n_unique} unique!"


def test_contiguous_week_grid(features_weekly):
    """Assert every learner in the grid starts at week 0 and has contiguous weeks."""
    key = ["code_module", "code_presentation", "id_student"]
    # Group by learner to check min week and continuity: max_week - min_week + 1 == count
    stats = features_weekly.group_by(key).agg([
        pl.col("week").min().alias("min_w"),
        pl.col("week").max().alias("max_w"),
        pl.len().alias("count_w"),
    ])

    assert (stats["min_w"] == 0).all(), "Some learners do not start at week 0!"
    is_contiguous = (stats["max_w"] - stats["min_w"] + 1) == stats["count_w"]
    assert is_contiguous.all(), "Non-contiguous weeks detected for some learners!"


def test_label_validity_and_enrolled_condition(features_weekly):
    """Assert that all labeled rows have is_enrolled_at_week_t == True and labels in {0, 1}."""
    for k in [2, 3, 4]:
        valid_rows = features_weekly.filter(
            pl.col("is_enrolled_at_week_t") & ~pl.col(f"is_censored_k{k}")
        )
        assert (valid_rows[f"label_dropout_k{k}"].is_not_null()).all()
        assert set(valid_rows[f"label_dropout_k{k}"].unique().to_list()).issubset({0, 1})

        # When learner is NOT enrolled at week t, label must be null
        un_enrolled = features_weekly.filter(~pl.col("is_enrolled_at_week_t"))
        assert (un_enrolled[f"label_dropout_k{k}"].is_null()).all()


def test_inactivity_onset_monotonicity(features_weekly):
    """
    Verify that on rows eligible for all three onset horizons (k=2, k=3, k=4),
    label_inactive_k4 <= label_inactive_k3 <= label_inactive_k2 strictly holds.
    (Requiring 0 clicks for 4 weeks is strictly harder than for 3 or 2 weeks).
    """
    eligible = features_weekly.filter(
        pl.col("label_inactive_k2").is_not_null()
        & pl.col("label_inactive_k3").is_not_null()
        & pl.col("label_inactive_k4").is_not_null()
    )
    violations = eligible.filter(
        (pl.col("label_inactive_k4") > pl.col("label_inactive_k3"))
        | (pl.col("label_inactive_k3") > pl.col("label_inactive_k2"))
    )
    assert len(violations) == 0, f"Monotonicity violated in {len(violations)} rows!"


def test_fairness_audit_table_isolation(audit_table, manifest):
    """
    Assert fairness audit table has exactly one row per learner,
    no duplicate keys, and no model features.
    """
    key = ["code_module", "code_presentation", "id_student"]
    assert len(audit_table) == len(audit_table.select(key).unique())

    all_model_features = set()
    for grp in ["static", "behavior", "temporal", "assessment", "curriculum"]:
        all_model_features.update(manifest["features"][grp])

    overlap = set(audit_table.columns) & all_model_features
    assert len(overlap) == 0, f"Model features leaked into audit table: {overlap}"

