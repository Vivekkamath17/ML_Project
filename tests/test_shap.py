"""
Unit tests for SHAP feature attribution artifacts and data loading.
"""

from pathlib import Path
import pandas as pd
import pytest

from app.data_loader import load_shap_top5

REPO_ROOT = Path(__file__).resolve().parent.parent
SHAP_DIR = REPO_ROOT / "reports" / "shap"

EXPECTED_MODELS = [
    "lgbm_v3",
    "xgb",
    "catboost",
    "rf",
    "lr_39feat_unweighted",
]


@pytest.mark.parametrize("model_id", EXPECTED_MODELS)
def test_shap_files_exist_and_non_empty(model_id: str):
    """Verify that all 5 target SHAP parquet files exist and are non-empty."""
    p = SHAP_DIR / f"{model_id}_k3_val_top5.parquet"
    assert p.exists(), f"Missing SHAP file: {p}"
    assert p.stat().st_size > 1_000_000, f"SHAP file is unexpectedly small: {p.stat().st_size} bytes"

    df = pd.read_parquet(p)
    assert len(df) > 50_000, f"Expected >50,000 rows, found {len(df)}"
    assert df["id_student"].nunique() > 1_000, f"Expected >1,000 learners, found {df['id_student'].nunique()}"


@pytest.mark.parametrize("model_id", EXPECTED_MODELS)
def test_shap_schema_and_ranks(model_id: str):
    """Verify schema, rank bounds (1 to 5), and absence of NaNs in key attribution columns."""
    p = SHAP_DIR / f"{model_id}_k3_val_top5.parquet"
    df = pd.read_parquet(p)

    required_cols = {"id_student", "week", "rank", "feature", "shap_value", "model", "k"}
    assert required_cols.issubset(set(df.columns)), f"Missing required columns in {p.name}: {required_cols - set(df.columns)}"

    # Ranks strictly 1..5
    unique_ranks = sorted(df["rank"].unique().tolist())
    assert unique_ranks == [1, 2, 3, 4, 5], f"Unexpected ranks: {unique_ranks}"

    # No nulls in key columns
    assert not df["id_student"].isna().any(), "Found NaNs in id_student"
    assert not df["week"].isna().any(), "Found NaNs in week"
    assert not df["rank"].isna().any(), "Found NaNs in rank"
    assert not df["feature"].isna().any(), "Found NaNs in feature"
    assert not df["shap_value"].isna().any(), "Found NaNs in shap_value"

    # Both positive (risk drivers) and negative (protective factors) exist
    assert (df["shap_value"] > 0).any(), "Expected positive SHAP values"
    assert (df["shap_value"] < 0).any(), "Expected negative SHAP values"


def test_load_shap_top5_integration():
    """Verify that app.data_loader.load_shap_top5 integrates cleanly with generated parquet."""
    sample_p = SHAP_DIR / "lgbm_v3_k3_val_top5.parquet"
    df = pd.read_parquet(sample_p)
    sample_learner = int(df["id_student"].iloc[0])
    sample_week = int(df["week"].iloc[0])

    res = load_shap_top5("lgbm_v3", sample_learner, sample_week)
    assert res is not None
    assert len(res) == 5
    assert list(res["rank"]) == [1, 2, 3, 4, 5]
    assert "feature" in res.columns
    assert "shap_value" in res.columns


def test_missing_model_graceful_return():
    """Verify that requesting an ungenerated model returns None gracefully."""
    res = load_shap_top5("non_existent_deep_model", 99999, 5)
    assert res is None
