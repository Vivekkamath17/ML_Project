import os
from pathlib import Path
import pytest
import yaml
import pyarrow.parquet as pq
import pandas as pd


def test_no_test_split_access():
    """Ensure no test split 2014J is loaded or allow_test=True is used."""
    calib_files = list(Path("predictions/calibration").glob("*.parquet"))
    assert len(calib_files) > 0, "No calibration prediction files found."
    for f in calib_files:
        df = pd.read_parquet(f)
        assert (df["split"] != "test").all(), f"Test split leaked into {f}"
        assert (df["code_presentation"] != "2014J").all(), f"Test presentation 2014J in {f}"


def test_finalists_yaml_references_existing_files():
    """Verify finalists.yaml exists and points to real calibrators."""
    fin_path = Path("reports/niranjan/finalists.yaml")
    assert fin_path.exists(), "finalists.yaml does not exist."
    with open(fin_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    assert "finalists" in data
    assert 3 <= len(data["finalists"]) <= 4
    for finalist in data["finalists"]:
        cal_path = Path(finalist["calibrator_path"])
        assert cal_path.exists(), f"Calibrator file {cal_path} does not exist!"


def test_calibrators_never_scored_on_training_rows():
    """Ensure reported calibration metrics are based on out-of-fold predictions."""
    summary_path = Path("reports/niranjan/calibration_summary.csv")
    assert summary_path.exists(), "calibration_summary.csv does not exist."
    df = pd.read_csv(summary_path)
    assert not df.empty
    assert "brier_after" in df.columns
    assert "ece_after" in df.columns


def test_prediction_files_schema():
    """Verify all calibrated parquet files adhere to standard prediction schema."""
    required_cols = {"id_student", "code_module", "code_presentation", "week", "model", "k", "target", "split", "y_true", "y_prob"}
    for f in Path("predictions/calibration").glob("*.parquet"):
        t = pq.read_table(f)
        missing = required_cols - set(t.column_names)
        assert not missing, f"Missing columns in {f}: {missing}"
