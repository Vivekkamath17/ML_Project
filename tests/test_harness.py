"""
Unit tests for the shared prediction-file contract (predictions.py)
and evaluation harness (harness.py).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.predictions import write_predictions, read_predictions
from src.models.harness import (
    evaluate_model,
    paired_bootstrap,
    learner_group_cv,
    tune_with_optuna,
)


@pytest.fixture
def sample_predictions_df():
    return pd.DataFrame({
        "id_student": [101, 102, 103, 104, 105],
        "code_module": ["BBB", "BBB", "CCC", "DDD", "DDD"],
        "code_presentation": ["2014B", "2014B", "2014B", "2014B", "2014B"],
        "week": [2, 2, 2, 2, 2],
        "y_true": [0, 1, 0, 0, 1],
        "y_prob": [0.05, 0.85, 0.12, 0.03, 0.72],
    })


def test_predictions_roundtrip(tmp_path, sample_predictions_df):
    """Test that writing and reading predictions preserves data and JSON metadata exactly."""
    target_path = tmp_path / "preds_test.parquet"
    metadata = {
        "model_name": "test_lgbm",
        "split": "val",
        "target": "label_dropout_k3",
        "horizon": 3,
        "custom_metric": 0.85,
    }

    written_path = write_predictions(sample_predictions_df, target_path, metadata)
    assert written_path.is_file()

    df_read, meta_read = read_predictions(written_path)
    pd.testing.assert_frame_equal(df_read, sample_predictions_df)
    assert meta_read["model_name"] == "test_lgbm"
    assert meta_read["split"] == "val"
    assert meta_read["horizon"] == 3
    assert meta_read["custom_metric"] == 0.85


@pytest.mark.parametrize("forbidden_split", ["test", "Test", " test", "TEST"])
def test_predictions_rejects_test_split(tmp_path, sample_predictions_df, forbidden_split):
    """Test that any attempt to write predictions for the test split is rejected."""
    target_path = tmp_path / "preds_forbidden.parquet"
    metadata = {"model_name": "leak_model", "split": forbidden_split}

    with pytest.raises(ValueError, match="CRITICAL TEST QUARANTINE"):
        write_predictions(sample_predictions_df, target_path, metadata)


def test_predictions_rejects_test_presentation(tmp_path, sample_predictions_df):
    """Test that presence of '2014J' in dataframe raises quarantine ValueError."""
    target_path = tmp_path / "preds_forbidden_pres.parquet"
    df = sample_predictions_df.copy()
    df.loc[0, "code_presentation"] = "2014J"
    metadata = {"model_name": "test_model", "split": "val"}

    with pytest.raises(ValueError, match="CRITICAL TEST QUARANTINE"):
        write_predictions(df, target_path, metadata)


def test_predictions_validation_errors(tmp_path, sample_predictions_df):
    """Test that invalid probabilities, missing columns, NaNs, and duplicate keys are rejected."""
    target_path = tmp_path / "preds_invalid.parquet"
    meta = {"model_name": "m", "split": "val"}

    # Invalid probability > 1.0
    df_high = sample_predictions_df.copy()
    df_high.loc[0, "y_prob"] = 1.25
    with pytest.raises(ValueError, match="Probabilities in 'y_prob' out of bounds"):
        write_predictions(df_high, target_path, meta)

    # Invalid probability < 0.0
    df_low = sample_predictions_df.copy()
    df_low.loc[0, "y_prob"] = -0.05
    with pytest.raises(ValueError, match="Probabilities in 'y_prob' out of bounds"):
        write_predictions(df_low, target_path, meta)

    # NaN in probability
    df_nan = sample_predictions_df.copy()
    df_nan.loc[0, "y_prob"] = np.nan
    with pytest.raises(ValueError, match="Column 'y_prob' contains NaN values"):
        write_predictions(df_nan, target_path, meta)

    # Missing identifier column
    df_no_key = sample_predictions_df.drop(columns=["code_module"])
    with pytest.raises(ValueError, match="Missing required identifier column"):
        write_predictions(df_no_key, target_path, meta)

    # Duplicate row keys
    df_dup = pd.concat([sample_predictions_df, sample_predictions_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Unique key violation"):
        write_predictions(df_dup, target_path, meta)


def test_existing_prediction_files_contain_no_test_rows():
    """Verify that any saved parquet files in predictions/ contain zero test split rows."""
    pred_dir = Path("predictions")
    if not pred_dir.exists():
        return

    for p in pred_dir.glob("**/*.parquet"):
        df, meta = read_predictions(p)
        assert meta.get("split", "").strip().lower() != "test", f"File {p} has test split in metadata!"
        if "code_presentation" in df.columns:
            assert not (df["code_presentation"] == "2014J").any(), f"File {p} contains test presentation 2014J!"


def test_evaluate_model():
    """Test evaluate_model outputs correct dictionary and respects threshold."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

    res = evaluate_model(y_true, y_prob, threshold=0.5)
    assert "roc_auc" in res
    assert "pr_auc" in res
    assert "brier_score" in res
    assert "precision" in res
    assert "recall" in res
    assert "f1" in res
    assert "mcc" in res

    assert res["roc_auc"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["mcc"] == 1.0


def test_learner_group_cv_no_student_leakage():
    """Verify that learner_group_cv strictly isolates id_student between folds."""
    X = pd.DataFrame({"feat1": range(20)})
    y = np.random.randint(0, 2, size=20)
    # 5 students, each appearing 4 times
    students = np.array([101, 101, 101, 101,
                         102, 102, 102, 102,
                         103, 103, 103, 103,
                         104, 104, 104, 104,
                         105, 105, 105, 105])

    folds = list(learner_group_cv(X, y, groups=students, n_splits=5))
    assert len(folds) == 5

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_students = set(students[train_idx])
        val_students = set(students[val_idx])
        overlap = train_students.intersection(val_students)
        assert len(overlap) == 0, f"Fold {fold_idx} leaks students: {overlap}"


def test_paired_bootstrap_identical_models():
    """Test paired_bootstrap produces zero delta and zero CI when comparing a model to itself."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=100, p=[0.8, 0.2])
    y_prob = np.random.uniform(0.0, 1.0, size=100)
    cluster_ids = np.repeat(np.arange(20), 5)  # 20 students, 5 rows each

    res = paired_bootstrap(y_true, y_prob, y_prob, cluster_ids=cluster_ids, n_boot=100, seed=42)

    assert pytest.approx(res["delta_roc"], abs=1e-6) == 0.0
    assert pytest.approx(res["delta_pr"], abs=1e-6) == 0.0
    assert pytest.approx(res["delta_roc_ci"][0], abs=1e-6) == 0.0
    assert pytest.approx(res["delta_roc_ci"][1], abs=1e-6) == 0.0
    assert pytest.approx(res["delta_pr_ci"][0], abs=1e-6) == 0.0
    assert pytest.approx(res["delta_pr_ci"][1], abs=1e-6) == 0.0


def test_optuna_tuning_wrapper():
    """Test that tune_with_optuna executes trials and returns the best study."""
    def toy_objective(trial):
        x = trial.suggest_float("x", -10.0, 10.0)
        # Maximize -(x - 3)^2 => optimum at x = 3, value = 0
        return -(x - 3.0) ** 2

    study = tune_with_optuna(toy_objective, n_trials=5, seed=42, direction="maximize")
    assert len(study.trials) == 5
    assert "x" in study.best_params
    assert study.best_value <= 0.0
