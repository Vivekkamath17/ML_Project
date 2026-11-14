"""
Unit tests for predictions.py and harness.py under the new schema contract.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.predictions import (
    write_predictions,
    read_predictions,
    get_prediction_path,
)
from src.models.harness import (
    evaluate_model,
    paired_bootstrap,
    learner_group_cv,
    tune_with_optuna,
    log_run,
)


@pytest.fixture
def sample_predictions_df():
    return pd.DataFrame({
        "id_student": [101, 102, 103, 104, 105],
        "code_module": ["BBB", "BBB", "CCC", "DDD", "DDD"],
        "code_presentation": ["2014B", "2014B", "2014B", "2014B", "2014B"],
        "week": [2, 2, 2, 2, 2],
        "model": ["test_model"] * 5,
        "k": [3] * 5,
        "target": ["label_dropout_k3"] * 5,
        "split": ["val"] * 5,
        "y_true": [0, 1, 0, 0, 1],
        "y_prob": [0.05, 0.85, 0.12, 0.03, 0.72],
    })


def test_predictions_roundtrip(tmp_path, sample_predictions_df):
    """Test write and read roundtrip with schema columns and metadata."""
    target_path = tmp_path / "preds_test.parquet"
    written_path = write_predictions(sample_predictions_df, target_path, metadata={"note": "roundtrip"})
    assert written_path.is_file()

    df_read, meta_read = read_predictions(written_path)
    pd.testing.assert_frame_equal(df_read, sample_predictions_df)
    assert meta_read["model"] == "test_model"
    assert meta_read["k"] == 3
    assert meta_read["target"] == "label_dropout_k3"
    assert meta_read["split"] == "val"
    assert meta_read["note"] == "roundtrip"


@pytest.mark.parametrize("invalid_split", ["test", "Test", " test", "holdout", "2014J"])
def test_predictions_rejects_invalid_split(tmp_path, sample_predictions_df, invalid_split):
    """Test that split outside {'train', 'val'} is strictly rejected."""
    df = sample_predictions_df.copy()
    df["split"] = invalid_split
    with pytest.raises(ValueError, match="Only splits in"):
        write_predictions(df, tmp_path / "preds_invalid.parquet")


def test_predictions_grade_band_schema(tmp_path):
    """Test that grade_band requires y_score_fail, y_score_pass, y_score_distinction."""
    df_gb = pd.DataFrame({
        "id_student": [101, 102],
        "code_module": ["BBB", "BBB"],
        "code_presentation": ["2014B", "2014B"],
        "week": [2, 2],
        "model": ["rf", "rf"],
        "k": [3, 3],
        "target": ["grade_band", "grade_band"],
        "split": ["val", "val"],
        "y_true": ["Pass", "Fail"],
        "y_score_fail": [0.1, 0.8],
        "y_score_pass": [0.7, 0.2],
        "y_score_distinction": [0.2, 0.0],
    })

    p = write_predictions(df_gb, tmp_path / "gb.parquet")
    df_read, _ = read_predictions(p)
    assert "y_score_fail" in df_read.columns
    assert "y_score_distinction" in df_read.columns

    # Missing one grade band column
    df_bad = df_gb.drop(columns=["y_score_distinction"])
    with pytest.raises(ValueError, match="Missing required grade_band prediction column"):
        write_predictions(df_bad, tmp_path / "gb_bad.parquet")


def test_prediction_path_generation():
    """Test get_prediction_path creates standard relative path."""
    p = get_prediction_path("label_dropout_k3", 3, "lgbm_v3", "val")
    assert str(p).replace("\\", "/") == "predictions/dropout/k3/lgbm_v3_val.parquet"


def test_existing_prediction_files_contain_no_test_rows():
    """Verify that all files in predictions/ contain zero test split rows."""
    pred_dir = Path("predictions")
    if not pred_dir.exists():
        return

    for p in pred_dir.glob("**/*.parquet"):
        df, meta = read_predictions(p)
        assert set(df["split"].unique()).issubset({"train", "val"}), f"Invalid split in {p}"
        assert not (df["code_presentation"] == "2014J").any(), f"Test presentation 2014J found in {p}"


def test_evaluate_model_full_suite():
    """Test evaluate_model returns point metrics, lift@top-10, and bootstrap CIs."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=100, p=[0.8, 0.2])
    y_prob = np.random.uniform(0.0, 1.0, size=100)
    cluster_ids = np.repeat(np.arange(20), 5)

    res = evaluate_model(y_true, y_prob, cluster_ids=cluster_ids, threshold=0.035, n_resamples=50)
    for key in ["roc_auc", "pr_auc", "brier_score", "lift_at_top10", "prec_at_top10"]:
        assert key in res
    assert "ci" in res
    assert "roc_auc" in res["ci"]


def test_log_run_enforces_run_name_and_experiment():
    """Test that log_run enforces <family>/<model>/<variant>/seed<N> and authorized experiments."""
    params = {"k": 3, "target": "label_dropout_k3"}
    metrics = {"roc_auc": 0.77}

    # Invalid run name format
    with pytest.raises(ValueError, match="Must match <family>/<model>/<variant>/seed<N>"):
        log_run("bad_run_name", params, metrics, experiment_name="dropout")

    # Invalid experiment name
    with pytest.raises(ValueError, match="Invalid experiment_name"):
        log_run("tree/lgbm/v3/seed42", params, metrics, experiment_name="unauthorized_exp")


def test_learner_group_cv_no_student_leakage():
    """Verify that learner_group_cv strictly isolates id_student between folds."""
    X = pd.DataFrame({"feat1": range(20)})
    y = np.random.randint(0, 2, size=20)
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
    cluster_ids = np.repeat(np.arange(20), 5)

    res = paired_bootstrap(y_true, y_prob, y_prob, cluster_ids=cluster_ids, n_boot=50, seed=42)
    assert pytest.approx(res["delta_roc"], abs=1e-6) == 0.0
    assert pytest.approx(res["delta_pr"], abs=1e-6) == 0.0


def test_optuna_tuning_wrapper():
    """Test that tune_with_optuna executes trials and returns the best study."""
    def toy_objective(trial):
        x = trial.suggest_float("x", -10.0, 10.0)
        return -(x - 3.0) ** 2

    study = tune_with_optuna(toy_objective, n_trials=3, seed=42, direction="maximize")
    assert len(study.trials) == 3
    assert "x" in study.best_params
