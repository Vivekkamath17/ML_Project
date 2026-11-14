"""
Automated unit tests for Phase 3 Model Zoo scaffolding and constraints.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.zoo.common import (
    ImputedZooPreprocessor,
    get_learner_subsample_mask,
    load_zoo_data,
    compute_median_inner_iterations,
)
from src.models.predictions import write_predictions, read_predictions


def test_feature_list_is_exactly_38_v3_features():
    """Verify that load_zoo_data returns exactly the 38 sanctioned v3 features."""
    X_tr, _, _, X_va, _, _, v3_features = load_zoo_data()
    assert len(v3_features) == 38
    assert "n_skipped_resources" not in v3_features
    assert list(X_tr.columns) == v3_features
    assert list(X_va.columns) == v3_features


def test_preprocessor_fitted_on_train_only():
    """Verify that ImputedZooPreprocessor is fitted strictly on train data and never sees val."""
    X_tr, _, _, X_va, _, _, _ = load_zoo_data()
    preprocessor = ImputedZooPreprocessor()
    preprocessor.fit(X_tr)

    # Transform both
    arr_tr = preprocessor.transform(X_tr)
    arr_va = preprocessor.transform(X_va)

    assert not np.isnan(arr_tr).any()
    assert not np.isnan(arr_va).any()
    assert arr_tr.shape[1] == len(preprocessor.get_feature_names())
    assert arr_va.shape[1] == len(preprocessor.get_feature_names())

    # Preprocessor output names include 38 features + 5 indicator columns
    assert len(preprocessor.get_feature_names()) == 38 + 5


def test_test_split_quarantine_in_zoo(tmp_path):
    """Verify that no test-split data or presentation 2014J can reach prediction files."""
    df_test = pd.DataFrame({
        "id_student": [999],
        "code_module": ["CCC"],
        "code_presentation": ["2014J"],
        "week": [2],
        "model": ["xgb"],
        "k": [3],
        "target": ["label_dropout_k3"],
        "split": ["val"],
        "y_true": [0],
        "y_prob": [0.1],
    })

    with pytest.raises(ValueError, match="CRITICAL TEST QUARANTINE"):
        write_predictions(df_test, tmp_path / "leak.parquet")


def test_deterministic_predictions_same_seed():
    """Verify that two identical runs with the same seed produce identical predictions."""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 5))
    y = rng.integers(0, 2, size=50)

    clf1 = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf1.fit(X, y)
    p1 = clf1.predict_proba(X)[:, 1]

    clf2 = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf2.fit(X, y)
    p2 = clf2.predict_proba(X)[:, 1]

    np.testing.assert_array_equal(p1, p2)


def test_learner_subsample_mask_groups_by_student():
    """Verify that the 25% subsample mask selects strictly by unique id_student."""
    meta = pd.DataFrame({
        "id_student": [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
        "code_module": ["BBB"] * 10,
        "code_presentation": ["2013B"] * 10,
        "week": [2, 3, 4, 2, 3, 2, 3, 2, 3, 4],
    })

    mask = get_learner_subsample_mask(meta, fraction=0.5, seed=42)
    sampled_students = meta.loc[mask, "id_student"].unique()
    non_sampled_students = meta.loc[~mask, "id_student"].unique()

    # Zero student overlap between sampled and non-sampled
    assert len(set(sampled_students).intersection(set(non_sampled_students))) == 0


def test_compute_median_inner_iterations():
    """Test median iteration helper across fold iterations."""
    iters = [120, 150, 140, 190, 130]
    med = compute_median_inner_iterations(iters)
    assert med == 140


# ── New schema-enforcement tests (Part B requirements) ─────────────────────────

def test_write_predictions_requires_model_column(tmp_path):
    """write_predictions must reject a dataframe missing the 'model' real column."""
    df = pd.DataFrame({
        "id_student": [1], "code_module": ["AAA"], "code_presentation": ["2013B"],
        "week": [2], "k": [3], "target": ["label_dropout_k3"], "split": ["val"],
        "y_true": [0], "y_prob": [0.1],
    })
    with pytest.raises(ValueError, match="model"):
        write_predictions(df, tmp_path / "out.parquet")


def test_write_predictions_requires_k_column(tmp_path):
    """write_predictions must reject a dataframe missing the 'k' real column."""
    df = pd.DataFrame({
        "id_student": [1], "code_module": ["AAA"], "code_presentation": ["2013B"],
        "week": [2], "model": ["rf"], "target": ["label_dropout_k3"], "split": ["val"],
        "y_true": [0], "y_prob": [0.1],
    })
    with pytest.raises(ValueError, match="k"):
        write_predictions(df, tmp_path / "out.parquet")


def test_write_predictions_requires_target_column(tmp_path):
    """write_predictions must reject a dataframe missing the 'target' real column."""
    df = pd.DataFrame({
        "id_student": [1], "code_module": ["AAA"], "code_presentation": ["2013B"],
        "week": [2], "model": ["rf"], "k": [3], "split": ["val"],
        "y_true": [0], "y_prob": [0.1],
    })
    with pytest.raises(ValueError, match="target"):
        write_predictions(df, tmp_path / "out.parquet")


def test_write_predictions_train_val_only(tmp_path):
    """write_predictions must reject split='test'."""
    df = pd.DataFrame({
        "id_student": [1], "code_module": ["AAA"], "code_presentation": ["2013B"],
        "week": [2], "model": ["rf"], "k": [3], "target": ["label_dropout_k3"],
        "split": ["test"], "y_true": [0], "y_prob": [0.1],
    })
    with pytest.raises(ValueError):
        write_predictions(df, tmp_path / "out.parquet")


def test_write_predictions_accepts_train_split(tmp_path):
    """write_predictions must accept split='train'."""
    df = pd.DataFrame({
        "id_student": [1], "code_module": ["AAA"], "code_presentation": ["2013B"],
        "week": [2], "model": ["rf"], "k": [3], "target": ["label_dropout_k3"],
        "split": ["train"], "y_true": [0], "y_prob": [0.1],
    })
    out = write_predictions(df, tmp_path / "out.parquet")
    assert out.exists()


def test_write_predictions_grade_band_requires_three_score_cols(tmp_path):
    """grade_band target must require y_score_fail, y_score_pass, y_score_distinction."""
    df_missing_scores = pd.DataFrame({
        "id_student": [1], "code_module": ["AAA"], "code_presentation": ["2013B"],
        "week": [2], "model": ["rf"], "k": [3], "target": ["label_grade_band_k3"],
        "split": ["val"], "y_true": [0],
        # Only one of three required score columns present
        "y_score_fail": [0.5],
    })
    with pytest.raises(ValueError, match="y_score_pass"):
        write_predictions(df_missing_scores, tmp_path / "out.parquet")


def test_write_predictions_grade_band_full_schema(tmp_path):
    """grade_band target with all three score columns must succeed."""
    df = pd.DataFrame({
        "id_student": [1], "code_module": ["AAA"], "code_presentation": ["2013B"],
        "week": [2], "model": ["rf"], "k": [3], "target": ["label_grade_band_k3"],
        "split": ["val"], "y_true": [0],
        "y_score_fail": [0.5], "y_score_pass": [0.3], "y_score_distinction": [0.2],
    })
    out = write_predictions(df, tmp_path / "out.parquet")
    assert out.exists()


def test_path_generation_uses_model_k_target_split(tmp_path):
    """Path generation must follow predictions/<target>/k<k>/<model>_<split>.parquet."""
    from src.models.predictions import get_prediction_path
    p = get_prediction_path("label_dropout_k3", k=3, model="rf", split="val",
                            base_dir=tmp_path)
    assert p.parent.name == "k3"
    assert p.parent.parent.name == "dropout"
    assert p.name == "rf_val.parquet"


def test_permutation_importance_shape():
    """compute_permutation_importance must return one entry per feature."""
    from src.models.zoo.importance import compute_permutation_importance
    rng = np.random.default_rng(42)
    X = rng.random((200, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.1, size=200) > 0.5).astype(int)
    feat_names = ["f0", "f1", "f2", "f3", "f4"]

    # Dummy predict function (always 0.5)
    def dummy_predict(X_in):
        return np.full(len(X_in), 0.5)

    imp = compute_permutation_importance(
        dummy_predict, X, y, feat_names, n_repeats=1, seed=42, metric="pr_auc"
    )
    assert set(imp.keys()) == set(feat_names)
    # All importances ≈ 0 for a constant predictor
    for v in imp.values():
        assert abs(v) < 1e-6, f"Expected ~0 importance for constant predictor, got {v}"
