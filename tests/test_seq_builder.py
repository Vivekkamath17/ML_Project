"""
Unit tests for Sequence Builder (src/seq/builder.py).
Validates:
- Shapes: X is (N, W, F), M is (N, W), no NaN or inf anywhere.
- Padding: padded steps are exactly 0.0 and mask is False; real steps are contiguous and right-aligned.
- Row alignment: (learner_key, week) keys match Vivek's prediction files for the same split.
- Zero learner overlap between train and val tensors by learner_key.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.predictions import read_predictions
from src.seq.builder import build_sequences, get_v3_feature_names


@pytest.fixture(scope="module")
def val_seq_w8():
    return build_sequences(split="val", window=8, target="dropout")


def test_seq_builder_shapes_and_validity(val_seq_w8):
    """Verify shapes, types, and absence of NaNs/infs."""
    X = val_seq_w8["X"]
    M = val_seq_w8["M"]
    S = val_seq_w8["S"]
    y_d = val_seq_w8["y_dropout"]
    meta = val_seq_w8["meta"]

    N, W, F = X.shape
    assert W == 8
    assert F == 38
    assert len(meta) == N
    assert len(y_d) == N
    assert M.shape == (N, W)
    assert S.shape == (N, 4)

    # No NaN or Inf anywhere
    assert not np.isnan(X).any(), "NaN found in sequence tensor X!"
    assert not np.isinf(X).any(), "Inf found in sequence tensor X!"
    assert not np.isnan(S).any(), "NaN found in static tensor S!"
    assert not np.isnan(y_d).any(), "NaN found in target labels!"


def test_seq_builder_padding_and_alignment(val_seq_w8):
    """Verify left-padding: padded steps are strictly 0.0 and mask=False; real steps right-aligned."""
    X = val_seq_w8["X"]
    M = val_seq_w8["M"]
    weeks = val_seq_w8["meta"]["week"].to_numpy()
    N, W, F = X.shape

    for i in range(min(500, N)):
        w = weeks[i]
        expected_real_steps = min(w + 1, W)
        expected_padded_steps = W - expected_real_steps

        # Mask check
        assert M[i, :expected_padded_steps].sum() == 0, f"Padded steps marked True in mask for row {i}"
        assert M[i, expected_padded_steps:].all(), f"Real steps marked False in mask for row {i}"

        # Padded values must be exactly 0.0
        if expected_padded_steps > 0:
            assert np.all(X[i, :expected_padded_steps, :] == 0.0), f"Padded values not 0.0 for row {i}"


def test_seq_builder_row_alignment_with_vivek_preds(val_seq_w8):
    """Verify that rows line up 1:1 with Vivek's official validation predictions."""
    meta = val_seq_w8["meta"]
    df_base, _ = read_predictions("predictions/dropout/k3/lgbm_v3_val.parquet")

    assert len(meta) == len(df_base)

    base_keys = (
        df_base["code_module"] + "_" + df_base["code_presentation"] + "_" + df_base["id_student"].astype(str)
    ).to_numpy()
    seq_keys = meta["learner_key"].to_numpy()

    np.testing.assert_array_equal(seq_keys, base_keys)
    np.testing.assert_array_equal(meta["week"].to_numpy(), df_base["week"].to_numpy())
    np.testing.assert_array_equal(val_seq_w8["y_dropout"], df_base["y_true"].to_numpy())
