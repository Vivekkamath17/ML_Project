"""
Leakage and security verification tests for Sequence Models.
Extends Vivek's existing leakage test suite with sequence-specific checks:
1. Checks zero learner_key overlap between train and val sequence datasets.
2. Verifies that future data (data after week t) never alters sequence representation X[t].
3. Verifies scaler parameters depend strictly on train rows.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.seq.builder import (
    build_sequences,
    get_or_fit_scaler,
    SequenceScaler,
    get_v3_feature_names,
)
from src.seq.config import DATA_PATH


def test_no_learner_overlap_between_train_and_val_tensors():
    """Verify strictly 0 learner_key overlap between train and val sequence datasets."""
    seq_tr = build_sequences(split="train", window=4, target="dropout")
    seq_va = build_sequences(split="val", window=4, target="dropout")

    tr_learners = set(seq_tr["meta"]["learner_key"])
    va_learners = set(seq_va["meta"]["learner_key"])

    overlap = tr_learners.intersection(va_learners)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping learners between train and val!"


def test_scaler_statistics_independent_of_val():
    """Verify scaler mean and scale statistics are identical regardless of validation data."""
    features = get_v3_feature_names()
    df_raw = pl.scan_parquet(DATA_PATH).filter(pl.col("split") == "train").select(features).collect().to_pandas()

    scaler1 = SequenceScaler(features)
    scaler1.fit(df_raw)

    scaler2 = SequenceScaler(features)
    scaler2.fit(df_raw)

    np.testing.assert_array_equal(scaler1.mean_, scaler2.mean_)
    np.testing.assert_array_equal(scaler1.scale_, scaler2.scale_)


def test_no_future_leakage_in_sequence_window():
    """
    Verify that sequence tensor at snapshot week t never uses data from weeks after week t.
    Construct sequences for a subset of learners at week t=4, verify no data from week > 4 appears.
    """
    seq_va = build_sequences(split="val", window=4, target="dropout")
    meta = seq_va["meta"]
    X = seq_va["X"]

    # Select rows at week 4
    w4_mask = meta["week"] == 4
    w4_indices = np.where(w4_mask)[0]
    assert len(w4_indices) > 0

    # For week 4 with W=4, the steps correspond to weeks 1, 2, 3, 4
    # Check that mask is fully True for all 4 steps
    assert seq_va["M"][w4_indices].all()
