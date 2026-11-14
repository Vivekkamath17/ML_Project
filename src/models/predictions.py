"""
Shared Prediction-File Contract for OULAD Models.

This module provides standardized serialization and deserialization for model predictions.
Enforces validation checks:
- Required identifier and target/prediction columns
- Key uniqueness across (id_student, code_module, code_presentation, week)
- Probability bounds [0.0, 1.0] for classification
- Strict test-split quarantine: rejects 'test', 'Test', ' test', and presentation '2014J'.
- Embedded JSON metadata in Parquet table schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


METADATA_KEY = b"oulad_metadata"
FORBIDDEN_SPLITS = {"test", "Test", " test", "TEST"}


def write_predictions(
    df: pd.DataFrame,
    path: Union[str, Path],
    metadata: Dict[str, Any],
) -> Path:
    """
    Validate and serialize predictions to a Parquet file with embedded metadata.

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame containing predictions.
        Must contain: ['id_student', 'code_module', 'code_presentation', 'week', 'y_true']
        and at least one of ['y_prob', 'y_pred'].
    path: Union[str, Path]
        Target file path (convention: predictions/<target_type>/<horizon>/<model>_<split>.parquet).
    metadata: Dict[str, Any]
        Dictionary of run/model metadata (model_name, split, target, horizon, etc.).

    Returns:
    --------
    Path: Absolute Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Split Hygiene & Test Split Quarantine Check
    split_val = str(metadata.get("split", ""))
    if split_val in FORBIDDEN_SPLITS or split_val.strip().lower() == "test":
        raise ValueError(
            f"CRITICAL TEST QUARANTINE: Writing test split predictions ('{split_val}') "
            "is strictly forbidden in Phase 2! The test split (2014J) must remain quarantined."
        )

    # Check for test presentation in dataframe
    if "code_presentation" in df.columns:
        if (df["code_presentation"] == "2014J").any():
            raise ValueError(
                "CRITICAL TEST QUARANTINE: Test presentation '2014J' detected in prediction dataframe! "
                "The test split must not be accessed or written in Phase 2."
            )

    # 2. Required columns validation
    required_id_cols = ["id_student", "code_module", "code_presentation", "week"]
    for col in required_id_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required identifier column: '{col}'")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values; null identifiers are forbidden.")

    if "y_true" not in df.columns:
        raise ValueError("Missing required ground truth column: 'y_true'")
    if df["y_true"].isna().any():
        raise ValueError("Column 'y_true' contains NaN values.")

    has_prob = "y_prob" in df.columns
    has_pred = "y_pred" in df.columns
    if not has_prob and not has_pred:
        raise ValueError("Prediction DataFrame must contain either 'y_prob' (classification) or 'y_pred' (regression).")

    # 3. Probability bounds check
    if has_prob:
        if df["y_prob"].isna().any():
            raise ValueError("Column 'y_prob' contains NaN values.")
        prob_min = df["y_prob"].min()
        prob_max = df["y_prob"].max()
        if prob_min < 0.0 or prob_max > 1.0:
            raise ValueError(
                f"Probabilities in 'y_prob' out of bounds [0.0, 1.0]. Range found: [{prob_min}, {prob_max}]. "
                "Did you pass uncalibrated logits or regression targets?"
            )

    if has_pred and df["y_pred"].isna().any():
        raise ValueError("Column 'y_pred' contains NaN values.")

    # 4. Uniqueness check
    duplicates = df.duplicated(subset=required_id_cols)
    if duplicates.any():
        n_dup = duplicates.sum()
        raise ValueError(
            f"Unique key violation: {n_dup} duplicate rows found for composite key "
            f"{required_id_cols}."
        )

    # 5. Serialization with PyArrow and JSON metadata
    table = pa.Table.from_pandas(df)
    existing_meta = table.schema.metadata or {}
    metadata_bytes = json.dumps(metadata, default=str).encode("utf-8")
    combined_meta = {**existing_meta, METADATA_KEY: metadata_bytes}
    table = table.replace_schema_metadata(combined_meta)

    pq.write_table(table, path, compression="zstd")
    return path.resolve()


def read_predictions(path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read predictions Parquet file and extract embedded metadata.

    Parameters:
    -----------
    path: Union[str, Path]
        Path to Parquet file.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]:
        DataFrame of predictions, and dictionary of metadata.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    table = pq.read_table(path)
    df = table.to_pandas()

    schema_meta = table.schema.metadata or {}
    raw_meta = schema_meta.get(METADATA_KEY)
    if raw_meta:
        metadata = json.loads(raw_meta.decode("utf-8"))
    else:
        metadata = {}

    return df, metadata
