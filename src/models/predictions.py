"""
Shared Prediction-File Contract for OULAD Models.

This module provides standardized serialization and deserialization for model predictions.
Enforces validation checks:
- Required real columns:
  ['id_student', 'code_module', 'code_presentation', 'week', 'model', 'k', 'target', 'split', 'y_true']
  and predictions: 'y_prob' (or 'y_pred' for regression, or 'y_score_fail', 'y_score_pass', 'y_score_distinction' for grade_band).
- Split must be in {'train', 'val'} only. Test split is strictly forbidden.
- Key uniqueness across (id_student, code_module, code_presentation, week).
- Probability bounds [0.0, 1.0] for classification.
- Standard path generation: predictions/<target>/k<k>/<model>_<split>.parquet
- Embedded JSON metadata in Parquet table schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


METADATA_KEY = b"oulad_metadata"
ALLOWED_SPLITS = {"train", "val"}


def get_prediction_path(
    target: str,
    k: int,
    model: str,
    split: str,
    base_dir: Union[str, Path] = "predictions",
) -> Path:
    """
    Generate standard prediction file path: predictions/<target>/k<k>/<model>_<split>.parquet
    """
    # Sanitize target name for directory (e.g. label_dropout_k3 -> dropout, label_inactive_k3 -> onset)
    if "dropout" in target:
        t_dir = "dropout"
    elif "inactive" in target or "onset" in target:
        t_dir = "onset"
    elif "score" in target:
        t_dir = "score"
    elif "grade" in target:
        t_dir = "grade_band"
    else:
        t_dir = target

    return Path(base_dir) / t_dir / f"k{k}" / f"{model}_{split}.parquet"


def write_predictions(
    df: pd.DataFrame,
    path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Validate and serialize predictions to a Parquet file with embedded metadata.

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame containing predictions.
        Required real columns:
        ['id_student', 'code_module', 'code_presentation', 'week', 'model', 'k', 'target', 'split', 'y_true']
        plus predictions:
        - binary classification: 'y_prob'
        - continuous regression: 'y_pred'
        - grade_band multiclass: ['y_score_fail', 'y_score_pass', 'y_score_distinction']
    path: Optional[Union[str, Path]]
        Target file path. If None, automatically derived via get_prediction_path().
    metadata: Optional[Dict[str, Any]]
        Optional additional run/model metadata.

    Returns:
    --------
    Path: Absolute Path to the written file.
    """
    # 1. Required identifier and context columns
    required_cols = [
        "id_student",
        "code_module",
        "code_presentation",
        "week",
        "model",
        "k",
        "target",
        "split",
        "y_true",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in prediction dataframe: '{col}'")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values; null values are forbidden.")

    # 2. Split hygiene: strictly {'train', 'val'} only
    splits_present = set(df["split"].unique())
    invalid_splits = splits_present - ALLOWED_SPLITS
    if invalid_splits:
        raise ValueError(
            f"Invalid split(s) {invalid_splits} found in dataframe. "
            f"Only splits in {ALLOWED_SPLITS} are permitted. Test split is strictly quarantined."
        )

    # Check for test presentation
    if (df["code_presentation"] == "2014J").any():
        raise ValueError(
            "CRITICAL TEST QUARANTINE: Test presentation '2014J' detected in prediction dataframe! "
            "Test data must never be written."
        )

    # 3. Target and prediction columns validation
    target_name = str(df["target"].iloc[0])
    is_grade_band = "grade" in target_name
    is_score = "score" in target_name
    has_prob = "y_prob" in df.columns
    has_pred = "y_pred" in df.columns

    if is_grade_band:
        gb_cols = ["y_score_fail", "y_score_pass", "y_score_distinction"]
        for gbc in gb_cols:
            if gbc not in df.columns:
                raise ValueError(f"Missing required grade_band prediction column: '{gbc}'")
            if df[gbc].isna().any():
                raise ValueError(f"Column '{gbc}' contains NaN values.")
    elif is_score:
        if not has_pred:
            raise ValueError("Score regression prediction dataframe must contain 'y_pred'.")
        if df["y_pred"].isna().any():
            raise ValueError("Column 'y_pred' contains NaN values.")
    else:
        # Binary classification (dropout, onset)
        if not has_prob:
            raise ValueError("Binary prediction dataframe must contain 'y_prob'.")
        if df["y_prob"].isna().any():
            raise ValueError("Column 'y_prob' contains NaN values.")
        prob_min = df["y_prob"].min()
        prob_max = df["y_prob"].max()
        if prob_min < 0.0 or prob_max > 1.0:
            raise ValueError(
                f"Probabilities in 'y_prob' out of bounds [0.0, 1.0]. Range found: [{prob_min}, {prob_max}]."
            )

    # 4. Composite key uniqueness
    id_cols = ["id_student", "code_module", "code_presentation", "week"]
    duplicates = df.duplicated(subset=id_cols)
    if duplicates.any():
        n_dup = duplicates.sum()
        raise ValueError(
            f"Unique key violation: {n_dup} duplicate rows found for composite key {id_cols}."
        )

    # 5. Determine output path
    if path is None:
        target_val = str(df["target"].iloc[0])
        k_val = int(df["k"].iloc[0])
        model_val = str(df["model"].iloc[0])
        split_val = str(df["split"].iloc[0])
        out_path = get_prediction_path(target_val, k_val, model_val, split_val)
    else:
        out_path = Path(path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Serialization with PyArrow and JSON metadata
    table = pa.Table.from_pandas(df)
    existing_meta = table.schema.metadata or {}
    meta_dict = metadata.copy() if metadata else {}
    meta_dict.setdefault("model", str(df["model"].iloc[0]))
    meta_dict.setdefault("k", int(df["k"].iloc[0]))
    meta_dict.setdefault("target", str(df["target"].iloc[0]))
    meta_dict.setdefault("split", str(df["split"].iloc[0]))

    metadata_bytes = json.dumps(meta_dict, default=str).encode("utf-8")
    combined_meta = {**existing_meta, METADATA_KEY: metadata_bytes}
    table = table.replace_schema_metadata(combined_meta)

    pq.write_table(table, out_path, compression="zstd")
    return out_path.resolve()


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
