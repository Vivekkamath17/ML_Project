"""
Prediction file validation module for Phase 4 Sequence Models.
Enforces Hard Rule 4 and Checkpoint A4 requirements:
- Shared prediction schema (exact column order):
    learner_key, week, split, model, k, y_true, y_score
- No NaN values anywhere.
- Scores bounded in [0.0, 1.0] for probabilities.
- Composite key uniqueness: (learner_key, week, split, model, k).
- No test split rows; test presentation (2014J) is strictly quarantined.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "learner_key",
    "week",
    "split",
    "model",
    "k",
    "y_true",
    "y_score",
]

FINALSCORE_COLUMNS = [
    "learner_key",
    "week",
    "split",
    "model",
    "y_true",
    "y_score",
]

ALLOWED_SPLITS = {"train", "val"}


def validate_predictions(
    df_or_path: Union[pd.DataFrame, str, Path],
    is_final_score: bool = False,
) -> bool:
    """
    Validate a prediction table or file against the Phase 4 prediction contract.

    Parameters:
    -----------
    df_or_path: pd.DataFrame or path to CSV file.
    is_final_score: bool, whether this is a continuous final score prediction file.

    Returns:
    --------
    True if valid. Raises ValueError if invalid.
    """
    if isinstance(df_or_path, (str, Path)):
        p = Path(df_or_path)
        if not p.is_file():
            raise FileNotFoundError(f"Prediction file does not exist: {p}")
        if p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported file format: {p.suffix}. Expected .csv or .parquet")
    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        raise TypeError(f"Expected DataFrame or path, got {type(df_or_path)}")

    expected_cols = FINALSCORE_COLUMNS if is_final_score else REQUIRED_COLUMNS

    # 1. Column names and exact order
    actual_cols = list(df.columns)
    if actual_cols != expected_cols:
        raise ValueError(
            f"Schema mismatch: expected exact columns {expected_cols}, "
            f"got {actual_cols}"
        )

    # 2. No NaN values in any column
    for col in expected_cols:
        if df[col].isna().any():
            nan_count = df[col].isna().sum()
            raise ValueError(f"Column '{col}' contains {nan_count} NaN values.")

    # 3. Splits must be in ALLOWED_SPLITS only
    splits_present = set(df["split"].unique())
    invalid_splits = splits_present - ALLOWED_SPLITS
    if invalid_splits:
        raise ValueError(
            f"Invalid split(s) {invalid_splits} detected! Only {ALLOWED_SPLITS} are allowed. "
            f"Test split is strictly quarantined."
        )

    # Check for test presentation in learner_key
    if df["learner_key"].astype(str).str.contains("2014J").any():
        raise ValueError(
            "CRITICAL TEST QUARANTINE: Test presentation '2014J' detected in learner_key!"
        )

    # 4. Scores bounded in [0.0, 1.0] for classification
    if not is_final_score:
        score_min = df["y_score"].min()
        score_max = df["y_score"].max()
        if score_min < -1e-6 or score_max > 1.0 + 1e-6:
            raise ValueError(
                f"Probabilities out of bounds [0.0, 1.0]: min={score_min}, max={score_max}"
            )
        # Binary target check
        unique_targets = set(df["y_true"].unique())
        if not unique_targets.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Expected binary labels {0, 1}, got {unique_targets}")

    # 5. Composite key uniqueness
    id_cols = (
        ["learner_key", "week", "split", "model"]
        if is_final_score
        else ["learner_key", "week", "split", "model", "k"]
    )
    duplicates = df.duplicated(subset=id_cols)
    if duplicates.any():
        n_dup = int(duplicates.sum())
        raise ValueError(
            f"Unique key violation: {n_dup} duplicate rows found for composite key {id_cols}."
        )

    return True


# Alias for backward compatibility
validate_prediction_file = validate_predictions


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        is_score = "--score" in sys.argv
        validate_predictions(path, is_final_score=is_score)
        print(f"Validation PASSED for {path}")
