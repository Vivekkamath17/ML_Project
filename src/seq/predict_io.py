"""
Prediction I/O module for Phase 4 Sequence Models.
Enforces:
- Hard Rule 4 shared prediction schema:
    learner_key, week, split, model, k, y_true, y_score
- Multi-task final score schema:
    learner_key, week, split, model, y_true, y_score
- Automatic validation via validate_predictions() on every write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from src.seq.config import PREDICTIONS_DIR
from src.seq.validate_predictions import validate_predictions


def write_seq_predictions(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    target: str = "dropout",
    window: int = 8,
    k: int = 3,
    split: str = "val",
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Format and serialize predictions in the exact Hard Rule 4 schema:
    ['learner_key', 'week', 'split', 'model', 'k', 'y_true', 'y_score']
    Automatically validates schema before returning.
    """
    if out_path is None:
        out_path = PREDICTIONS_DIR / f"{model_name}_w{window}_{target}.csv"
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure learner_key exists
    if "learner_key" in meta_df.columns:
        learner_keys = meta_df["learner_key"].to_numpy()
    else:
        learner_keys = (
            meta_df["code_module"] + "_" + meta_df["code_presentation"] + "_" + meta_df["id_student"].astype(str)
        ).to_numpy()

    weeks = meta_df["week"].to_numpy()
    N = len(meta_df)

    df_out = pd.DataFrame({
        "learner_key": learner_keys,
        "week": weeks,
        "split": [split] * N,
        "model": [f"{model_name}_w{window}_{target}"] * N,
        "k": [k] * N,
        "y_true": y_true,
        "y_score": np.clip(y_score, 0.0, 1.0),
    })

    # Validate schema
    validate_predictions(df_out, is_final_score=False)

    df_out.to_csv(out_path, index=False)
    print(f"Successfully saved and validated sequence predictions to: {out_path}")
    return out_path


def write_finalscore_predictions(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    window: int = 8,
    split: str = "val",
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Write multi-task final score regression predictions:
    ['learner_key', 'week', 'split', 'model', 'y_true', 'y_score']
    """
    if out_path is None:
        out_path = PREDICTIONS_DIR / f"{model_name}_w{window}_finalscore.csv"
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "learner_key" in meta_df.columns:
        learner_keys = meta_df["learner_key"].to_numpy()
    else:
        learner_keys = (
            meta_df["code_module"] + "_" + meta_df["code_presentation"] + "_" + meta_df["id_student"].astype(str)
        ).to_numpy()

    weeks = meta_df["week"].to_numpy()
    N = len(meta_df)

    df_out = pd.DataFrame({
        "learner_key": learner_keys,
        "week": weeks,
        "split": [split] * N,
        "model": [f"{model_name}_w{window}_finalscore"] * N,
        "y_true": y_true,
        "y_score": y_score,
    })

    validate_predictions(df_out, is_final_score=True)

    df_out.to_csv(out_path, index=False)
    print(f"Successfully saved and validated final-score predictions to: {out_path}")
    return out_path
