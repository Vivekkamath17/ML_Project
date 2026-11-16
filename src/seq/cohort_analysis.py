"""
Cohort and demographic subgroup analysis module for Phase 4.
Evaluates sequence model predictions across demographic and academic subgroups:
- STEM vs Non-STEM modules
- Snapshot weeks (e.g. week 4, week 8, week 12)
- Previous attempts (0 vs 1+)
- IMD deprivation bands (deprived vs affluent)
- Gender and Disability
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.seq.config import DATA_DIR, VAL_PRESENTATIONS

STEM_MODULES = {"CCC", "DDD", "EEE", "FFF"}


def load_val_demographics() -> pd.DataFrame:
    """Load demographic and cohort metadata for validation split strictly."""
    demo_df = (
        pl.scan_csv(DATA_DIR / "raw" / "studentInfo.csv")
        .filter(pl.col("code_presentation").is_in(list(VAL_PRESENTATIONS)))
        .select([
            "code_module",
            "code_presentation",
            "id_student",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "disability",
        ])
        .with_columns([
            (pl.col("code_module") + "_" + pl.col("code_presentation") + "_" + pl.col("id_student").cast(pl.String)).alias("learner_key"),
            pl.col("code_module").is_in(list(STEM_MODULES)).alias("is_stem"),
            (pl.col("num_of_prev_attempts") > 0).alias("has_prev_attempts"),
        ])
        .collect()
        .to_pandas()
    )
    return demo_df


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC, PR-AUC, Brier score safely."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"n": len(y_true), "roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "base_rate": float(np.mean(y_true)) if len(y_true) > 0 else np.nan}
    return {
        "n": len(y_true),
        "base_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def analyze_cohorts(pred_df: pd.DataFrame, demo_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """
    Perform deep cohort analysis on validation predictions.
    Returns dictionary of DataFrames:
      - 'overall'
      - 'by_week'
      - 'by_stem'
      - 'by_prev_attempts'
      - 'by_imd'
      - 'by_gender'
      - 'by_disability'
    """
    if demo_df is None:
        demo_df = load_val_demographics()

    merged = pred_df.merge(demo_df, on="learner_key", how="inner")

    results: Dict[str, pd.DataFrame] = {}

    # Overall
    overall = compute_metrics(merged["y_true"].to_numpy(), merged["y_score"].to_numpy())
    results["overall"] = pd.DataFrame([overall])

    # By Week
    week_rows = []
    for week, group in merged.groupby("week"):
        m = compute_metrics(group["y_true"].to_numpy(), group["y_score"].to_numpy())
        m["week"] = week
        week_rows.append(m)
    results["by_week"] = pd.DataFrame(week_rows).sort_values("week")

    # By STEM
    stem_rows = []
    for is_stem, group in merged.groupby("is_stem"):
        m = compute_metrics(group["y_true"].to_numpy(), group["y_score"].to_numpy())
        m["cohort"] = "STEM" if is_stem else "Arts/Social Science"
        stem_rows.append(m)
    results["by_stem"] = pd.DataFrame(stem_rows)

    # By Previous Attempts
    prev_rows = []
    for has_prev, group in merged.groupby("has_prev_attempts"):
        m = compute_metrics(group["y_true"].to_numpy(), group["y_score"].to_numpy())
        m["cohort"] = "1+ Previous Attempts" if has_prev else "First Attempt (0)"
        prev_rows.append(m)
    results["by_prev_attempts"] = pd.DataFrame(prev_rows)

    # By IMD Band
    imd_rows = []
    for imd, group in merged.groupby("imd_band"):
        if pd.isna(imd) or imd == "?":
            continue
        m = compute_metrics(group["y_true"].to_numpy(), group["y_score"].to_numpy())
        m["imd_band"] = imd
        imd_rows.append(m)
    results["by_imd"] = pd.DataFrame(imd_rows)

    # By Gender
    gender_rows = []
    for g, group in merged.groupby("gender"):
        m = compute_metrics(group["y_true"].to_numpy(), group["y_score"].to_numpy())
        m["gender"] = g
        gender_rows.append(m)
    results["by_gender"] = pd.DataFrame(gender_rows)

    # By Disability
    dis_rows = []
    for d, group in merged.groupby("disability"):
        m = compute_metrics(group["y_true"].to_numpy(), group["y_score"].to_numpy())
        m["disability"] = d
        dis_rows.append(m)
    results["by_disability"] = pd.DataFrame(dis_rows)

    return results
