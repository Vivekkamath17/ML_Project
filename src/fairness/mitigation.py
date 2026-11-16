"""
Fairness Mitigation Experiment Module for OULAD.

Evaluates post-processing mitigation (group-specific threshold adjustment) on a 50/50
learner-split of the validation cohort to quantify the trade-off in overall recall and precision.
Scientific note: Worded strictly as 'measured and reduced', never 'fixed'.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.fairness.audit import (
    compute_group_metrics,
    load_audit_data,
    load_config,
)

logger = logging.getLogger(__name__)


def run_mitigation_experiment(
    pred_path: Union[str, Path] = "predictions/dropout/k3/lgbm_v3_val.parquet",
    demographics_path: str = "data/processed/audit_demographics.parquet",
    output_dir: str = "reports/fairness",
    target_attribute: str = "highest_education",
    capacity_fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate group-specific threshold mitigation vs global threshold baseline
    on a held-out learner-split of validation.
    """
    df = load_audit_data(pred_path, demographics_path=demographics_path)
    model_name = str(df["model"].iloc[0])

    if target_attribute not in df.columns:
        raise ValueError(f"Attribute '{target_attribute}' not found in audit dataset.")

    # 1. 50/50 Learner-level split of validation data
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    train_idx, eval_idx = next(gss.split(df, groups=df["id_student"]))

    df_tune = df.iloc[train_idx].copy()
    df_eval = df.iloc[eval_idx].copy()

    # 2. Baseline: Single global capacity threshold (Top-10% weekly cut)
    eval_weekly_th = df_eval.groupby("week")["y_prob"].quantile(1.0 - capacity_fraction)
    df_eval["flag_baseline"] = (df_eval["y_prob"] >= df_eval["week"].map(eval_weekly_th)).astype(int)

    # Compute baseline metrics on eval split
    base_m = compute_group_metrics(
        df_eval["y_true"].to_numpy(int),
        df_eval["y_prob"].to_numpy(float),
        df_eval["flag_baseline"].to_numpy(int),
    )

    # Subgroup baseline disparity
    groups = sorted(df_eval[target_attribute].dropna().unique())
    base_group_sel = []
    base_group_tpr = []
    base_group_fpr = []
    for g in groups:
        gdf = df_eval[df_eval[target_attribute] == g]
        gm = compute_group_metrics(
            gdf["y_true"].to_numpy(int),
            gdf["y_prob"].to_numpy(float),
            gdf["flag_baseline"].to_numpy(int),
        )
        base_group_sel.append(gm["selection_rate"])
        base_group_tpr.append(gm["recall_tpr"])
        base_group_fpr.append(gm["fpr"])

    base_dp_diff = float(np.max(base_group_sel) - np.min(base_group_sel))
    base_tpr_gap = float(np.max(base_group_tpr) - np.min(base_group_tpr))
    base_fpr_gap = float(np.max(base_group_fpr) - np.min(base_group_fpr))
    base_eo_diff = float(max(base_tpr_gap, base_fpr_gap))

    # 3. Mitigation: Learn group-specific weekly cuts on df_tune to equalize group selection rates
    tune_thresholds = (
        df_tune.groupby(["week", target_attribute])["y_prob"]
        .quantile(1.0 - capacity_fraction)
        .rename("group_th")
    )
    overall_weekly_th = df_tune.groupby("week")["y_prob"].quantile(1.0 - capacity_fraction)

    # Merge thresholds onto eval split
    df_eval = df_eval.merge(tune_thresholds, on=["week", target_attribute], how="left")
    df_eval["group_th"] = df_eval["group_th"].fillna(df_eval["week"].map(overall_weekly_th)).fillna(0.035)
    df_eval["flag_mitigated"] = (df_eval["y_prob"] >= df_eval["group_th"]).astype(int)

    # Compute mitigated metrics on eval split
    mit_m = compute_group_metrics(
        df_eval["y_true"].to_numpy(int),
        df_eval["y_prob"].to_numpy(float),
        df_eval["flag_mitigated"].to_numpy(int),
    )

    mit_group_sel = []
    mit_group_tpr = []
    mit_group_fpr = []
    for g in groups:
        gdf = df_eval[df_eval[target_attribute] == g]
        gm = compute_group_metrics(
            gdf["y_true"].to_numpy(int),
            gdf["y_prob"].to_numpy(float),
            gdf["flag_mitigated"].to_numpy(int),
        )
        mit_group_sel.append(gm["selection_rate"])
        mit_group_tpr.append(gm["recall_tpr"])
        mit_group_fpr.append(gm["fpr"])

    mit_dp_diff = float(np.max(mit_group_sel) - np.min(mit_group_sel))
    mit_tpr_gap = float(np.max(mit_group_tpr) - np.min(mit_group_tpr))
    mit_fpr_gap = float(np.max(mit_group_fpr) - np.min(mit_group_fpr))
    mit_eo_diff = float(max(mit_tpr_gap, mit_fpr_gap))

    rows = [
        {
            "model": model_name,
            "target_attribute": target_attribute,
            "policy": "Global Baseline (Top-10% Cohort)",
            "selection_rate": base_m["selection_rate"],
            "overall_precision": base_m["precision"],
            "overall_recall": base_m["recall_tpr"],
            "demographic_parity_difference": base_dp_diff,
            "equalized_odds_difference": base_eo_diff,
            "tpr_gap": base_tpr_gap,
            "fpr_gap": base_fpr_gap,
            "note": "Standard whole-cohort capacity cut",
        },
        {
            "model": model_name,
            "target_attribute": target_attribute,
            "policy": "Group-Specific Thresholds (Mitigated)",
            "selection_rate": mit_m["selection_rate"],
            "overall_precision": mit_m["precision"],
            "overall_recall": mit_m["recall_tpr"],
            "demographic_parity_difference": mit_dp_diff,
            "equalized_odds_difference": mit_eo_diff,
            "tpr_gap": mit_tpr_gap,
            "fpr_gap": mit_fpr_gap,
            "note": "Tuned on 50% val split, evaluated out-of-sample on remaining 50% (measured and reduced, in-sample on val presentation)",
        },
    ]

    out_df = pd.DataFrame(rows)
    out_path = Path(output_dir) / "mitigation_experiment.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved mitigation experiment results to {out_path}")
    return out_df
