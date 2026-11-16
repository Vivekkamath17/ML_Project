"""
Intersectional Fairness Analysis Module for OULAD.

Evaluates performance and fairness disparities across intersections of demographic attributes
(e.g., gender x age_band, gender x imd_band, gender x highest_education)
with explicit small-sample warnings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from src.fairness.audit import (
    apply_cohort_capacity_policy,
    compute_group_bootstrap_ci,
    compute_group_metrics,
    compute_learner_level_policy_metrics,
    load_audit_data,
    load_config,
)

logger = logging.getLogger(__name__)


def run_intersectional_audit(
    pred_path: Union[str, Path],
    demographics_path: str = "data/processed/audit_demographics.parquet",
    output_dir: str = "reports/fairness",
    config_path: str = "config/app.yaml",
) -> pd.DataFrame:
    """
    Compute intersectional fairness metrics across attribute pairs.
    """
    cfg = load_config(config_path)
    f_cfg = cfg["fairness"]
    pairs = f_cfg.get("intersectional_pairs", [["gender", "age_band"], ["gender", "imd_band"]])
    capacity_fraction = f_cfg.get("capacity_fraction", 0.10)
    min_week = f_cfg.get("min_week", 2)
    n_resamples = f_cfg.get("n_bootstrap", 500)
    seed = f_cfg.get("random_seed", 42)
    min_learners = f_cfg["stability"]["min_learners"]
    min_positives = f_cfg["stability"]["min_positives"]

    df = load_audit_data(pred_path, demographics_path=demographics_path, min_week=min_week)
    model_name = str(df["model"].iloc[0])
    horizon = int(df["k"].iloc[0])

    flag_col = f"flag_top_{int(capacity_fraction * 100)}"
    df = apply_cohort_capacity_policy(df, capacity_fraction=capacity_fraction)

    intersectional_rows = []

    for attr1, attr2 in pairs:
        if attr1 not in df.columns or attr2 not in df.columns:
            continue

        pair_name = f"{attr1}_x_{attr2}"
        df[pair_name] = df[attr1].astype(str) + " × " + df[attr2].astype(str)
        groups = sorted(df[pair_name].unique())

        for grp in groups:
            gdf = df[df[pair_name] == grp].copy()
            n_learners = len(gdf["id_student"].unique())
            y_t = gdf["y_true"].to_numpy(dtype=int)
            y_p = gdf["y_prob"].to_numpy(dtype=float)
            fl = gdf[flag_col].to_numpy(dtype=int)

            m = compute_group_metrics(y_t, y_p, fl)
            n_pos = m["n_positives"]

            is_unstable = (n_learners < min_learners) or (n_pos < min_positives)
            warning_reasons = []
            if n_learners < min_learners:
                warning_reasons.append(f"Learners < {min_learners} (found {n_learners})")
            if n_pos < min_positives:
                warning_reasons.append(f"Positives < {min_positives} (found {n_pos})")
            small_group_warning = "; ".join(warning_reasons) if warning_reasons else "None"

            ci = compute_group_bootstrap_ci(gdf, flag_col=flag_col, n_resamples=n_resamples, seed=seed)
            l_m = compute_learner_level_policy_metrics(gdf, flag_col=flag_col)

            row = {
                "model": model_name,
                "horizon": horizon,
                "pair": pair_name,
                "group": grp,
                "n_learners": n_learners,
                "n_rows": m["n_rows"],
                "n_positives": n_pos,
                "base_rate": m["base_rate"],
                "is_unstable": is_unstable,
                "small_group_warning": small_group_warning,
                "selection_rate": m["selection_rate"],
                "selection_rate_ci_lower": ci["selection_rate"][0],
                "selection_rate_ci_upper": ci["selection_rate"][1],
                "precision": m["precision"],
                "precision_ci_lower": ci["precision"][0],
                "precision_ci_upper": ci["precision"][1],
                "recall_tpr": m["recall_tpr"],
                "recall_tpr_ci_lower": ci["recall_tpr"][0],
                "recall_tpr_ci_upper": ci["recall_tpr"][1],
                "fpr": m["fpr"],
                "fpr_ci_lower": ci["fpr"][0],
                "fpr_ci_upper": ci["fpr"][1],
                "roc_auc": m["roc_auc"],
                "roc_auc_ci_lower": ci["roc_auc"][0],
                "roc_auc_ci_upper": ci["roc_auc"][1],
                "pr_auc": m["pr_auc"],
                "pr_auc_ci_lower": ci["pr_auc"][0],
                "pr_auc_ci_upper": ci["pr_auc"][1],
                "mean_pred_risk": m["mean_pred_risk"],
                "calibration_ratio": m["calibration_ratio"],
                "learner_total_n": l_m["total_learners"],
                "learner_dropouts_n": l_m["total_dropouts"],
                "learner_flagged_n": l_m["flagged_learners"],
                "learner_caught_n": l_m["caught_dropouts"],
                "learner_selection_rate": l_m["learner_selection_rate"],
                "learner_recall": l_m["learner_recall"],
                "learner_precision": l_m["learner_precision"],
            }
            intersectional_rows.append(row)

    out_df = pd.DataFrame(intersectional_rows)
    out_path = Path(output_dir) / "intersectional_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved intersectional metrics to {out_path}")
    return out_df
