"""
Fairness Audit Module for OULAD Prediction Pipeline.

Evaluates demographic parity, equalized odds, subgroup calibration, and
learner-level policy capture across protected demographic attributes.
Strict rule: Demographics are joined strictly at evaluation time and
never exposed to model feature spaces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/app.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_audit_data(
    pred_path: Union[str, Path],
    demographics_path: str = "data/processed/audit_demographics.parquet",
    min_week: int = 2,
) -> pd.DataFrame:
    """
    Load model predictions and join demographic attributes strictly for evaluation.
    Verifies that no rows are silently lost or duplicated.
    """
    pred_path = Path(pred_path)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    demo_path = Path(demographics_path)
    if not demo_path.is_file():
        raise FileNotFoundError(f"Demographics table not found: {demo_path}")

    # Read predictions
    df_pred = pq.read_table(pred_path).to_pandas()
    if "week" in df_pred.columns:
        df_pred = df_pred[df_pred["week"] >= min_week].copy()

    # Read demographics
    df_demo = pq.read_table(demo_path).to_pandas()

    learner_keys = ["code_module", "code_presentation", "id_student"]
    pred_unique_learners = df_pred[learner_keys].drop_duplicates()
    n_pred_learners = len(pred_unique_learners)

    # Left join strictly on composite learner key
    merged = df_pred.merge(df_demo, on=learner_keys, how="left")

    # Audit join completeness
    unmatched_mask = merged["gender"].isna()
    unmatched_count = unmatched_mask.sum()
    if unmatched_count > 0:
        raise ValueError(
            f"Fairness Audit Join Failure: {unmatched_count} rows in predictions "
            f"could not be matched to audit_demographics!"
        )

    assert len(merged) == len(df_pred), (
        f"Row count mismatch after join: {len(merged)} != {len(df_pred)}"
    )

    # Ensure missing imd_band is categorized as 'Unknown'
    if "imd_band" in merged.columns:
        merged["imd_band"] = merged["imd_band"].fillna("Unknown")

    return merged


def apply_cohort_capacity_policy(
    df: pd.DataFrame,
    capacity_fraction: float = 0.10,
    prob_col: str = "y_prob",
    week_col: str = "week",
) -> pd.DataFrame:
    """
    Apply weekly capacity top-X% flagging policy over the WHOLE cohort.
    The threshold is computed on the entire cohort each week (not per group).
    Vectorized implementation using per-week quantiles.
    """
    df = df.copy()
    flag_col = f"flag_top_{int(capacity_fraction * 100)}"

    # Compute exact weekly threshold matching capacity_fraction
    weekly_th = df.groupby(week_col)[prob_col].quantile(1.0 - capacity_fraction)
    df[flag_col] = (df[prob_col] >= df[week_col].map(weekly_th)).astype(int)
    return df


def compute_group_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    flag: np.ndarray,
) -> Dict[str, float]:
    """
    Compute point metrics for a demographic subgroup.
    """
    n_total = len(y_true)
    n_pos = int(np.sum(y_true))
    base_rate = float(n_pos / n_total) if n_total > 0 else 0.0

    n_flagged = int(np.sum(flag))
    selection_rate = float(n_flagged / n_total) if n_total > 0 else 0.0

    tp = int(np.sum((y_true == 1) & (flag == 1)))
    fp = int(np.sum((y_true == 0) & (flag == 1)))
    fn = int(np.sum((y_true == 1) & (flag == 0)))
    tn = int(np.sum((y_true == 0) & (flag == 0)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # ROC-AUC & PR-AUC
    if n_pos > 0 and n_pos < n_total:
        try:
            roc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc = 0.5
        try:
            pr = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr = base_rate
    else:
        roc = 0.5
        pr = base_rate

    mean_pred = float(np.mean(y_prob)) if n_total > 0 else 0.0
    cal_ratio = float(mean_pred / base_rate) if base_rate > 0 else 1.0

    return {
        "n_rows": n_total,
        "n_positives": n_pos,
        "base_rate": base_rate,
        "selection_rate": selection_rate,
        "precision": prec,
        "recall_tpr": tpr,
        "fpr": fpr,
        "roc_auc": roc,
        "pr_auc": pr,
        "mean_pred_risk": mean_pred,
        "calibration_ratio": cal_ratio,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_group_bootstrap_ci(
    group_df: pd.DataFrame,
    flag_col: str,
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Ultra-fast learner-clustered 95% bootstrap confidence intervals for group metrics
    using pre-sorted arrays and vectorized cumulative integration.
    """
    codes, uniques = pd.factorize(group_df["id_student"])
    n_clusters = len(uniques)

    if n_clusters < 5:
        return {
            m: (np.nan, np.nan)
            for m in ["roc_auc", "pr_auc", "selection_rate", "recall_tpr", "fpr", "precision", "calibration_ratio"]
        }

    raw_yp = group_df["y_prob"].to_numpy(dtype=np.float64)
    raw_yt = group_df["y_true"].to_numpy(dtype=np.float64)
    raw_fl = group_df[flag_col].to_numpy(dtype=np.float64)

    # Pre-sort descending by prediction probability once
    order = np.argsort(-raw_yp)
    y_p = raw_yp[order]
    y_t = raw_yt[order]
    fl = raw_fl[order]
    sorted_codes = codes[order]

    # Pre-generate bootstrap resample cluster matrices
    rng = np.random.default_rng(seed)
    all_sampled = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))

    sel_rates = []
    tprs = []
    precs = []
    fprs = []
    cal_ratios = []
    rocs = []
    prs = []

    fl_pos = fl * y_t
    fl_neg = fl * (1.0 - y_t)

    for b in range(n_resamples):
        counts = np.bincount(all_sampled[b], minlength=n_clusters)
        w = counts[sorted_codes].astype(np.float64)
        
        w_pos = w * y_t
        total_pos = np.sum(w_pos)
        total_w = np.sum(w)
        total_neg = total_w - total_pos

        if total_pos <= 0 or total_neg <= 0:
            continue

        tp = np.sum(w * fl_pos)
        fp = np.sum(w * fl_neg)
        flagged = tp + fp

        sel_rates.append(flagged / total_w)
        tprs.append(tp / total_pos)
        precs.append(tp / flagged if flagged > 0 else 0.0)
        fprs.append(fp / total_neg)

        mean_p = np.sum(w * y_p) / total_w
        mean_y = total_pos / total_w
        cal_ratios.append(mean_p / mean_y if mean_y > 0 else 1.0)

        # Vectorized trapezoidal ROC-AUC and PR-AUC
        cum_w = np.cumsum(w)
        cum_pos = np.cumsum(w_pos)
        
        # Unique threshold collapse points
        diff_mask = np.diff(y_p, append=y_p[-1] - 1.0) != 0.0
        c_pos = cum_pos[diff_mask]
        c_w = cum_w[diff_mask]
        c_neg = c_w - c_pos

        # ROC: True Positive Rate vs False Positive Rate
        tpr_curve = np.concatenate(([0.0], c_pos / total_pos))
        fpr_curve = np.concatenate(([0.0], c_neg / total_neg))
        roc_val = float(np.sum(np.diff(fpr_curve) * (tpr_curve[:-1] + tpr_curve[1:]) / 2.0))
        rocs.append(roc_val)

        # PR: Precision vs Recall
        rec_curve = np.concatenate(([0.0], c_pos / total_pos))
        prec_arr = np.divide(c_pos, c_w, out=np.zeros_like(c_pos), where=c_w > 0)
        prec_curve = np.concatenate(([prec_arr[0] if len(prec_arr) > 0 else 0.0], prec_arr))
        pr_val = float(np.sum(np.diff(rec_curve) * (prec_curve[:-1] + prec_curve[1:]) / 2.0))
        prs.append(pr_val)

    def get_ci(arr):
        if len(arr) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "selection_rate": get_ci(sel_rates),
        "recall_tpr": get_ci(tprs),
        "precision": get_ci(precs),
        "fpr": get_ci(fprs),
        "calibration_ratio": get_ci(cal_ratios),
        "roc_auc": get_ci(rocs),
        "pr_auc": get_ci(prs),
    }


def compute_learner_level_policy_metrics(
    group_df: pd.DataFrame,
    flag_col: str,
) -> Dict[str, Any]:
    """
    Compute learner-level version of the policy metrics:
    - Learner flagged at least once across weeks.
    - Recall of eventually withdrawn learners.
    - Explicit denominators stated.
    """
    grouped = group_df.groupby(["code_module", "code_presentation", "id_student"]).agg({
        "y_true": "max",
        flag_col: "max",
    }).reset_index()

    total_learners = len(grouped)
    total_dropouts = int(grouped["y_true"].sum())

    flagged_learners = int(grouped[flag_col].sum())
    caught_dropouts = int(((grouped["y_true"] == 1) & (grouped[flag_col] == 1)).sum())

    learner_selection_rate = float(flagged_learners / total_learners) if total_learners > 0 else 0.0
    learner_recall = float(caught_dropouts / total_dropouts) if total_dropouts > 0 else 0.0
    learner_precision = float(caught_dropouts / flagged_learners) if flagged_learners > 0 else 0.0

    return {
        "total_learners": total_learners,
        "total_dropouts": total_dropouts,
        "flagged_learners": flagged_learners,
        "caught_dropouts": caught_dropouts,
        "learner_selection_rate": learner_selection_rate,
        "learner_recall": learner_recall,
        "learner_precision": learner_precision,
    }


def audit_single_model(
    pred_path: Union[str, Path],
    demographics_path: str = "data/processed/audit_demographics.parquet",
    config_path: str = "config/app.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform full fairness audit for a single model across all demographic attributes.

    Returns:
    --------
    metrics_df: Detailed subgroup metrics table (long format).
    gaps_df: Disparity and fairness gaps table across attributes.
    """
    cfg = load_config(config_path)
    f_cfg = cfg["fairness"]
    attributes = f_cfg["attributes"]
    capacity_fraction = f_cfg.get("capacity_fraction", 0.10)
    min_week = f_cfg.get("min_week", 2)
    n_resamples = f_cfg.get("n_bootstrap", 500)
    seed = f_cfg.get("random_seed", 42)
    min_learners = f_cfg["stability"]["min_learners"]
    min_positives = f_cfg["stability"]["min_positives"]

    # 1. Load data
    df = load_audit_data(pred_path, demographics_path=demographics_path, min_week=min_week)
    model_name = str(df["model"].iloc[0])
    horizon = int(df["k"].iloc[0])

    # 2. Apply whole-cohort capacity policy
    flag_col = f"flag_top_{int(capacity_fraction * 100)}"
    df = apply_cohort_capacity_policy(df, capacity_fraction=capacity_fraction)

    # Overall cohort baseline metrics
    y_true_all = df["y_true"].to_numpy(dtype=int)
    y_prob_all = df["y_prob"].to_numpy(dtype=float)
    flag_all = df[flag_col].to_numpy(dtype=int)
    overall_metrics = compute_group_metrics(y_true_all, y_prob_all, flag_all)
    overall_sel_rate = overall_metrics["selection_rate"]

    subgroup_rows = []
    gaps_rows = []

    for attr in attributes:
        if attr not in df.columns:
            logger.warning(f"Attribute {attr} not found in dataset; skipping.")
            continue

        groups = sorted(df[attr].dropna().unique())
        group_sel_rates = []
        group_tprs = []
        group_fprs = []
        group_rocs = []

        for grp in groups:
            gdf = df[df[attr] == grp].copy()
            n_learners = len(gdf["id_student"].unique())
            y_t = gdf["y_true"].to_numpy(dtype=int)
            y_p = gdf["y_prob"].to_numpy(dtype=float)
            fl = gdf[flag_col].to_numpy(dtype=int)

            m = compute_group_metrics(y_t, y_p, fl)
            n_pos = m["n_positives"]
            is_unstable = (n_learners < min_learners) or (n_pos < min_positives)

            ci = compute_group_bootstrap_ci(gdf, flag_col=flag_col, n_resamples=n_resamples, seed=seed)
            l_m = compute_learner_level_policy_metrics(gdf, flag_col=flag_col)

            disparate_impact_ratio = (m["selection_rate"] / overall_sel_rate) if overall_sel_rate > 0 else 1.0

            row = {
                "model": model_name,
                "horizon": horizon,
                "attribute": attr,
                "group": str(grp),
                "n_learners": n_learners,
                "n_rows": m["n_rows"],
                "n_positives": n_pos,
                "base_rate": m["base_rate"],
                "is_unstable": is_unstable,
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
                "calibration_ratio_ci_lower": ci["calibration_ratio"][0],
                "calibration_ratio_ci_upper": ci["calibration_ratio"][1],
                "disparate_impact_ratio": disparate_impact_ratio,
                # Learner-level metrics
                "learner_total_n": l_m["total_learners"],
                "learner_dropouts_n": l_m["total_dropouts"],
                "learner_flagged_n": l_m["flagged_learners"],
                "learner_caught_n": l_m["caught_dropouts"],
                "learner_selection_rate": l_m["learner_selection_rate"],
                "learner_recall": l_m["learner_recall"],
                "learner_precision": l_m["learner_precision"],
            }
            subgroup_rows.append(row)

            group_sel_rates.append(m["selection_rate"])
            group_tprs.append(m["recall_tpr"])
            group_fprs.append(m["fpr"])
            group_rocs.append(m["roc_auc"])

        # Compute Fairness Gaps for this attribute
        dp_diff = float(np.max(group_sel_rates) - np.min(group_sel_rates))
        tpr_gap = float(np.max(group_tprs) - np.min(group_tprs))
        fpr_gap = float(np.max(group_fprs) - np.min(group_fprs))
        eq_odds_diff = float(max(tpr_gap, fpr_gap))
        roc_gap = float(np.max(group_rocs) - np.min(group_rocs))

        # Check Fairlearn verification if available
        if FAIRLEARN_AVAILABLE:
            fl_dp = float(demographic_parity_difference(y_true_all, flag_all, sensitive_features=df[attr]))
            fl_eo = float(equalized_odds_difference(y_true_all, flag_all, sensitive_features=df[attr]))
        else:
            fl_dp = dp_diff
            fl_eo = eq_odds_diff

        gaps_rows.append({
            "model": model_name,
            "horizon": horizon,
            "attribute": attr,
            "n_groups": len(groups),
            "demographic_parity_difference": dp_diff,
            "fairlearn_dp_diff": fl_dp,
            "equalized_odds_difference": eq_odds_diff,
            "fairlearn_eo_diff": fl_eo,
            "max_min_tpr_gap": tpr_gap,
            "max_min_fpr_gap": fpr_gap,
            "max_min_roc_gap": roc_gap,
            "worst_group_sel_rate": float(np.min(group_sel_rates)),
            "best_group_sel_rate": float(np.max(group_sel_rates)),
            "worst_group_tpr": float(np.min(group_tprs)),
            "best_group_tpr": float(np.max(group_tprs)),
        })

    return pd.DataFrame(subgroup_rows), pd.DataFrame(gaps_rows)


def run_fairness_audit(
    predictions_dir: str = "predictions",
    demographics_path: str = "data/processed/audit_demographics.parquet",
    output_dir: str = "reports/fairness",
    config_path: str = "config/app.yaml",
    include_calibration: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discover all dropout models and run full fairness audit across all discovered models.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_dir = Path(predictions_dir)
    # Search for all val dropout prediction files
    dropout_files = sorted(list(p_dir.glob("dropout/k*/*_val.parquet")))
    if include_calibration:
        # Include calibrated prediction files
        calib_files = sorted(list(p_dir.glob("calibration/*_val.parquet")))
        all_files = dropout_files + calib_files
    else:
        all_files = dropout_files

    if not all_files:
        raise FileNotFoundError(f"No validation dropout prediction files found in {predictions_dir}")

    all_metrics = []
    all_gaps = []

    for f in all_files:
        logger.info(f"Auditing fairness for: {f}")
        try:
            m_df, g_df = audit_single_model(f, demographics_path=demographics_path, config_path=config_path)
            all_metrics.append(m_df)
            all_gaps.append(g_df)
        except Exception as e:
            logger.error(f"Error auditing {f}: {e}")
            raise e

    final_metrics_df = pd.concat(all_metrics, ignore_index=True)
    final_gaps_df = pd.concat(all_gaps, ignore_index=True)

    metrics_out = out_dir / "fairness_metrics.csv"
    gaps_out = out_dir / "fairness_gaps.csv"

    final_metrics_df.to_csv(metrics_out, index=False)
    final_gaps_df.to_csv(gaps_out, index=False)
    logger.info(f"Saved fairness metrics to {metrics_out}")
    logger.info(f"Saved fairness gaps to {gaps_out}")

    return final_metrics_df, final_gaps_df
