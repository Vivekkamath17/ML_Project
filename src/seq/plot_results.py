"""
Publication figure generator for Phase 4 Sequence Models.
Produces:
1. reports/phase4/figures/metric_vs_model_ci.png
   Validation PR-AUC with 95% bootstrap confidence intervals comparing baselines and sequence models.
2. reports/phase4/figures/metric_by_snapshot_week.png
   Validation performance trajectory across early, mid, and late course weeks (weeks 4, 8, 12, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.seq.config import REPORTS_DIR

FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_metric_vs_model_ci(
    summary_csv: Path,
    output_path: Optional[Path] = None,
    target: str = "dropout",
) -> Path:
    """
    Plot Val PR-AUC with 95% paired bootstrap CIs across models.
    """
    if output_path is None:
        output_path = FIG_DIR / f"metric_vs_model_ci_{target}.png"

    df = pd.read_csv(summary_csv)
    target_df = df[df["target"] == target].copy()

    # Sort by PR-AUC descending
    target_df = target_df.sort_values("val_pr_auc", ascending=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    y_pos = np.arange(len(target_df))
    means = target_df["val_pr_auc"].values
    
    # Calculate error bars from CI bounds if available
    if "pr_auc_ci_lower" in target_df.columns and "pr_auc_ci_upper" in target_df.columns:
        xerr_low = np.maximum(0, means - target_df["pr_auc_ci_lower"].values)
        xerr_high = np.maximum(0, target_df["pr_auc_ci_upper"].values - means)
        xerr = np.vstack([xerr_low, xerr_high])
    else:
        xerr = None

    # Colors: baseline models in grey/navy, sequence models in coral/blue
    colors = []
    labels = []
    for _, row in target_df.iterrows():
        m_name = str(row.get("model", ""))
        var = str(row.get("variant", ""))
        label = f"{m_name.upper()} ({var})" if var and var != "nan" else m_name.upper()
        labels.append(label)
        if "lgbm" in m_name.lower():
            colors.append("#2b5c8f")
        elif "catboost" in m_name.lower() or "xgb" in m_name.lower():
            colors.append("#4a7bb0")
        else:
            colors.append("#d95f02")

    bars = ax.barh(y_pos, means, xerr=xerr, align="center", color=colors, alpha=0.85, capsize=4, edgecolor="black", linewidth=0.8)

    # Add benchmark line for LightGBM v3
    lgbm_rows = target_df[target_df["model"].str.contains("lgbm", case=False, na=False)]
    if not lgbm_rows.empty:
        lgbm_val = lgbm_rows["val_pr_auc"].values[0]
        ax.axvline(lgbm_val, color="#2b5c8f", linestyle="--", linewidth=1.5, label=f"LightGBM v3 Baseline ({lgbm_val:.4f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Validation PR-AUC (Higher is Better)", fontsize=11, fontweight="bold")
    ax.set_title(f"Model Comparison with 95% Confidence Intervals (Target: {target.capitalize()})", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)

    # Annotate bar values
    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")
    return output_path


def plot_metric_by_snapshot_week(
    cohort_week_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    target: str = "dropout",
) -> Path:
    """
    Plot metric trajectory across course snapshot weeks.
    """
    if output_path is None:
        output_path = FIG_DIR / f"metric_by_snapshot_week_{target}.png"

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    weeks = cohort_week_df["week"].values
    pr_aucs = cohort_week_df["pr_auc"].values
    roc_aucs = cohort_week_df["roc_auc"].values

    ax1.plot(weeks, pr_aucs, marker="o", color="#d95f02", linewidth=2.2, label="PR-AUC")
    ax1.set_xlabel("Course Snapshot Week", fontsize=11, fontweight="bold")
    ax1.set_ylabel("PR-AUC", color="#d95f02", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#d95f02")

    ax2 = ax1.twinx()
    ax2.plot(weeks, roc_aucs, marker="s", color="#2b5c8f", linewidth=2.0, linestyle="--", label="ROC-AUC")
    ax2.set_ylabel("ROC-AUC", color="#2b5c8f", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#2b5c8f")

    # Mark key intervention weeks
    for kw in [4, 8, 12]:
        if kw in weeks:
            ax1.axvline(kw, color="gray", linestyle=":", alpha=0.7)

    plt.title(f"Validation Discrimination by Course Snapshot Week ({target.capitalize()})", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")
    return output_path
