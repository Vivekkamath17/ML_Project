"""
Publication Figures Generator for Fairness Audit.
Produces clean, accessible matplotlib plots saved to reports/fairness/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_fairness_gaps(
    gaps_df: pd.DataFrame,
    output_dir: Union[str, Path] = "reports/fairness/figures",
    official_model: str = "lgbm_v3",
):
    """
    Bar chart showing Demographic Parity and Equalized Odds differences by attribute
    for the official model.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_model = gaps_df[gaps_df["model"] == official_model].copy()
    if df_model.empty:
        df_model = gaps_df.drop_duplicates(subset=["attribute"]).copy()

    df_model = df_model.sort_values("demographic_parity_difference", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    y_pos = np.arange(len(df_model))
    height = 0.35

    ax.barh(y_pos - height / 2, df_model["demographic_parity_difference"], height, label="Demographic Parity Gap (Δ Selection Rate)", color="#1f77b4")
    ax.barh(y_pos + height / 2, df_model["equalized_odds_difference"], height, label="Equalized Odds Gap (max(ΔTPR, ΔFPR))", color="#ff7f0e")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_model["attribute"], fontsize=10)
    ax.set_xlabel("Disparity Difference", fontsize=11)
    ax.set_title(f"Fairness Disparities by Demographic Attribute ({official_model})", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = out_dir / "fairness_gaps_by_attribute.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved figure: {fig_path}")


def plot_subgroup_metrics(
    metrics_df: pd.DataFrame,
    output_dir: Union[str, Path] = "reports/fairness/figures",
    official_model: str = "lgbm_v3",
):
    """
    Scatter plot of TPR vs FPR (Equalized Odds plane) across all demographic subgroups.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_m = metrics_df[metrics_df["model"] == official_model].copy()
    if df_m.empty:
        df_m = metrics_df.copy()

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    attributes = df_m["attribute"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(attributes)))

    for attr, color in zip(attributes, colors):
        sub = df_m[df_m["attribute"] == attr]
        ax.scatter(sub["fpr"], sub["recall_tpr"], label=attr, s=sub["n_learners"] / 10, color=color, alpha=0.7, edgecolors="none")

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.set_title(f"Equalized Odds Trade-Off Across Demographic Subgroups ({official_model})", fontsize=12, fontweight="bold")
    ax.legend(title="Attribute", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = out_dir / "tpr_fpr_disparity.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved figure: {fig_path}")


def plot_subgroup_calibration(
    metrics_df: pd.DataFrame,
    output_dir: Union[str, Path] = "reports/fairness/figures",
    official_model: str = "lgbm_v3",
):
    """
    Calibration plot: Mean Predicted Risk vs Observed Dropout Rate by demographic subgroup.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_m = metrics_df[metrics_df["model"] == official_model].copy()
    if df_m.empty:
        df_m = metrics_df.copy()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Reference diagonal
    max_val = max(df_m["base_rate"].max(), df_m["mean_pred_risk"].max()) * 1.15
    ax.plot([0, max_val], [0, max_val], "k--", label="Perfect Calibration (Ratio=1.0)", alpha=0.7)

    for attr in df_m["attribute"].unique():
        sub = df_m[df_m["attribute"] == attr]
        ax.scatter(sub["base_rate"], sub["mean_pred_risk"], label=attr, alpha=0.8, s=45)

    ax.set_xlabel("Observed Dropout Rate (Base Rate)", fontsize=11)
    ax.set_ylabel("Mean Predicted Risk", fontsize=11)
    ax.set_title(f"Subgroup Calibration: Predicted vs Observed Risk ({official_model})", fontsize=12, fontweight="bold")
    ax.legend(title="Attribute", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = out_dir / "subgroup_calibration.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved figure: {fig_path}")


def generate_all_figures(
    metrics_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    output_dir: str = "reports/fairness/figures",
    official_model: str = "lgbm_v3",
):
    plot_fairness_gaps(gaps_df, output_dir=output_dir, official_model=official_model)
    plot_subgroup_metrics(metrics_df, output_dir=output_dir, official_model=official_model)
    plot_subgroup_calibration(metrics_df, output_dir=output_dir, official_model=official_model)
