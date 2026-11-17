"""
Reusable UI Components and Plotly Visualizations for Streamlit App.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def render_sealed_test_banner(inventory: Dict[str, Any]):
    """Render mandatory sealed-test notification banner."""
    if inventory.get("test_split_present", False):
        st.info("ℹ️ **Test Split Evaluation Detected**: Final test predictions (`2014J`) discovered in repository.")
    else:
        st.warning(
            "🔒 **Strict Holdout Protocol**: The final test split (`2014J`) is sealed. "
            "All metrics and curves shown are computed exclusively on the temporal validation split (`2014B`, $N=6,474$ enrolled learners, weeks >= 2). "
            "Final test results: **Pending teammate submission**."
        )


def render_tie_callout():
    """Render scientific notice regarding statistical ties among tree models."""
    st.info(
        "📊 **Empirical Equivalence Notice**: Under 500-iteration learner-clustered paired bootstrap tests (ΔPR and ΔROC), "
        "**all four tree architectures (LightGBM v3, CatBoost, XGBoost, and Random Forest) are statistically tied** on validation performance (~0.078 PR-AUC, ~0.776 ROC-AUC). "
        "Differences fall entirely within sampling variability across the student cohort."
    )


def plot_roc_curves(curves_df: pd.DataFrame, selected_models: List[str]) -> go.Figure:
    """Plot interactive multi-model ROC curves."""
    fig = go.Figure()
    
    # 50% chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="rgba(150, 150, 150, 0.5)", dash="dash"),
        name="Random Classifier (AUC = 0.50)"
    ))
    
    color_map = {
        "lgbm_v3": "#1f77b4",
        "catboost": "#2ca02c",
        "xgb": "#ff7f0e",
        "rf": "#9467bd",
        "lr_39feat_unweighted": "#d62728",
    }
    
    for m in selected_models:
        sub = curves_df[(curves_df["model_id"] == m) & (curves_df["curve_type"] == "ROC")]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"],
                mode="lines",
                name=m,
                line=dict(color=color_map.get(m, "#333333"), width=2.5),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>" + m + "</extra>"
            ))
            
    fig.update_layout(
        title=dict(text="Validation ROC Curves (2014B, weeks >= 2)", font=dict(size=15)),
        xaxis=dict(title=dict(text="False Positive Rate (1 - Specificity)", standoff=15), range=[-0.02, 1.02], automargin=True),
        yaxis=dict(title=dict(text="True Positive Rate (Recall)", standoff=15), range=[-0.02, 1.02], automargin=True),
        hovermode="x unified",
        height=480,
        margin=dict(l=60, r=30, t=50, b=100),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(200, 200, 200, 0.4)",
            borderwidth=1,
        )
    )
    return fig


def plot_pr_curves(curves_df: pd.DataFrame, selected_models: List[str], base_rate: float) -> go.Figure:
    """Plot interactive multi-model Precision-Recall curves with base-rate benchmark."""
    fig = go.Figure()
    
    # Base rate horizontal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[base_rate, base_rate],
        mode="lines",
        line=dict(color="rgba(200, 50, 50, 0.6)", dash="dash"),
        name=f"Cohort Base Rate ({base_rate*100:.2f}%)"
    ))
    
    color_map = {
        "lgbm_v3": "#1f77b4",
        "catboost": "#2ca02c",
        "xgb": "#ff7f0e",
        "rf": "#9467bd",
        "lr_39feat_unweighted": "#d62728",
    }
    
    for m in selected_models:
        sub = curves_df[(curves_df["model_id"] == m) & (curves_df["curve_type"] == "PR")]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"],
                mode="lines",
                name=m,
                line=dict(color=color_map.get(m, "#333333"), width=2.5),
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra>" + m + "</extra>"
            ))
            
    fig.update_layout(
        title=dict(text="Validation Precision-Recall Curves (Primary Ranking Metric)", font=dict(size=15)),
        xaxis=dict(title=dict(text="Recall (Sensitivity)", standoff=15), range=[-0.02, 1.02], automargin=True),
        yaxis=dict(title=dict(text="Precision (PPV)", standoff=15), range=[-0.02, 1.02], automargin=True),
        hovermode="x unified",
        height=480,
        margin=dict(l=60, r=30, t=50, b=100),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(200, 200, 200, 0.4)",
            borderwidth=1,
        )
    )
    return fig


def plot_learner_trajectory(
    traj_df: pd.DataFrame,
    student_id: int,
    module: str,
    withdrawn: bool
) -> go.Figure:
    """Plot weekly predicted risk trajectory for a student with flagged periods."""
    fig = go.Figure()
    
    prob_cols = [c for c in traj_df.columns if c.startswith("prob_")]
    color_map = {
        "prob_lgbm_v3": ("#1f77b4", "LightGBM v3 (Official)"),
        "prob_catboost": ("#2ca02c", "CatBoost"),
        "prob_xgb": ("#ff7f0e", "XGBoost"),
        "prob_rf": ("#9467bd", "Random Forest"),
        "prob_lr_39feat_unweighted": ("#d62728", "Logistic Regression"),
        "prob_gru_multi_dropout": ("#17becf", "GRU Multi-Task (w=8, Dropout)"),
        "prob_lstm_dropout": ("#e377c2", "LSTM (w=4, Dropout)"),
        "prob_tcn_dropout": ("#bcbd22", "TCN (w=4, Dropout)"),
        "prob_lstm_onset": ("#8c564b", "LSTM (w=8, Onset Champion)"),
    }
    
    for c in prob_cols:
        color, label = color_map.get(c, ("#7f7f7f", c.replace("prob_", "")))
        fig.add_trace(go.Scatter(
            x=traj_df["week"], y=traj_df[c],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate="Week %{x}: Predicted Risk = %{y:.3f}<extra>" + label + "</extra>"
        ))
        
    # Highlight weeks flagged by policy
    if "flagged_top10" in traj_df.columns:
        flagged_weeks = traj_df[traj_df["flagged_top10"] == 1]
        if not flagged_weeks.empty:
            fig.add_trace(go.Scatter(
                x=flagged_weeks["week"],
                y=flagged_weeks["prob_lgbm_v3"],
                mode="markers",
                name="Flagged (Top-10% Caseload)",
                marker=dict(symbol="star", size=14, color="#e377c2", line=dict(color="black", width=1.5)),
                hovertemplate="Week %{x}: ⚠️ FLAGGED FOR INTERVENTION<extra></extra>"
            ))
            
    # Add horizontal threshold guidance
    fig.add_hline(
        y=0.035, line_dash="dot", line_color="gray",
        annotation_text="Nominal weekly threshold (~0.035)",
        annotation_position="top left",
        annotation=dict(
            font=dict(size=10, color="#555555"),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(150, 150, 150, 0.5)",
            borderwidth=1,
            borderpad=3
        )
    )
    
    status_text = "WITHDRAWN" if withdrawn else "COMPLETED / ACTIVE"
    status_color = "red" if withdrawn else "green"
    
    fig.update_layout(
        title=dict(
            text=f"Learner {student_id} ({module}) Risk Trajectory — True Outcome: <span style='color:{status_color}'><b>{status_text}</b></span>",
            font=dict(size=15)
        ),
        xaxis=dict(title=dict(text="Course Week (Observed History)", standoff=15), automargin=True, dtick=2),
        yaxis=dict(
            title=dict(text="Predicted 3-Week Dropout Probability", standoff=15),
            range=[0, max(0.25, traj_df[prob_cols].max().max() * 1.15) if not traj_df.empty else 1.0],
            automargin=True
        ),
        hovermode="x unified",
        height=520,
        margin=dict(l=60, r=30, t=60, b=130),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(200, 200, 200, 0.4)",
            borderwidth=1
        )
    )
    return fig


def plot_reliability_diagram(rel_df: pd.DataFrame, selected_models: List[str]) -> go.Figure:
    """Plot calibration reliability diagram (observed frequency vs mean predicted risk)."""
    fig = go.Figure()
    
    # Perfect calibration 45-deg line
    fig.add_trace(go.Scatter(
        x=[0, 0.4], y=[0, 0.4],
        mode="lines",
        line=dict(color="black", dash="dash"),
        name="Perfect Calibration"
    ))
    
    for m in selected_models:
        sub_raw = rel_df[(rel_df["model_id"] == m) & (rel_df["method"] == "uncalibrated")]
        if not sub_raw.empty:
            fig.add_trace(go.Scatter(
                x=sub_raw["prob_pred"], y=sub_raw["prob_true"],
                mode="lines+markers",
                name=f"{m} (Uncalibrated)",
                line=dict(dash="dot"),
                marker=dict(size=7)
            ))
            
        sub_cal = rel_df[(rel_df["model_id"] == m) & (rel_df["method"] == "isotonic")]
        if not sub_cal.empty:
            fig.add_trace(go.Scatter(
                x=sub_cal["prob_pred"], y=sub_cal["prob_true"],
                mode="lines+markers",
                name=f"{m} (Isotonic Calibrated)",
                line=dict(width=2.5),
                marker=dict(size=7)
            ))
            
    fig.update_layout(
        title=dict(text="Calibration Reliability Diagram (Val 2014B)", font=dict(size=15)),
        xaxis=dict(title=dict(text="Mean Predicted Probability (Bin)", standoff=15), range=[0, 0.35], automargin=True),
        yaxis=dict(title=dict(text="Observed Empirical Dropout Rate", standoff=15), range=[0, 0.35], automargin=True),
        height=500,
        margin=dict(l=60, r=30, t=60, b=120),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(200, 200, 200, 0.4)",
            borderwidth=1
        )
    )
    return fig
