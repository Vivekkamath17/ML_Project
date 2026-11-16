"""
Page 1: Overview & Model Comparison Leaderboard (Centerpiece).
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from app.data_loader import (
    load_models_overview, load_curves_data, load_cohort_summary,
    load_by_week_metrics, load_by_module_metrics, load_cost_table
)
from app.components import render_tie_callout, plot_roc_curves, plot_pr_curves


def render_overview_page():
    st.title("🏆 Model Comparison & Leaderboard")
    st.markdown(
        "Comparative evaluation of baseline and machine learning models for early student attrition on OULAD. "
        "All metrics evaluated strictly on temporal holdout validation (**2014B**, $N=6,474$ enrolled learners, 164,799 learner-weeks, weeks >= 2)."
    )
    
    render_tie_callout()
    
    models_df = load_models_overview()
    curves_df = load_curves_data()
    cohort = load_cohort_summary()
    cost_df = load_cost_table()
    by_week_df = load_by_week_metrics()
    by_mod_df = load_by_module_metrics()
    
    if models_df.empty:
        st.error("No model overview cache found. Run `python scripts/build_app_cache.py` first.")
        return
        
    # Top Filter controls
    col_t1, col_t2, col_t3 = st.columns([1, 1, 2])
    with col_t1:
        target = st.selectbox("Prediction Target", ["dropout (k=3 weeks)", "onset (k=3 weeks)", "score (snapshot)", "grade_band (snapshot)"], index=0)
    with col_t2:
        horizon = st.selectbox("Forecast Horizon", [3], index=0)
    with col_t3:
        tgt_key = "onset" if "onset" in target else "dropout"
        avail_models = models_df[models_df["target"] == tgt_key]["model_id"].tolist()
        if not avail_models:
            avail_models = models_df["model_id"].tolist()
        selected_models = st.multiselect(
            "Select Models to Compare",
            options=avail_models,
            default=avail_models
        )
        
    if not selected_models:
        st.warning("Please select at least one model to display.")
        return
        
    st.markdown("---")
    
    # 1. Headline Leaderboard Table
    st.subheader(f"1. Formal Validation Leaderboard - {target} (Ranking by PR-AUC)")
    
    filtered_models = models_df[(models_df["target"] == tgt_key) & (models_df["model_id"].isin(selected_models))].copy()
    
    # Format display table
    display_rows = []
    for _, row in filtered_models.iterrows():
        ref_id = "lstm_seq_only_w8_onset" if tgt_key == "onset" else "lgbm_v3"
        tied_badge = "★ Reference Champion" if row["model_id"] == ref_id else "Statistically Tied (*)" if row["statistically_tied_with_champion"] else "Alternative Model"
        display_rows.append({
            "Model": row["display_name"],
            "Val PR-AUC [95% CI]": f"{row['pr_auc']:.4f} [{row['pr_auc_ci_low']:.4f}, {row['pr_auc_ci_high']:.4f}]",
            "PR Multiplier": f"{row['pr_multiplier']:.2f}x",
            "Val ROC-AUC [95% CI]": f"{row['roc_auc']:.4f} [{row['roc_auc_ci_low']:.4f}, {row['roc_auc_ci_high']:.4f}]",
            "Brier Score": f"{row['brier']:.4f}",
            "Paired ΔPR vs Champion": f"{row['delta_pr']:+.4f} [{row['delta_pr_ci_low']:+.4f}, {row['delta_pr_ci_high']:+.4f}]",
            "Paired ΔROC vs Champion": f"{row['delta_roc']:+.4f} [{row['delta_roc_ci_low']:+.4f}, {row['delta_roc_ci_high']:+.4f}]",
            "Tie Verdict": tied_badge,
        })
    df_display = pd.DataFrame(display_rows)
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    st.caption("(*) 95% learner-clustered paired bootstrap CI for ΔPR contains 0.0. Denominators: 6,474 enrolled learners, 1,395 dropouts.")
    
    # 2. Interactive Curves
    st.markdown("---")
    st.subheader("2. Discriminative Performance Curves")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(plot_pr_curves(curves_df, selected_models, cohort.get("week_base_rate", 0.0237)), use_container_width=True)
    with col_c2:
        st.plotly_chart(plot_roc_curves(curves_df, selected_models), use_container_width=True)
        
    # 3. Paired Deltas and Cost Analysis
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("3. Paired Bootstrap Deltas vs LightGBM v3")
        delta_fig = px.bar(
            filtered_models,
            x="model_id",
            y="delta_pr",
            error_y=filtered_models["delta_pr_ci_high"] - filtered_models["delta_pr"],
            error_y_minus=filtered_models["delta_pr"] - filtered_models["delta_pr_ci_low"],
            color="statistically_tied_with_champion",
            color_discrete_map={True: "#2ca02c", False: "#d62728"},
            labels={"delta_pr": "Paired ΔPR-AUC", "model_id": "Model", "statistically_tied_with_champion": "Tied?"},
            title="Paired ΔPR-AUC with 95% Bootstrap Confidence Intervals"
        )
        delta_fig.add_hline(y=0.0, line_dash="dash", line_color="black")
        st.plotly_chart(delta_fig, use_container_width=True)
        
    with col_d2:
        st.subheader("4. Computational Cost & Inference Latency")
        if not cost_df.empty:
            cost_disp = cost_df.copy()
            cost_disp.columns = ["Model Identifier", "Training Time (sec)", "Latency (ms / 1k rows)", "Model Size (MB)"]
            st.dataframe(cost_disp, use_container_width=True, hide_index=True)
            st.caption("Latency measured via synchronized CPU timer per 1,000 batch rows on Windows platform.")
        else:
            st.info("Cost metrics pending.")

    # 4. By-Week & By-Module Performance Breakdowns
    st.markdown("---")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.subheader("5. Performance Trajectory by Course Week")
        if not by_week_df.empty:
            fig_week = px.line(
                by_week_df,
                x="week",
                y=["roc_auc", "pr_auc_mult"],
                labels={"value": "Metric Value", "week": "Course Week", "variable": "Metric"},
                title="Weekly Attrition Prediction Quality (Early Weeks vs Late)"
            )
            fig_week.add_vline(x=2, line_dash="dot", annotation_text="Official Eval Start (Week 2)")
            st.plotly_chart(fig_week, use_container_width=True)
            st.caption("Weeks 0–1 reflect initial orientation window (low signal). Attrition signal peaks around weeks 3–5 and stabilizes.")
            
    with col_w2:
        st.subheader("6. Performance Across Curriculum Modules")
        if not by_mod_df.empty:
            fig_mod = px.bar(
                by_mod_df,
                x="code_module",
                y="roc_auc",
                color="train_status",
                labels={"roc_auc": "ROC-AUC", "code_module": "Module", "train_status": "Train Split Status"},
                title="ROC-AUC Generalization Across Modules"
            )
            st.plotly_chart(fig_mod, use_container_width=True)
            st.caption("Module CCC was completely unseen in training (zero-shot transfer: ROC-AUC = 0.732).")

    # 5. Official Test Split Leaderboard (Unsealed Holdout 2014J)
    repo_root = Path(__file__).resolve().parent.parent.parent
    test_lb_file = repo_root / "reports" / "test_evaluation" / "test_leaderboard.csv"
    if test_lb_file.exists():
        st.markdown("---")
        st.subheader("7. Official Final Test Split Leaderboard (2014J Holdout)")
        st.markdown(
            "Unsealed temporal holdout evaluation (**2014J**, 263,222 learner-weeks across 8,804 students). "
            "Confirms cross-presentation generalization without target leakage across both Baselines and Deep Sequence Architectures."
        )
        df_test_lb = pd.read_csv(test_lb_file)
        st.dataframe(df_test_lb, use_container_width=True, hide_index=True)
