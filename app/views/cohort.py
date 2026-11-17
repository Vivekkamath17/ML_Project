"""
Page 2: Cohort Risk & Flagging Policy Simulation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.data_loader import (
    load_cohort_summary, load_policy_tradeoffs, load_learner_index
)


def render_cohort_page():
    st.title("👥 Cohort Risk & Operational Flagging Policy")
    st.markdown(
        "Simulate operational caseload policies, evaluate trade-offs between advisor outreach capacity and recall, "
        "and inspect risk distributions across course modules and demographic cohorts."
    )
    
    cohort = load_cohort_summary()
    policy_df = load_policy_tradeoffs()
    learner_df = load_learner_index()
    
    # 1. Official Cohort Denominators KPI Bar
    st.markdown("### 1. Population & Cohort Denominators (Validation Split: 2014B)")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Total Enrolled Learners", f"{cohort['total_enrolled_learners']:,}")
    with col_k2:
        st.metric("Unique Eventual Dropouts", f"{cohort['total_eventual_withdrawn']:,}")
    with col_k3:
        st.metric("Learner-Level Base Rate", f"{cohort['learner_base_rate']*100:.2f}%")
    with col_k4:
        st.metric("Weekly Positive Event Rate", f"{cohort['week_base_rate']*100:.2f}%")
        
    st.markdown("---")
    
    # 2. Interactive Flagging Policy Simulator
    st.subheader("2. Operational Flagging Policy Simulator")
    st.markdown(
        "Advising services operate under finite caseload capacity. Choose a policy type to simulate real-time caseload, precision, recall, and learner-level lift."
    )
    
    col_pol1, col_pol2 = st.columns([1, 2])
    with col_pol1:
        policy_mode = st.radio(
            "Policy Mechanism",
            ["Weekly Capacity Cut (Top-X% per week)", "Fixed Probability Threshold (Risk >= p)"],
            index=0
        )
        
        if "Capacity" in policy_mode:
            capacity_pct = st.slider("Weekly Caseload Capacity (Top-X%)", min_value=1.0, max_value=30.0, value=10.0, step=1.0)
            sub_pol = policy_df[(policy_df["policy_type"] == "weekly_capacity") & (policy_df["parameter"] == capacity_pct)]
        else:
            prob_th = st.slider("Predicted Dropout Probability Threshold", min_value=0.01, max_value=0.40, value=0.035, step=0.005)
            # Find closest parameter
            policy_df_sub = policy_df[policy_df["policy_type"] == "prob_threshold"].copy()
            policy_df_sub["diff"] = (policy_df_sub["parameter"] - prob_th).abs()
            sub_pol = policy_df_sub.sort_values("diff").head(1)
            
    with col_pol2:
        if not sub_pol.empty:
            cur = sub_pol.iloc[0]
            st.markdown(f"#### Active Policy Metrics ({cur['policy_type']}: `{cur['parameter']}`)")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Flagged Learner-Weeks", f"{int(cur['n_rows_flagged']):,}", f"{cur['selection_rate']*100:.1f}% cohort")
                st.metric("Weekly Alert Precision", f"{cur['precision']*100:.2f}%", f"{cur['lift_over_random']:.2f}x lift")
            with col_m2:
                st.metric("Weekly Alert Recall (TPR)", f"{cur['recall']*100:.2f}%")
                st.metric("Weekly False Positive Rate", f"{cur['fpr']*100:.2f}%")
            with col_m3:
                st.metric("Unique Learners Flagged >= 1x", f"{int(cur['unique_learners_flagged']):,}", f"out of {cohort['total_enrolled_learners']:,}")
                st.metric("Eventual Dropouts Caught", f"{int(cur['unique_withdrawn_caught']):,}", f"{cur['learner_recall']*100:.1f}% recall")
                st.metric("Learner Lift over Random", f"{cur['learner_lift']:.2f}x")
        else:
            st.info("Policy trade-off simulation data unavailable.")
            
    # Trade-off curve
    st.markdown("#### Precision-Recall Trade-off Curve as Capacity Scales")
    cap_curve_data = policy_df[policy_df["policy_type"] == "weekly_capacity"].sort_values("parameter")
    if not cap_curve_data.empty:
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Scatter(
            x=cap_curve_data["parameter"], y=cap_curve_data["recall"] * 100,
            mode="lines+markers", name="Weekly Recall (%)", line=dict(color="#2ca02c", width=2.5)
        ))
        fig_cap.add_trace(go.Scatter(
            x=cap_curve_data["parameter"], y=cap_curve_data["precision"] * 100,
            mode="lines+markers", name="Weekly Precision (%)", line=dict(color="#1f77b4", width=2.5)
        ))
        fig_cap.add_trace(go.Scatter(
            x=cap_curve_data["parameter"], y=cap_curve_data["learner_recall"] * 100,
            mode="lines+markers", name="Unique Dropout Recall (%)", line=dict(color="#ff7f0e", width=2.5, dash="dash")
        ))
        fig_cap.add_vline(
            x=10.0,
            line_dash="dot",
            line_color="gray",
            annotation_text="Default 10% Policy",
            annotation_position="top right",
            annotation=dict(
                font=dict(size=10, color="#555555"),
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="rgba(150, 150, 150, 0.5)",
                borderwidth=1,
                borderpad=3
            )
        )
        fig_cap.update_layout(
            xaxis=dict(title=dict(text="Weekly Caseload Capacity (Top-X%)", standoff=15), automargin=True),
            yaxis=dict(title=dict(text="Metric (%)", standoff=15), automargin=True),
            hovermode="x unified",
            height=460,
            margin=dict(l=60, r=30, t=40, b=90),
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
        st.plotly_chart(fig_cap, use_container_width=True)

    st.markdown("---")
    
    # 3. Cohort Filters and Risk Tiers
    st.subheader("3. Cohort Filters & Risk Tier Distribution")
    st.markdown(
        "Risk tiers are defined operationally from weekly caseload capacity cuts (rather than arbitrary absolute thresholds):\n\n"
        "- **Tier 1 (Critical Risk)**: Top 5% weekly caseload.\n"
        "- **Tier 2 (High Risk)**: 5%–10% weekly caseload.\n"
        "- **Tier 3 (Moderate Risk)**: 10%–20% weekly caseload.\n"
        "- **Tier 4 (Standard / Low Risk)**: Bottom 80% weekly caseload."
    )
    
    if not learner_df.empty:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sel_module = st.selectbox("Filter Course Module", ["All Modules"] + sorted(learner_df["code_module"].unique().tolist()))
        with col_f2:
            sel_gender = st.selectbox("Filter Gender", ["All Genders"] + sorted(learner_df["gender"].dropna().unique().tolist()))
        with col_f3:
            sel_edu = st.selectbox("Filter Highest Education", ["All Education Levels"] + sorted(learner_df["highest_education"].dropna().unique().tolist()))
            
        filtered_learners = learner_df.copy()
        if sel_module != "All Modules":
            filtered_learners = filtered_learners[filtered_learners["code_module"] == sel_module]
        if sel_gender != "All Genders":
            filtered_learners = filtered_learners[filtered_learners["gender"] == sel_gender]
        if sel_edu != "All Education Levels":
            filtered_learners = filtered_learners[filtered_learners["highest_education"] == sel_edu]
            
        st.write(f"Showing **{len(filtered_learners):,}** learners matching filters (Withdrawals: **{int(filtered_learners['is_withdrawn'].sum()):,}**).")
        
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig_hist = px.histogram(
                filtered_learners,
                x="max_risk",
                nbins=40,
                color="is_withdrawn",
                labels={"max_risk": "Peak Predicted Dropout Probability", "is_withdrawn": "Withdrawn?"},
                title="Distribution of Peak Predicted Risk per Student",
                color_discrete_map={1: "#d62728", 0: "#1f77b4"}
            )
            fig_hist.update_layout(
                xaxis=dict(title=dict(text="Peak Predicted Dropout Probability", standoff=15), automargin=True),
                yaxis=dict(title=dict(text="Learner Count", standoff=15), automargin=True),
                margin=dict(l=50, r=20, t=50, b=80),
                height=420,
                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5, font=dict(size=11))
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_h2:
            # Module counts
            mod_counts = filtered_learners.groupby("code_module").agg(
                Total=("id_student", "count"),
                Withdrawn=("is_withdrawn", "sum")
            ).reset_index()
            mod_counts["Dropout_Rate"] = mod_counts["Withdrawn"] / mod_counts["Total"]
            fig_bar = px.bar(
                mod_counts,
                x="code_module",
                y="Total",
                text="Total",
                title="Enrolled Learners by Course Module",
                labels={"code_module": "Module", "Total": "Enrolled Learners"}
            )
            fig_bar.update_layout(
                xaxis=dict(title=dict(text="Module", standoff=15), automargin=True),
                yaxis=dict(title=dict(text="Enrolled Learners", standoff=15), automargin=True),
                margin=dict(l=50, r=20, t=50, b=60),
                height=420
            )
            st.plotly_chart(fig_bar, use_container_width=True)
