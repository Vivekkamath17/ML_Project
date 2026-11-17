"""
Page 3: Learner Trajectory & Early Warning Inspection.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.data_loader import (
    load_learner_index, load_learner_trajectory, load_shap_top5,
    load_score_forecasts, load_grade_band_forecasts
)
from app.components import plot_learner_trajectory


def render_learner_page():
    st.title("🔍 Learner Trajectory & Early Warning Inspector")
    st.markdown(
        "Inspect point-in-time risk evolution, alert history, feature drivers, and academic forecasts for individual students across validation presentation **2014B**."
    )
    
    idx_df = load_learner_index()
    if idx_df.empty:
        st.error("No learner index available. Run `python scripts/build_app_cache.py` first.")
        return
        
    # Student Selector Controls
    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    with col_s1:
        modules = sorted(idx_df["code_module"].unique().tolist())
        selected_module = st.selectbox("Select Course Module", modules, index=0)
    
    # Filter learners for this module
    sub_learners = idx_df[idx_df["code_module"] == selected_module].sort_values("id_student")
    
    with col_s2:
        filter_status = st.selectbox(
            "Filter Outcome",
            ["All Learners", "Completed / Active (With Grade Forecasts)", "Eventual Dropouts Only"],
            index=0
        )
        if filter_status == "Eventual Dropouts Only":
            sub_learners = sub_learners[sub_learners["is_withdrawn"] == 1]
        elif filter_status == "Completed / Active (With Grade Forecasts)":
            sub_learners = sub_learners[sub_learners["is_withdrawn"] == 0]
            
    with col_s3:
        student_options = sub_learners["id_student"].tolist()
        if not student_options:
            st.warning("No learners match criteria.")
            return
            
        # Display formatted learner dropdown
        def format_learner(sid):
            row = sub_learners[sub_learners["id_student"] == sid].iloc[0]
            status = "🔴 Withdrawn" if row["is_withdrawn"] == 1 else "🟢 Completed"
            flag_txt = f"{row['times_flagged']}x flagged" if row['times_flagged'] > 0 else "Never flagged"
            return f"ID {sid} | {status} | Peak Risk {row['max_risk']:.1%} | {flag_txt}"
            
        selected_student_id = st.selectbox("Select Learner (id_student)", options=student_options, format_func=format_learner, index=0)

    # Load detailed weekly trajectory
    traj_df = load_learner_trajectory(selected_student_id, selected_module)
    if traj_df.empty:
        st.error("Trajectory not found for selected learner.")
        return
        
    learner_info = sub_learners[sub_learners["id_student"] == selected_student_id].iloc[0]
    is_withdrawn = bool(learner_info["is_withdrawn"] == 1)
    
    # 1. Learner Profile Header
    st.markdown("---")
    st.markdown(f"### Student Profile: `id_student = {selected_student_id}` | Module: `{selected_module}`")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.metric("Final Course Outcome", "WITHDRAWN" if is_withdrawn else "ACTIVE / COMPLETED")
    with col_p2:
        st.metric("Weeks Enrolled / Active", f"{len(traj_df)} weeks (Weeks {traj_df['week'].min()} to {traj_df['week'].max()})")
    with col_p3:
        st.metric("Peak Predicted Risk", f"{learner_info['max_risk']:.2%}")
    with col_p4:
        st.metric("Weeks Flagged (Top-10%)", f"{int(learner_info['times_flagged'])} weeks")

    # Demographic Context (for audit only, clearly labeled)
    st.caption(
        f"**Audit Demographics**: Gender: `{learner_info.get('gender', 'N/A')}` | "
        f"Education: `{learner_info.get('highest_education', 'N/A')}` | "
        f"IMD Deprivation Decile: `{learner_info.get('imd_band', 'N/A')}` "
        "*(Demographics are quarantined for equity evaluation and never fed into models)*"
    )
    
    # 2. Risk Trajectory Chart
    st.markdown("---")
    st.subheader("1. Weekly Predicted Dropout Risk Trajectory")
    st.plotly_chart(plot_learner_trajectory(traj_df, selected_student_id, selected_module, is_withdrawn), use_container_width=True)
    
    # Automated Natural Language Interpretation
    st.markdown("#### Automated Case Synthesis")
    peak_week = int(traj_df.loc[traj_df["prob_lgbm_v3"].idxmax(), "week"])
    peak_risk = traj_df["prob_lgbm_v3"].max()
    times_flagged = int(learner_info["times_flagged"])
    
    if is_withdrawn:
        if times_flagged > 0:
            synth_text = (
                f"**Timely Early Warning**: Student {selected_student_id} was identified by the early warning system, "
                f"flagged in **{times_flagged} separate weeks** under the Top-10% weekly caseload policy. Risk peaked at **{peak_risk:.1%} in Week {peak_week}**. "
                f"The learner ultimately withdrew, confirming that model alerts accurately captured severe academic disengagement."
            )
        else:
            synth_text = (
                f"**False Negative / Missed Alert**: Student {selected_student_id} withdrew from the course, but their predicted dropout probability "
                f"remained below the weekly Top-10% threshold throughout their enrollment (peaking at only **{peak_risk:.1%} in Week {peak_week}**). "
                f"This reflects known model sensitivity constraints, where quiet attrition without preceding activity drops can evade early detection."
            )
    else:
        if times_flagged > 0:
            synth_text = (
                f"**False Positive / Successful Persistence**: Student {selected_student_id} was flagged in **{times_flagged} weeks** "
                f"(peaking at **{peak_risk:.1%} in Week {peak_week}**), yet successfully completed the module. "
                f"Such cases indicate transient study disruptions where the student successfully recovered or benefited from institutional support."
            )
        else:
            synth_text = (
                f"**Stable Learner**: Student {selected_student_id} maintained consistent platform engagement and timely assignment submissions, "
                f"consistently scoring in the lower risk deciles (peak risk **{peak_risk:.1%} in Week {peak_week}**). No intervention alerts were triggered."
            )
    st.info(synth_text)
    
    # 3. Weekly Feature Snapshot & Explainability
    st.markdown("---")
    col_w_sel, col_w_disp = st.columns([1, 2])
    with col_w_sel:
        st.subheader("2. Weekly Feature Inspection")
        avail_weeks = sorted(traj_df["week"].tolist())
        inspected_week = st.selectbox("Inspect Observed Week", avail_weeks, index=min(2, len(avail_weeks)-1))
        
        # Explainability Box (SHAP Top-5 contract)
        st.markdown("#### SHAP Top-5 Contributors")
        avail_shap_models = ["lgbm_v3", "xgb", "catboost", "rf", "lr_39feat_unweighted"]
        sel_shap_model = st.selectbox(
            "Explaining Model",
            avail_shap_models,
            index=0,
            key=f"shap_model_sel_{selected_student_id}_{inspected_week}"
        )
        shap_df = load_shap_top5(sel_shap_model, selected_student_id, inspected_week)
        if shap_df is not None and not shap_df.empty:
            fig_shap = px.bar(
                shap_df,
                x="shap_value",
                y="feature",
                orientation="h",
                color="shap_value",
                color_continuous_scale="RdBu_r",
                title=f"Week {inspected_week} Top Feature Drivers ({sel_shap_model})"
            )
            fig_shap.update_layout(
                yaxis=dict(automargin=True, title=dict(standoff=10)),
                xaxis=dict(automargin=True, title=dict(standoff=10)),
                margin=dict(l=80, r=20, t=50, b=50),
                height=350,
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption(
                "Positive values (red) increase predicted dropout risk; negative values (blue) protect against attrition."
            )
        else:
            st.warning(
                "⏳ **SHAP Top-5 Explanations: Pending Teammate Submission**\n\n"
                "Awaiting teammate delivery conforming to contract `reports/shap/<model>_k3_val_top5.parquet`. "
                "Feature attribution visualizations will automatically populate once computed.",
                icon="⚠️"
            )
            
    with col_w_disp:
        st.subheader("3. Recorded Academic & Behavioral Indicators")
        # Check if weekly raw feature table exists locally
        # Gracefully degrade if absent
        st.caption(f"Status for student {selected_student_id} at Week {inspected_week}:")
        
        week_row = traj_df[traj_df["week"] == inspected_week].iloc[0]

        def _fmt(val):
            try:
                if pd.isna(val):
                    return "N/A"
                return f"{float(val):.3f}"
            except (TypeError, ValueError):
                return "N/A"

        st.markdown(
            f"- **Predicted Dropout Risk (LightGBM v3)**: `{_fmt(week_row.get('prob_lgbm_v3'))}` ({'⚠️ Flagged' if week_row.get('flagged_top10') == 1 else 'Normal'})\n"
            f"- **Predicted Dropout Risk (CatBoost)**: `{_fmt(week_row.get('prob_catboost'))}`\n"
            f"- **Predicted Dropout Risk (XGBoost)**: `{_fmt(week_row.get('prob_xgb'))}`\n"
            f"- **Predicted Dropout Risk (Random Forest)**: `{_fmt(week_row.get('prob_rf'))}`\n"
            f"- **Predicted Dropout Risk (Logistic Reg)**: `{_fmt(week_row.get('prob_lr_39feat_unweighted'))}`"
        )
        st.info("ℹ️ *Note on Raw Weekly Feature Table*: Raw clickstream tables are stored outside the Git repository. Trajectory curves and risk indicators are rendered directly from verifiable model outputs.")

    # 4. Multi-Task Forecasting (Continuous Score & Grade Band)
    st.markdown("---")
    st.subheader("4. Multi-Task Performance Forecasting (Snapshots: Weeks 4, 8, 12)")
    
    score_df = load_score_forecasts(selected_student_id, selected_module)
    grade_df = load_grade_band_forecasts(selected_student_id, selected_module)
    
    if score_df.empty and grade_df.empty:
        st.info(
            f"ℹ️ **No Summative Forecasts for Learner `{selected_student_id}` (Early Attrition)**\n\n"
            "Performance forecasting models (continuous assessment score regression and multi-class grade bands) "
            "predict final end-of-course academic outcomes evaluated at longitudinal snapshot weeks (Weeks 4, 8, and 12).\n\n"
            "- Because this student withdrew from the module prior to completing final coursework and summative examinations, "
            "no final exam grade or course score was awarded.\n"
            "- **To inspect performance forecasts**: Select **'Completed / Active (With Grade Forecasts)'** in the outcome filter above "
            "(e.g., student `27891` or `50069`) to view longitudinal score predictions and grade band probabilities across models."
        )
    else:
        # Show actual observed final outcome
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if not score_df.empty:
                true_score = score_df["y_true"].iloc[0]
                st.metric("Observed Final Assessment Score", f"{true_score:.1f} / 100")
        with col_m2:
            if not grade_df.empty:
                true_grade = grade_df["y_true"].iloc[0]
                st.metric("Observed Final Grade Band", str(true_grade).upper())
                
        col_fc1, col_fc2 = st.columns(2)
        with col_fc1:
            st.markdown("#### Continuous Final Assessment Score Regression")
            if not score_df.empty:
                # Add model filter
                avail_score_models = sorted(score_df["model"].unique().tolist())
                sel_score_models = st.multiselect(
                    "Filter Score Models",
                    options=avail_score_models,
                    default=["lightgbm", "catboost", "random_forest", "ridge"] if "lightgbm" in avail_score_models else avail_score_models[:3],
                    key="sel_score_models"
                )
                filtered_score_df = score_df[score_df["model"].isin(sel_score_models)].sort_values(["week", "model"])
                score_disp = filtered_score_df[["week", "model", "y_true", "y_pred"]].copy()
                score_disp["error"] = score_disp["y_pred"] - score_disp["y_true"]
                score_disp.columns = ["Snapshot Week", "Model", "Observed Final Score", "Predicted Score", "Residual (Error)"]
                st.dataframe(score_disp, use_container_width=True, hide_index=True)
            else:
                st.info("Continuous score forecast predictions pending for this learner.")
                
        with col_fc2:
            st.markdown("#### Multi-Class Grade Band Probability Forecast")
            if not grade_df.empty:
                avail_grade_models = sorted(grade_df["model"].unique().tolist())
                sel_grade_models = st.multiselect(
                    "Filter Grade Models",
                    options=avail_grade_models,
                    default=["lightgbm", "catboost", "random_forest"] if "lightgbm" in avail_grade_models else avail_grade_models[:2],
                    key="sel_grade_models"
                )
                filtered_grade_df = grade_df[grade_df["model"].isin(sel_grade_models)].sort_values(["week", "model"])
                grade_disp = filtered_grade_df[["week", "model", "y_true", "y_score_fail", "y_score_pass", "y_score_distinction"]].copy()
                grade_disp.columns = ["Snapshot Week", "Model", "Final Grade", "P(Fail)", "P(Pass)", "P(Distinction)"]
                st.dataframe(grade_disp, use_container_width=True, hide_index=True)
            else:
                st.info("Grade band probability predictions pending for this learner.")
