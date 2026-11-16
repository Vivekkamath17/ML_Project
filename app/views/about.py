"""
Page 6: About, Methodology, Protocol & Governance.
"""
import streamlit as st
from app.data_loader import discover_prediction_inventory


def render_about_page():
    st.title("📖 Project Methodology, Governance & Limitations")
    st.markdown(
        "A rigorous, scientifically transparent early attrition warning and grade forecasting framework on the "
        "Open University Learning Analytics Dataset (OULAD)."
    )
    
    inventory = discover_prediction_inventory()
    
    # 1. Project Implementation Status
    st.subheader("1. Pipeline Implementation Status")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### ✅ Completed & Validated Components")
        st.markdown(
            "- **Phase 0 & 1**: Raw data audit, temporal point-in-time leak prevention, student cohort definition.\n"
            "- **Phase 2**: Baseline zoo (Prior, Recency, Clicks, Score, LR, LightGBM v1–v3).\n"
            "- **Phase 3**: Tree model zoo (LightGBM v3, CatBoost, XGBoost, Random Forest with 5-seed protocol).\n"
            "- **Calibration Suite**: Platt, Isotonic, and Temperature scaling calibration on validation predictions.\n"
            "- **Algorithmic Fairness Audit**: Comprehensive equity evaluation across 6 demographic attributes and intersectional pairs.\n"
            "- **Streamlit Analytics Dashboard**: Cached interactive exploration suite."
        )
    with col_s2:
        st.markdown("#### ⏳ Pending Teammate Deliverables")
        st.markdown(
            "- **Sequence & Deep Learning Models**: LSTM / Transformer sequence models under training by Teammate.\n"
            "- **SHAP Top-5 Parquet Files**: Feature attribution explanations under generation (Contract: `reports/shap/<model>_k3_val_top5.parquet`).\n"
            "- **Final Test Split Evaluation**: Evaluation on sealed presentation `2014J` pending final unsealing."
        )

    st.markdown("---")
    
    # 2. Strict Holdout Protocol
    st.subheader("2. Strict Holdout Protocol & Test Split Isolation")
    st.markdown(
        "To prevent scientific contamination and optimistic reporting bias, the project strictly adheres to temporal split isolation:\n"
        "- **Train Set (`2013B`, `2013J`)**: $N = 11,835$ learners, 330,940 learner-weeks. Used exclusively for hyperparameter tuning and model fitting.\n"
        "- **Validation Set (`2014B`)**: $N = 6,474$ learners, 164,799 learner-weeks. Used for model selection, calibration, fairness audits, and dashboard visualization.\n"
        "- **Test Set (`2014J`)**: **SEALED**. Evaluated only once at project completion. Unsealing test data is strictly prohibited during ongoing development."
    )

    st.markdown("---")
    
    # 3. Honest Methodological Limitations
    st.subheader("3. Explicit Scientific & Operational Limitations")
    st.markdown(
        "1. **Single Validation Cohort (`2014B`)**:\n"
        "   All audit numbers and operational trade-offs reflect a February intake ('B' presentation). "
        "   Course pacing, student demographics, and withdrawal dynamics differ from October ('J') intakes.\n\n"
        "2. **Modest Operational Lift (1.4x – 1.9x)**:\n"
        "   At a learner level, the best model achieves only a **1.4x to 1.9x lift** over random selection. "
        "   While statistically significant, early warning systems cannot reliably predict quiet, non-behavioral student departures.\n\n"
        "3. **Assessment-Driven Signal Shift**:\n"
        "   Platform clicks and login frequency provide weak predictive signal before week 6 (MCC ~ 0.14). "
        "   From week 8 onwards, model importance is overwhelmingly dominated by assessment submission scores (`weighted_score_to_date`).\n\n"
        "4. **Proxy Interactions**:\n"
        "   VLE clickstreams capture digital resource engagement but cannot record offline study groups, text-message communications, or external life challenges."
    )

    st.markdown("---")
    
    # 4. Teammate Collaboration Contracts
    st.subheader("4. Teammate Collaboration Contracts (App Integration)")
    
    st.markdown("#### A. SHAP Top-5 Feature Contributions")
    st.code(
        "# Target location:\n"
        "reports/shap/<model>_k3_val_top5.parquet\n\n"
        "# Schema:\n"
        "code_module: str\n"
        "code_presentation: str\n"
        "id_student: int64\n"
        "week: int64\n"
        "feature: str\n"
        "shap_value: float64\n"
        "rank: int64 (1 to 5)",
        language="yaml"
    )
    
    st.markdown("#### B. Multi-Task Forecasting Predictions")
    st.code(
        "# Continuous Score Regression:\n"
        "predictions/score/<model>_w<week>_val.parquet\n"
        "Columns: code_module, code_presentation, id_student, week, y_true, y_pred\n\n"
        "# Multi-Class Grade Band Forecasting:\n"
        "predictions/grade_band/<model>_w<week>_val.parquet\n"
        "Columns: code_module, code_presentation, id_student, week, y_true, y_score_fail, y_score_pass, y_score_distinction",
        language="yaml"
    )

    st.markdown("---")
    
    # 5. Repository Documentation Links
    st.subheader("5. Key Project Reports & References")
    st.markdown(
        "- **Phase 2 Baseline Modeling**: `reports/phase2/phase2_summary.md`\n"
        "- **Phase 3 Model Zoo & 5-Seed Benchmark**: `reports/phase3/phase3_summary.md`\n"
        "- **Phase 4 Calibration Report**: `reports/niranjan/summary.md`\n"
        "- **Algorithmic Fairness Audit Report**: `reports/fairness/fairness_summary.md`\n"
        "- **Team Collaboration Contract**: `HOWTO.md`\n"
        "- **Dataset Documentation**: `docs/data_card.md`"
    )
