"""
OULAD Early Warning & Fairness Suite - Main Streamlit Application Entrypoint.
Run with: streamlit run app/main.py
"""
import streamlit as st
from app.data_loader import discover_prediction_inventory
from app.components import render_sealed_test_banner
from app.views.overview import render_overview_page
from app.views.cohort import render_cohort_page
from app.views.learner import render_learner_page
from app.views.fairness import render_fairness_page
from app.views.calibration import render_calibration_page
from app.views.about import render_about_page

# Page configuration
st.set_page_config(
    page_title="OULAD Predictive Analytics & Fairness Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Header & Navigation
st.sidebar.title("🎓 OULAD Early Warning Suite")
st.sidebar.caption("Machine Learning for Attrition Early Warning & Algorithmic Fairness")

inventory = discover_prediction_inventory()

# Mandatory Sealed-Test protocol banner
render_sealed_test_banner(inventory)

# Sidebar page selector
pages = {
    "1. Overview & Model Comparison": render_overview_page,
    "2. Cohort Risk & Caseload Policy": render_cohort_page,
    "3. Learner Trajectory & Early Warning": render_learner_page,
    "4. Algorithmic Fairness Audit": render_fairness_page,
    "5. Calibration & Policy Reliability": render_calibration_page,
    "6. Methodology, Contracts & Governance": render_about_page,
}

selected_page_name = st.sidebar.radio(
    "Navigation",
    list(pages.keys()),
    index=0
)

# Sidebar metadata summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Inventory Overview")
st.sidebar.markdown(
    f"- **Discovered Prediction Files**: `{inventory['files_count']}`\n"
    f"- **Targets**: `{', '.join(inventory['targets'])}`\n"
    f"- **Horizons**: `{', '.join(str(h) for h in inventory['horizons'])}`\n"
    f"- **Evaluation Split**: `val (2014B)`\n"
    f"- **Test Status**: `{'Unsealed' if inventory['test_split_present'] else 'Sealed (Pending)'}`"
)
st.sidebar.caption("OULAD College ML Project | Tanmay's Downstream Pipeline")

# Render selected page view
pages[selected_page_name]()
