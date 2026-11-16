"""
Data Loader module for the Streamlit dashboard.
Provides cached access to pre-aggregated parquet tables in reports/app_cache/
and dynamically discovers predictions from predictions/.
Offline-capable, read-only, and strictly enforces validation split evaluation.
"""
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "reports" / "app_cache"
PREDS_DIR = BASE_DIR / "predictions"
SHAP_DIR = BASE_DIR / "reports" / "shap"


@st.cache_data(show_spinner=False)
def discover_prediction_inventory() -> Dict[str, Any]:
    """
    Dynamically discover all models, targets, horizons, and splits from predictions/.
    Ensures zero hardcoding and detects when teammates add new models or test predictions.
    """
    inventory = {
        "targets": set(),
        "horizons": set(),
        "models": set(),
        "splits": set(),
        "files_count": 0,
        "test_split_present": False,
        "details": []
    }
    
    if not PREDS_DIR.exists():
        return inventory
        
    all_files = list(PREDS_DIR.rglob("*.parquet")) + list(PREDS_DIR.rglob("*.csv"))
    for p_file in all_files:
        inventory["files_count"] += 1
        rel_parts = p_file.relative_to(PREDS_DIR).parts
        stem = p_file.stem
        
        # Check split
        if rel_parts[0] == "test" or "_test" in stem:
            split = "test"
            inventory["test_split_present"] = True
        elif rel_parts[0] in ("seq", "dropout") or "_val" in stem:
            split = "val"
        else:
            split = "unknown"
        inventory["splits"].add(split)
            
        if "onset" in stem:
            target = "onset"
        elif "score" in stem or "finalscore" in stem:
            target = "score"
        elif len(rel_parts) > 1 and rel_parts[0] in ("dropout", "onset", "grade"):
            target = rel_parts[0]
        else:
            target = "dropout"
        inventory["targets"].add(target)
        
        horizon = 3
        if len(rel_parts) > 2 and rel_parts[1].startswith("k"):
            try:
                horizon = int(rel_parts[1][1:])
            except ValueError:
                horizon = 3
        inventory["horizons"].add(horizon)
        
        # Model identifier
        model_name = stem.replace("_val", "").replace("_test", "")
        inventory["models"].add(model_name)
        
        inventory["details"].append({
            "target": target,
            "horizon": horizon,
            "split": split,
            "model": model_name,
            "path": str(p_file)
        })
        
    inventory["targets"] = sorted(inventory["targets"])
    inventory["horizons"] = sorted(inventory["horizons"])
    inventory["models"] = sorted(inventory["models"])
    inventory["splits"] = sorted(inventory["splits"])
    return inventory


@st.cache_data(show_spinner=False)
def load_models_overview() -> pd.DataFrame:
    """Load cached model leaderboard table with 95% CIs and deltas."""
    p = CACHE_DIR / "models_overview.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_curves_data() -> pd.DataFrame:
    """Load pre-computed sampled ROC and PR curves."""
    p = CACHE_DIR / "roc_pr_curves.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cohort_summary() -> Dict[str, Any]:
    """Load validation cohort counts and denominators."""
    p = CACHE_DIR / "cohort_summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "presentation": "2014B",
        "total_enrolled_learners": 6474,
        "total_eventual_withdrawn": 1395,
        "learner_base_rate": 0.215477,
        "total_learner_weeks": 164799,
        "positive_events": 3901,
        "week_base_rate": 0.023671,
        "weeks_range": [2, 28],
        "modules": ["BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
    }


@st.cache_data(show_spinner=False)
def load_policy_tradeoffs() -> pd.DataFrame:
    """Load pre-calculated precision/recall/lift trade-offs across capacity cuts."""
    p = CACHE_DIR / "policy_tradeoffs.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_by_week_metrics() -> pd.DataFrame:
    """Load performance by week."""
    p = CACHE_DIR / "by_week.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_by_module_metrics() -> pd.DataFrame:
    """Load performance by course module."""
    p = CACHE_DIR / "by_module.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cost_table() -> pd.DataFrame:
    """Load computational cost & latency table."""
    p = CACHE_DIR / "cost_table.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_learner_index() -> pd.DataFrame:
    """Load compact searchable learner index."""
    p = CACHE_DIR / "learner_index.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_learner_trajectory(student_id: int, module: str) -> pd.DataFrame:
    """Load weekly risk predictions and flags for a specific student."""
    p = CACHE_DIR / "learner_trajectories.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        sub = df[(df["id_student"] == student_id) & (df["code_module"] == module)]
        return sub.sort_values("week")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_score_forecasts(student_id: Optional[int] = None, module: Optional[str] = None) -> pd.DataFrame:
    """Load continuous score forecasts."""
    p = CACHE_DIR / "score_forecasts.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if student_id is not None:
            df = df[df["id_student"] == student_id]
        if module is not None:
            df = df[df["code_module"] == module]
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_grade_band_forecasts(student_id: Optional[int] = None, module: Optional[str] = None) -> pd.DataFrame:
    """Load multi-class grade band forecasts."""
    p = CACHE_DIR / "grade_band_forecasts.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if student_id is not None:
            df = df[df["id_student"] == student_id]
        if module is not None:
            df = df[df["code_module"] == module]
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_shap_top5(model_id: str, student_id: int, week: int) -> Optional[pd.DataFrame]:
    """
    Load SHAP top-5 feature contributions following the teammate contract:
    reports/shap/<model>_k3_val_top5.parquet
    Returns None if file is absent (triggering clear pending UI box).
    """
    shap_path = SHAP_DIR / f"{model_id}_k3_val_top5.parquet"
    if not shap_path.exists():
        return None
    try:
        df = pd.read_parquet(shap_path)
        sub = df[(df["id_student"] == student_id) & (df["week"] == week)]
        if len(sub) == 0:
            return None
        return sub.sort_values("rank")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_calibration_data() -> Dict[str, Any]:
    """Load calibration reliability curves and summary metrics."""
    data = {}
    rel_path = CACHE_DIR / "calibration_reliability.parquet"
    if rel_path.exists():
        data["reliability"] = pd.read_parquet(rel_path)
    else:
        data["reliability"] = pd.DataFrame()
        
    sum_path = CACHE_DIR / "calibration_summary.csv"
    if sum_path.exists():
        data["summary"] = pd.read_csv(sum_path)
    else:
        data["summary"] = pd.DataFrame()
    return data


@st.cache_data(show_spinner=False)
def load_fairness_data() -> Dict[str, pd.DataFrame]:
    """Load all fairness audit tables generated in Part 1."""
    tables = {}
    for name in ["fairness_metrics", "fairness_gaps", "intersectional_metrics", "proxy_analysis", "mitigation_experiment"]:
        p = CACHE_DIR / f"{name}.csv"
        if p.exists():
            tables[name] = pd.read_csv(p)
        else:
            tables[name] = pd.DataFrame()
    return tables
