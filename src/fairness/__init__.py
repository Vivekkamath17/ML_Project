"""
Algorithmic Fairness Audit and Disparity Evaluation Package for OULAD.
"""
from src.fairness.audit import run_fairness_audit, load_audit_data
from src.fairness.proxy import run_proxy_analysis
from src.fairness.intersectional import run_intersectional_audit
from src.fairness.mitigation import run_mitigation_experiment

__all__ = [
    "run_fairness_audit",
    "load_audit_data",
    "run_proxy_analysis",
    "run_intersectional_audit",
    "run_mitigation_experiment",
]
