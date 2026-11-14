"""
Phase 3 Model Zoo Package.
Provides standardized model wrappers, preprocessing pipelines, training drivers,
and analysis tools for OULAD dropout benchmarks.
"""

from src.models.zoo.common import get_imputed_pipeline, load_zoo_data
from src.models.zoo.xgb import fit_and_evaluate_xgb, tune_xgb
from src.models.zoo.catboost_model import fit_and_evaluate_catboost, tune_catboost

__all__ = [
    "load_zoo_data",
    "get_imputed_pipeline",
    "tune_xgb",
    "fit_and_evaluate_xgb",
    "tune_catboost",
    "fit_and_evaluate_catboost",
]
