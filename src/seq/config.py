"""
Configuration and constants for Phase 4 Sequence Modeling.
Follows Vivek's 5-seed protocol and baseline conventions:
- 5 seeds: [42, 101, 2024, 7, 999]
- Window lengths: [4, 8]
- Targets: 'dropout' (label_dropout_k3), 'onset' (label_inactive_k3), 'final_score'
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any


# 5 fixed seeds from Phase 3 protocol (reports/phase3/phase3_summary.md:13)
SEEDS: List[int] = [42, 101, 2024, 7, 999]

# Sequence window lengths (weeks)
WINDOWS: List[int] = [4, 8]

# Target definitions and column mappings
TARGET_MAP: Dict[str, str] = {
    "dropout": "label_dropout_k3",
    "onset": "label_inactive_k3",
    "final_score": "final_score",
}

# Directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_weekly.parquet"
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "seq_cache"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions" / "seq"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase4"
TUNING_DIR = REPORTS_DIR / "tuning"
MODELS_DIR = PROJECT_ROOT / "models" / "seq"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TUNING_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Static feature list (4 features)
STATIC_FEATURES: List[str] = [
    "num_of_prev_attempts",
    "studied_credits",
    "registration_lead_time",
    "precourse_clicks",
]

# Tuning budget
OPTUNA_TRIALS = 30
CV_FOLDS = 5
MAX_EPOCHS = 60
PATIENCE = 8
GRADIENT_CLIP = 1.0

# Batch sizes
BATCH_SIZES = [128, 256, 512]
