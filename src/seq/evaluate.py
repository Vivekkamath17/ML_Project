"""
Evaluation and MLflow logging module for Phase 4 Sequence Models.
Reuses:
- src.models.harness.evaluate_model (point metrics, lift@top-10%, cluster bootstrap CIs)
- src.models.harness.paired_bootstrap (paired delta ROC and delta PR vs LightGBM v3)
- src.models.harness.log_run (MLflow logging to sqlite:///mlruns.db)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.harness import evaluate_model, log_run, paired_bootstrap
from src.models.predictions import read_predictions
from src.seq.config import REPORTS_DIR


def evaluate_seq_model(
    y_true: np.ndarray,
    val_probs: np.ndarray,
    cluster_ids: np.ndarray,
    target: str = "dropout",
    baseline_path: Optional[str] = "predictions/dropout/k3/lgbm_v3_val.parquet",
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate sequence model predictions using Vivek's official metrics harness.
    Computes:
      - Point metrics (ROC-AUC, PR-AUC, Brier, MCC, lift@top-10%)
      - Cluster bootstrap 95% CIs
      - Paired bootstrap deltas vs LightGBM v3 (if baseline predictions exist)
    """
    threshold = 0.035 if target == "dropout" else 0.5
    metrics = evaluate_model(
        y_true=y_true,
        y_prob=val_probs,
        cluster_ids=cluster_ids,
        threshold=threshold,
        n_resamples=n_resamples,
        seed=seed,
    )

    paired_res = None
    if target == "dropout" and baseline_path and Path(baseline_path).is_file():
        df_base, _ = read_predictions(baseline_path)
        # Check alignment
        if len(df_base) == len(val_probs):
            paired_res = paired_bootstrap(
                y_true=y_true,
                y_prob_a=val_probs,
                y_prob_b=df_base["y_prob"].to_numpy(),
                cluster_ids=cluster_ids,
                n_boot=n_resamples,
                seed=seed,
            )

    return {
        "metrics": metrics,
        "paired_vs_lgbm": paired_res,
    }


def log_to_mlflow(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    tags: Dict[str, Any],
    experiment_name: str = "dropout",
) -> str:
    """Log run to MLflow SQLite database with strict naming schema."""
    clean_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating, np.integer))}

    return log_run(
        run_name=run_name,
        params=params,
        metrics=clean_metrics,
        experiment_name=experiment_name,
    )
