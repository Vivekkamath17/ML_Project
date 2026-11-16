"""
Orchestrator for Phase 4 Sequence Model Experiments.
Executes the experimental matrix across architectures, windows, targets, and variants:
- Checks if valid prediction file exists before running (idempotent / resumable).
- Logs results to reports/phase4/by_model_seq.csv.
- Re-generates figures and updates STATUS.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.seq.config import PREDICTIONS_DIR, REPORTS_DIR
from src.seq.run_experiment import run_experiment
from src.seq.validate_predictions import validate_prediction_file
from src.seq.plot_results import plot_metric_vs_model_ci, plot_metric_by_snapshot_week
from src.seq.cohort_analysis import analyze_cohorts


EXPERIMENTS = [
    # 1. GRU core
    {"model": "gru", "window": 8, "target": "dropout", "variant": "seq_only"},
    {"model": "gru", "window": 4, "target": "dropout", "variant": "seq_only"},
    {"model": "gru", "window": 8, "target": "onset", "variant": "seq_only"},
    {"model": "gru", "window": 4, "target": "onset", "variant": "seq_only"},

    # 2. LSTM core
    {"model": "lstm", "window": 8, "target": "dropout", "variant": "seq_only"},
    {"model": "lstm", "window": 4, "target": "dropout", "variant": "seq_only"},
    {"model": "lstm", "window": 8, "target": "onset", "variant": "seq_only"},
    {"model": "lstm", "window": 4, "target": "onset", "variant": "seq_only"},

    # 3. TCN core
    {"model": "tcn", "window": 8, "target": "dropout", "variant": "seq_only"},
    {"model": "tcn", "window": 4, "target": "dropout", "variant": "seq_only"},
    {"model": "tcn", "window": 8, "target": "onset", "variant": "seq_only"},
    {"model": "tcn", "window": 4, "target": "onset", "variant": "seq_only"},

    # 4. Hybrid & Multi-task variants (applied to GRU and TCN)
    {"model": "gru", "window": 8, "target": "dropout", "variant": "hybrid"},
    {"model": "gru", "window": 8, "target": "dropout", "variant": "seq_only_multi"},
    {"model": "gru", "window": 8, "target": "dropout", "variant": "hybrid_multi"},
    {"model": "gru", "window": 8, "target": "onset", "variant": "hybrid"},
    {"model": "gru", "window": 8, "target": "onset", "variant": "seq_only_multi"},
    {"model": "gru", "window": 8, "target": "onset", "variant": "hybrid_multi"},
]


def is_experiment_complete(exp: Dict[str, Any]) -> bool:
    """Check if prediction file exists and passes validation."""
    pred_path = PREDICTIONS_DIR / f"{exp['model']}_w{exp['window']}_{exp['target']}_{exp['variant']}.csv"
    if not pred_path.is_file():
        return False
    try:
        report = validate_prediction_file(pred_path)
        return report["passed"]
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all remaining Phase 4 experiments")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    print(f"=== PHASE 4 EXPERIMENT ORCHESTRATOR ===")
    print(f"Total experiments configured: {len(EXPERIMENTS)}")
    print(f"Trials per tuning: {args.trials} | Seeds: {args.seeds} | Smoke: {args.smoke}\n")

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        tag = f"{exp['model']}_w{exp['window']}_{exp['target']}_{exp['variant']}"
        print(f"\n[{idx}/{len(EXPERIMENTS)}] Checking {tag}...")

        if not args.smoke and is_experiment_complete(exp):
            print(f"-> Already completed and validated. Skipping.")
            continue

        print(f"-> Running experiment: {tag}")
        t0 = time.perf_counter()
        run_experiment(
            model_name=exp["model"],
            window=exp["window"],
            target=exp["target"],
            variant=exp["variant"],
            n_trials=args.trials,
            n_seeds=args.seeds,
            max_epochs=args.max_epochs,
            smoke=args.smoke,
        )
        dt = time.perf_counter() - t0
        print(f"-> Finished {tag} in {dt/60.0:.1f} minutes.")

    # After running, regenerate figures
    summary_csv = REPORTS_DIR / "by_model_seq.csv"
    if summary_csv.is_file():
        print("\nUpdating publication figures...")
        for target in ["dropout", "onset"]:
            try:
                plot_metric_vs_model_ci(summary_csv, target=target)
            except Exception as e:
                print(f"Could not generate plot for {target}: {e}")

    print("\nAll experiments in matrix finished successfully!")


if __name__ == "__main__":
    main()
