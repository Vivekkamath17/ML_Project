"""Entry point for forecasting comparison (final_score & grade_band).

Usage:
    python run_forecast_cmp.py [--extra-predictions DIR]

The script:
  1. Reads configuration from `config/forecast_cmp.yaml`.
  2. Builds a unified learner set for the three snapshot weeks (4, 8, 12) from
     `data/processed/features_weekly.parquet` using the 38 v3 features.
  3. Loads targets (`final_score` and `grade_band`) from `labels/score_audit.csv`.
  4. Trains each model listed in the config on the train split (2013B + 2013J) only,
     using 5‑fold GroupKFold by `id_student` and Optuna (15 trials per model).
  5. Evaluates on the validation split (2014B) for the three snapshot weeks,
     computing metrics and bootstrap CIs as described in the task.
  6. Writes predictions via `src.models.predictions.write_predictions` to the
     configured output directories.
  7. Generates CSV tables and figures under `reports/niranjan/`.
  8. If `--extra-predictions` is supplied, validates the schema of the provided
     prediction files and merges them into the tables without retraining.

Only standard libraries and the project's helper modules are used; no new
runtime dependencies are introduced.
"""

import argparse
import os
from pathlib import Path

from src.models.forecast_cmp import run_forecast_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run forecasting comparison")
    parser.add_argument(
        "--extra-predictions",
        type=Path,
        help="Directory containing additional prediction files to include in the tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Run the core forecast pipeline
    run_forecast_comparison(extra_pred_dir=args.extra_predictions)


if __name__ == "__main__":
    main()
