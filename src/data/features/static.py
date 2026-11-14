"""
Static Features module for OULAD data preparation pipeline.
Extracts academic history and static registration parameters:
- num_of_prev_attempts
- studied_credits
- registration_lead_time
- has_missing_date_registration
- precourse_clicks
- course_progress = (t + 1) * 7 / module_presentation_length
Strict fairness rule: NO demographic features (gender, age_band, region, etc.)
Saves to data/interim/features_static.parquet.
"""
from pathlib import Path
from typing import Dict, Any
import polars as pl
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_static_features(
    interim_dir: str = "data/interim",
    config_path: str = "config/config.yaml",
) -> pl.DataFrame:
    """
    Computes static features per learner-week:
    - num_of_prev_attempts
    - studied_credits
    - registration_lead_time
    - has_missing_date_registration
    - precourse_clicks
    - course_progress
    """
    interim_path = Path(interim_dir)
    config = load_config(config_path)
    grid_key = config["keys"]["grid"]

    grid = pl.read_parquet(interim_path / "grid.parquet")

    out_cols = grid_key + [
        "num_of_prev_attempts",
        "studied_credits",
        "registration_lead_time",
        "has_missing_date_registration",
        "precourse_clicks",
        "course_progress",
    ]

    static_df = grid.select(out_cols)
    out_path = interim_path / "features_static.parquet"
    static_df.write_parquet(out_path, compression="zstd")
    print(f"Static features built: {len(static_df):,} rows -> {out_path}")
    return static_df


if __name__ == "__main__":
    build_static_features()

