"""
Fairness Audit Table module for OULAD data preparation pipeline.
Contains ONLY the learner key plus sensitive demographic attributes:
- gender, age_band, region, highest_education, imd_band, disability
Missing imd_band values are imputed as 'Unknown'.
STRICT RULE: This table is NEVER merged into the weekly feature table.
Saves to data/processed/audit_demographics.parquet.
"""
from pathlib import Path
from typing import Dict, Any
import polars as pl
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_audit_demographics(
    interim_dir: str = "data/interim",
    processed_dir: str = "data/processed",
    config_path: str = "config/config.yaml",
) -> pl.DataFrame:
    """
    Constructs the fairness audit table with one row per learner in learners.parquet.
    Contains strictly learner keys and demographic columns.
    """
    interim_path = Path(interim_dir)
    processed_path = Path(processed_dir)
    config = load_config(config_path)

    learner_key = config["keys"]["learner"]
    learners = pl.read_parquet(processed_path / "learners.parquet")
    info = pl.read_parquet(interim_path / "studentInfo.parquet")

    # Select demographics from studentInfo for active learners in learners.parquet
    demographic_cols = [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]

    audit_df = learners.select(learner_key).join(
        info.select(learner_key + demographic_cols),
        on=learner_key,
        how="left",
    ).with_columns(
        pl.col("imd_band").fill_null("Unknown")
    )

    out_path = processed_path / "audit_demographics.parquet"
    audit_df.write_parquet(out_path, compression="zstd")

    print(f"Fairness audit demographics built: {len(audit_df):,} rows -> {out_path}")
    print(f"Unique learners: {len(audit_df.select(learner_key).unique()):,}")
    print(f"Columns: {audit_df.columns}")

    return audit_df


if __name__ == "__main__":
    build_audit_demographics()

