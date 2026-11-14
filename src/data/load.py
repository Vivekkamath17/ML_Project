"""
Data loading and conversion module for OULAD pipeline.
Converts raw CSV files to Parquet in data/interim/ with explicit schemas.
"""
from pathlib import Path
from typing import Dict, Optional
import polars as pl
import yaml


def get_explicit_schemas() -> Dict[str, Dict[str, pl.DataType]]:
    """Return explicit Polars schemas for all 7 OULAD tables."""
    return {
        "assessments": {
            "code_module": pl.String,
            "code_presentation": pl.String,
            "id_assessment": pl.Int64,
            "assessment_type": pl.String,
            "date": pl.Int32,
            "weight": pl.Float64,
        },
        "courses": {
            "code_module": pl.String,
            "code_presentation": pl.String,
            "module_presentation_length": pl.Int32,
        },
        "studentAssessment": {
            "id_assessment": pl.Int64,
            "id_student": pl.Int64,
            "date_submitted": pl.Int32,
            "is_banked": pl.Int8,
            "score": pl.Float32,  # float because of nulls
        },
        "studentInfo": {
            "code_module": pl.String,
            "code_presentation": pl.String,
            "id_student": pl.Int64,
            "gender": pl.String,
            "region": pl.String,
            "highest_education": pl.String,
            "imd_band": pl.String,
            "age_band": pl.String,
            "num_of_prev_attempts": pl.Int32,
            "studied_credits": pl.Int32,
            "disability": pl.String,
            "final_result": pl.String,
        },
        "studentRegistration": {
            "code_module": pl.String,
            "code_presentation": pl.String,
            "id_student": pl.Int64,
            "date_registration": pl.Int32,
            "date_unregistration": pl.Int32,
        },
        "studentVle": {
            "code_module": pl.String,
            "code_presentation": pl.String,
            "id_student": pl.Int64,
            "id_site": pl.Int64,
            "date": pl.Int32,
            "sum_click": pl.Int32,
        },
        "vle": {
            "id_site": pl.Int64,
            "code_module": pl.String,
            "code_presentation": pl.String,
            "activity_type": pl.String,
            "week_from": pl.Int32,
            "week_to": pl.Int32,
        },
    }


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load yaml configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def convert_csv_to_parquet(
    raw_dir: str = "data/raw",
    interim_dir: str = "data/interim",
) -> Dict[str, int]:
    """
    Convert all 7 raw CSV tables to Parquet format with explicit schemas.
    Uses lazy scanning and streaming where appropriate for heavy tables like studentVle.
    Returns dictionary of table names and their row counts.
    """
    raw_path = Path(raw_dir)
    interim_path = Path(interim_dir)
    interim_path.mkdir(parents=True, exist_ok=True)

    schemas = get_explicit_schemas()
    row_counts = {}

    null_values = ["", "?", "NA", "null", "NULL", "none", "None"]

    for table_name, schema in schemas.items():
        csv_file = raw_path / f"{table_name}.csv"
        parquet_file = interim_path / f"{table_name}.parquet"

        if not csv_file.exists():
            raise FileNotFoundError(f"Missing raw CSV: {csv_file}")

        # Scan csv lazily with explicit schema and null values
        lf = pl.scan_csv(
            csv_file,
            schema=schema,
            null_values=null_values,
        )

        # Sink to parquet
        lf.sink_parquet(parquet_file, compression="zstd")

        # Read row count from parquet metadata
        count = pl.scan_parquet(parquet_file).select(pl.len()).collect().item()
        row_counts[table_name] = count
        print(f"Converted {table_name}: {count:,} rows -> {parquet_file}")

    return row_counts


if __name__ == "__main__":
    convert_csv_to_parquet()

