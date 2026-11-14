"""
Curriculum Features module for OULAD data preparation pipeline.
Defines dynamic, leakage-free curriculum availability:
A resource is available at week t if and only if it was clicked by at least one student
in the same presentation cohort at or before week t.
Features computed:
- pct_resources_visited
- n_skipped_resources
- pacing_vs_cohort_median (ratio of learner's cumulative clicks to cohort median cumulative clicks)
- prerequisite_gap (share of resources accessed in the window before upcoming assessment)
Saves to data/interim/features_curriculum.parquet.
"""
from pathlib import Path
from typing import Dict, Any
import polars as pl
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_curriculum_features(
    interim_dir: str = "data/interim",
    config_path: str = "config/config.yaml",
) -> pl.DataFrame:
    """
    Computes leakage-safe curriculum and cohort pacing features:
    1. Dynamic resource availability:
       A resource (id_site) in presentation P is available at week t if
       min(date // 7 across cohort P) <= t.
    2. Learner cumulative distinct resources visited up to week t.
    3. pct_resources_visited = n_visited / n_available_in_cohort.
    4. n_skipped_resources = max(0, n_available_in_cohort - n_visited).
    5. Pacing vs cohort median:
       learner_cum_clicks / (cohort_median_cum_clicks_at_week_t + 1e-4).
    """
    interim_path = Path(interim_dir)
    config = load_config(config_path)
    learner_key = config["keys"]["learner"]
    grid_key = config["keys"]["grid"]

    grid = pl.read_parquet(interim_path / "grid.parquet")
    student_vle = pl.read_parquet(interim_path / "studentVle.parquet")

    # 1. Map studentVle date to week: date < 0 is -1, date >= 0 is date // 7
    svle = student_vle.with_columns(
        pl.when(pl.col("date") < 0)
        .then(-1)
        .otherwise(pl.col("date") // 7)
        .cast(pl.Int32)
        .alias("week")
    )

    # 2. Dynamic resource first-availability week per presentation cohort
    # For each (code_module, code_presentation, id_site), find the minimum week it was accessed
    site_first_available = (
        svle.group_by(["code_module", "code_presentation", "id_site"])
        .agg(pl.col("week").min().alias("site_first_week"))
    )

    # Count cumulative available resources up to each week t for each cohort
    cohort_weeks = grid.select(["code_module", "code_presentation", "week"]).unique()
    
    cohort_avail = (
        cohort_weeks.lazy()
        .join(site_first_available.lazy(), on=["code_module", "code_presentation"], how="inner")
        .filter(pl.col("site_first_week") <= pl.col("week"))
        .group_by(["code_module", "code_presentation", "week"])
        .agg(pl.col("id_site").n_unique().alias("n_cohort_available_resources"))
        .collect()
    )

    # 3. Learner first-visit week per resource:
    # (code_module, code_presentation, id_student, id_site) -> min(week)
    learner_site_first = (
        svle.group_by(learner_key + ["id_site"])
        .agg(pl.col("week").min().alias("learner_first_week"))
    )

    # Count cumulative distinct resources visited by learner up to week t
    grid_lean = grid.select(grid_key)
    learner_avail = (
        grid_lean.lazy()
        .join(learner_site_first.lazy(), on=learner_key, how="inner")
        .filter(pl.col("learner_first_week") <= pl.col("week"))
        .group_by(grid_key)
        .agg(pl.col("id_site").n_unique().alias("n_resources_visited"))
        .collect()
    )

    # 4. Cumulative clicks per learner up to week t
    # Group weekly clicks from grid: sort by learner + week, cum_sum(weekly_clicks)
    grid_sorted = grid.sort(learner_key + ["week"]).with_columns(
        pl.col("weekly_clicks").cum_sum().over(learner_key).alias("cum_clicks")
    )

    # Compute cohort median cumulative clicks at each week t
    cohort_median_clicks = (
        grid_sorted.group_by(["code_module", "code_presentation", "week"])
        .agg(pl.col("cum_clicks").median().alias("cohort_median_cum_clicks"))
    )

    # 5. Combine and compute pacing & resource gaps
    curriculum_df = (
        grid_sorted.select(grid_key + ["cum_clicks"])
        .join(cohort_avail, on=["code_module", "code_presentation", "week"], how="left")
        .join(learner_avail, on=grid_key, how="left")
        .join(cohort_median_clicks, on=["code_module", "code_presentation", "week"], how="left")
        .with_columns([
            pl.col("n_cohort_available_resources").fill_null(0).cast(pl.Int32),
            pl.col("n_resources_visited").fill_null(0).cast(pl.Int32),
            pl.col("cohort_median_cum_clicks").fill_null(0.0).cast(pl.Float32),
        ])
        .with_columns([
            pl.when(pl.col("n_cohort_available_resources") > 0)
            .then((pl.col("n_resources_visited") / pl.col("n_cohort_available_resources")).round(4))
            .otherwise(0.0)
            .alias("pct_resources_visited"),
            pl.max_horizontal(0, pl.col("n_cohort_available_resources") - pl.col("n_resources_visited")).alias("n_skipped_resources"),
            (
                pl.col("cum_clicks") / (pl.col("cohort_median_cum_clicks") + 0.0001)
            ).round(4).alias("pacing_vs_cohort_median"),
        ])
    )

    # Prerequisite gap approximation: 1.0 - pct_resources_visited
    curriculum_df = curriculum_df.with_columns(
        (1.0 - pl.col("pct_resources_visited")).round(4).alias("prerequisite_gap")
    )

    out_cols = grid_key + [
        "pct_resources_visited",
        "n_skipped_resources",
        "pacing_vs_cohort_median",
        "prerequisite_gap",
    ]

    final_curriculum = curriculum_df.select(out_cols)
    out_path = interim_path / "features_curriculum.parquet"
    final_curriculum.write_parquet(out_path, compression="zstd")
    print(f"Curriculum features built: {len(final_curriculum):,} rows -> {out_path}")
    return final_curriculum


if __name__ == "__main__":
    build_curriculum_features()

