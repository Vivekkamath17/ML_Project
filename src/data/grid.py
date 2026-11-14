"""
Week Grid module for OULAD data preparation pipeline.
Constructs the dense learner x week grid from week 0 to the course presentation length.
Saves to data/interim/grid.parquet.
"""
from pathlib import Path
from typing import Dict, Any
import polars as pl
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_week_grid(
    interim_dir: str = "data/interim",
    processed_dir: str = "data/processed",
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Constructs the learner x week grid:
    1. For every learner in learners.parquet, create weeks from 0 to ceil(module_presentation_length / 7) - 1.
    2. Left join weekly VLE activity (clicks, active_days). Zero-fill missing weeks.
    3. Add precourse_clicks (from week -1) as a static feature carried into weeks 0+.
    4. Compute is_enrolled_at_week_t = (effective_date_unregistration is null) | (effective_date_unregistration > 7 * (t + 1)).
    5. Drop rows strictly after the learner unregistered: keep only rows where learner was enrolled during week t.
    6. Add is_post_assessment_period: True if week t > last scheduled assessment deadline week in that presentation.
    7. Retain flags: never_active_learner, is_zero_click_withdrawal, label_is_proxy.
    8. Add course_progress = (t + 1) * 7 / module_presentation_length.
    9. Save to data/interim/grid.parquet.
    """
    interim_path = Path(interim_dir)
    processed_path = Path(processed_dir)
    config = load_config(config_path)

    learner_key = config["keys"]["learner"]
    learners = pl.read_parquet(processed_path / "learners.parquet")
    courses = pl.read_parquet(interim_path / "courses.parquet")
    assessments = pl.read_parquet(interim_path / "assessments.parquet")
    vle_lf = pl.scan_parquet(interim_path / "studentVle.parquet")

    # 1. Compute course length in weeks for each course presentation
    courses_weeks = courses.with_columns(
        ((pl.col("module_presentation_length") - 1) // 7).cast(pl.Int32).alias("max_week")
    )

    # Compute last assessment week per presentation (fill missing assessment dates with module_presentation_length)
    ass_dates = assessments.join(courses, on=["code_module", "code_presentation"], how="left").with_columns(
        pl.col("date").fill_null(pl.col("module_presentation_length")).alias("effective_ass_date")
    )
    last_ass_weeks = ass_dates.group_by(["code_module", "code_presentation"]).agg(
        ((pl.col("effective_ass_date").max() - 1) // 7).cast(pl.Int32).alias("last_assessment_week")
    )

    # 2. Extract weekly clicks and precourse clicks from studentVle
    vle_weekly = (
        vle_lf.with_columns(
            pl.when(pl.col("date") < 0)
            .then(-1)
            .otherwise(pl.col("date") // 7)
            .cast(pl.Int32)
            .alias("week")
        )
        .group_by(learner_key + ["week"])
        .agg([
            pl.col("sum_click").sum().alias("weekly_clicks"),
            pl.col("date").n_unique().alias("weekly_active_days"),
        ])
        .collect()
    )

    # Separate precourse clicks (week == -1)
    precourse_clicks_df = (
        vle_weekly.filter(pl.col("week") == -1)
        .select(learner_key + [pl.col("weekly_clicks").alias("precourse_clicks")])
    )

    # Weekly clicks for weeks >= 0
    vle_weekly_active = vle_weekly.filter(pl.col("week") >= 0)

    # 3. Build dense grid for each learner
    learners_with_max_week = learners.select(
        learner_key + [
            "module_presentation_length",
            "effective_date_unregistration",
            "label_is_proxy",
            "never_active_learner",
            "is_zero_click_withdrawal",
            "final_result",
            "registration_lead_time",
            "has_missing_date_registration",
            "num_of_prev_attempts",
            "studied_credits",
        ]
    ).join(
        courses_weeks.select(["code_module", "code_presentation", "max_week"]),
        on=["code_module", "code_presentation"],
        how="left",
    ).join(
        last_ass_weeks,
        on=["code_module", "code_presentation"],
        how="left",
    )

    # Create dense weeks for each learner: explode list of weeks [0 .. max_week]
    dense_grid = learners_with_max_week.with_columns(
        pl.int_ranges(0, pl.col("max_week") + 1).alias("week")
    ).explode("week").cast({"week": pl.Int32})

    # 4. Left join precourse_clicks
    dense_grid = dense_grid.join(precourse_clicks_df, on=learner_key, how="left").with_columns(
        pl.col("precourse_clicks").fill_null(0).cast(pl.Int32)
    )

    # 5. Left join weekly VLE clicks
    dense_grid = dense_grid.join(vle_weekly_active, on=learner_key + ["week"], how="left").with_columns([
        pl.col("weekly_clicks").fill_null(0).cast(pl.Int32),
        pl.col("weekly_active_days").fill_null(0).cast(pl.Int32),
    ])

    # 6. Compute is_enrolled_at_week_t and is_post_assessment_period
    dense_grid = dense_grid.with_columns([
        (
            pl.col("effective_date_unregistration").is_null()
            | (pl.col("effective_date_unregistration") > 7 * (pl.col("week") + 1))
        ).alias("is_enrolled_at_week_t"),
        (pl.col("week") > pl.col("last_assessment_week")).alias("is_post_assessment_period"),
    ])

    # 7. Drop rows after the learner unregistered
    grid_eligible = dense_grid.filter(
        pl.col("effective_date_unregistration").is_null()
        | (pl.col("effective_date_unregistration") >= 7 * pl.col("week"))
    )

    # 8. Add course_progress = (t + 1) * 7 / module_presentation_length
    grid_eligible = grid_eligible.with_columns(
        ((pl.col("week") + 1) * 7.0 / pl.col("module_presentation_length"))
        .round(4)
        .alias("course_progress")
    )

    output_path = interim_path / "grid.parquet"
    grid_eligible.write_parquet(output_path, compression="zstd")

    report = {
        "dense_rows_before_filter": len(dense_grid),
        "grid_rows_after_unreg_filter": len(grid_eligible),
        "unique_learners": len(grid_eligible.select(learner_key).unique()),
        "output_path": str(output_path),
    }

    print(f"Grid built: {len(grid_eligible):,} rows across {report['unique_learners']:,} learners -> {output_path}")

    return report


if __name__ == "__main__":
    build_week_grid()
