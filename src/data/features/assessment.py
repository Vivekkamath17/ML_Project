"""
Assessment Features module for OULAD data preparation pipeline.
Calculates point-in-time submission, lateness, and performance features.
Strict leakage control: uses ONLY submissions with date_submitted <= 7*(t+1) - 1.
Saves to data/interim/features_assessment.parquet.
"""
from pathlib import Path
from typing import Dict, Any
import polars as pl
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_assessment_features(
    interim_dir: str = "data/interim",
    config_path: str = "config/config.yaml",
) -> pl.DataFrame:
    """
    Computes assessment features at week t:
    - n_submitted: count of submissions with date_submitted <= end of week t
    - n_missed: count of assessments with deadline <= end of week t and no submission by end of week t
    - mean_score_submitted: mean score over submitted assessments (NaN until first submission)
    - last_score: score on the most recently submitted assessment
    - score_slope: change from first to last submitted score
    - mean_lateness: mean of (date_submitted - deadline) for unbanked submissions
    - weighted_score_to_date: sum(score * weight) / sum(weight of assessments due by week t)
    - has_any_assessment: boolean indicator
    - n_banked, has_banked, banked_score_mean: banked assessment features
    """
    interim_path = Path(interim_dir)
    config = load_config(config_path)
    learner_key = config["keys"]["learner"]
    grid_key = config["keys"]["grid"]

    grid = pl.read_parquet(interim_path / "grid.parquet")
    courses = pl.read_parquet(interim_path / "courses.parquet")
    assessments = pl.read_parquet(interim_path / "assessments.parquet")
    student_assessments = pl.read_parquet(interim_path / "studentAssessment.parquet")

    # 1. Fill missing assessment dates with module_presentation_length
    ass_with_date = assessments.join(
        courses, on=["code_module", "code_presentation"], how="left"
    ).with_columns([
        pl.col("date").fill_null(pl.col("module_presentation_length")).alias("deadline_day"),
        # deadline week = floor((deadline_day - 1) / 7)
        ((pl.col("date").fill_null(pl.col("module_presentation_length")) - 1) // 7).cast(pl.Int32).alias("deadline_week"),
    ])

    # 2. Join studentAssessment with assessments to get code_module, code_presentation, deadline_day, deadline_week, weight
    sa_full = student_assessments.join(
        ass_with_date.select([
            "id_assessment",
            "code_module",
            "code_presentation",
            "assessment_type",
            "deadline_day",
            "deadline_week",
            "weight",
        ]),
        on="id_assessment",
        how="inner",
    ).with_columns([
        # submission week = date_submitted // 7 (date < 0 is week -1)
        pl.when(pl.col("date_submitted") < 0)
        .then(-1)
        .otherwise(pl.col("date_submitted") // 7)
        .cast(pl.Int32)
        .alias("submission_week"),
        (pl.col("date_submitted") - pl.col("deadline_day")).alias("lateness"),
    ])

    # 3. Banked assessments summary per learner (available from week 0)
    banked_df = sa_full.filter(pl.col("is_banked") == 1).group_by(learner_key).agg([
        pl.len().alias("n_banked"),
        pl.col("score").mean().round(4).alias("banked_score_mean"),
    ])

    # 4. Total scheduled assessment weight due up to week t per course presentation
    # For each presentation and week t, compute total weight of assessments with deadline_week <= t
    pres_weeks = (
        grid.select(["code_module", "code_presentation", "week"])
        .unique()
        .join(
            ass_with_date.select(["code_module", "code_presentation", "deadline_week", "weight"]),
            on=["code_module", "code_presentation"],
            how="left",
        )
    )
    due_weights = (
        pres_weeks.filter(pl.col("deadline_week") <= pl.col("week"))
        .group_by(["code_module", "code_presentation", "week"])
        .agg(pl.col("weight").sum().alias("total_due_weight"))
    )

    # Count of assessments due by week t
    due_counts = (
        pres_weeks.filter(pl.col("deadline_week") <= pl.col("week"))
        .group_by(["code_module", "code_presentation", "week"])
        .agg(pl.len().alias("n_due_assessments"))
    )

    # 5. Point-in-time assessment features for each learner x week
    # To compute strictly point-in-time: for a learner at week t, filter submissions with submission_week <= t
    # Group submissions by learner + submission_week
    # Submissions occurring at submission_week <= t are known at week t.
    # Cumulative join:
    # A submission at submission_week w is valid for all grid weeks t >= w.
    grid_lean = grid.select(grid_key + ["module_presentation_length"])

    # Extract learner submissions
    learner_subs = sa_full.select(
        learner_key + [
            "id_assessment",
            "submission_week",
            "date_submitted",
            "deadline_day",
            "is_banked",
            "score",
            "weight",
            "lateness",
        ]
    )

    # We can join grid with learner_subs where submission_week <= week
    # To avoid memory blowup, group submissions by learner + submission_week first:
    # Actually, students submit at most ~10-15 assessments, so join on learner_key and filter submission_week <= week:
    # Since studentAssessment has 173k rows across 29k learners (~6 rows per student),
    # joining on learner_key produces ~6 * ~30 = 180 rows per student, total ~5M rows in memory, which Polars handles in < 1 second!
    
    joined_subs = (
        grid_lean.lazy()
        .join(learner_subs.lazy(), on=learner_key, how="inner")
        .filter(pl.col("submission_week") <= pl.col("week"))
        .group_by(grid_key)
        .agg([
            pl.len().alias("n_submitted"),
            pl.col("score").mean().round(4).alias("mean_score_submitted"),
            # last score: score with max date_submitted
            pl.col("score").sort_by("date_submitted").last().alias("last_score"),
            # score slope: last - first
            (pl.col("score").sort_by("date_submitted").last() - pl.col("score").sort_by("date_submitted").first()).alias("score_slope"),
            # mean lateness on unbanked submissions
            pl.col("lateness").filter(pl.col("is_banked") == 0).mean().round(2).alias("mean_lateness"),
            # sum(score * weight)
            (pl.col("score").fill_null(0.0) * pl.col("weight")).sum().alias("submitted_weighted_score_sum"),
        ])
        .collect()
    )

    # 6. Join back to grid
    ass_features = grid.select(grid_key).join(
        joined_subs, on=grid_key, how="left"
    ).join(
        banked_df, on=learner_key, how="left"
    ).join(
        due_weights, on=["code_module", "code_presentation", "week"], how="left"
    ).join(
        due_counts, on=["code_module", "code_presentation", "week"], how="left"
    ).with_columns([
        pl.col("n_submitted").fill_null(0).cast(pl.Int32),
        pl.col("n_due_assessments").fill_null(0).cast(pl.Int32),
        pl.col("total_due_weight").fill_null(0.0).cast(pl.Float32),
        pl.col("submitted_weighted_score_sum").fill_null(0.0).cast(pl.Float32),
        pl.col("n_banked").fill_null(0).cast(pl.Int32),
        (pl.col("n_banked").fill_null(0) > 0).alias("has_banked"),
        (pl.col("n_submitted").fill_null(0) > 0).alias("has_any_assessment"),
    ])

    # Compute n_missed = max(0, n_due_assessments - n_submitted)
    # weighted_score_to_date = sum(score * weight) / total_due_weight (0 if total_due_weight == 0)
    ass_features = ass_features.with_columns([
        pl.max_horizontal(0, pl.col("n_due_assessments") - pl.col("n_submitted")).cast(pl.Int32).alias("n_missed"),
        pl.when(pl.col("total_due_weight") > 0)
        .then((pl.col("submitted_weighted_score_sum") / pl.col("total_due_weight")).round(4))
        .otherwise(0.0)
        .alias("weighted_score_to_date"),
    ])

    out_cols = grid_key + [
        "n_submitted",
        "n_missed",
        "mean_score_submitted",
        "last_score",
        "score_slope",
        "mean_lateness",
        "weighted_score_to_date",
        "has_any_assessment",
        "n_banked",
        "has_banked",
        "banked_score_mean",
    ]

    final_ass = ass_features.select(out_cols)
    out_path = interim_path / "features_assessment.parquet"
    final_ass.write_parquet(out_path, compression="zstd")
    print(f"Assessment features built: {len(final_ass):,} rows -> {out_path}")
    return final_ass


if __name__ == "__main__":
    build_assessment_features()

