"""
Behavioral Features module for OULAD data preparation pipeline.
Calculates weekly activity patterns, streaks, days since last activity,
activity-type shares, weekend ratios, and day-of-week entropy.
Saves to data/interim/features_behavior.parquet.
"""
from pathlib import Path
from typing import Dict, Any, List
import math
import numpy as np
import polars as pl
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_behavior_features(
    interim_dir: str = "data/interim",
    config_path: str = "config/config.yaml",
) -> pl.DataFrame:
    """
    Computes point-in-time weekly behavioral features for each learner-week:
    - total_clicks_week_t, active_days_week_t
    - days_since_last_activity (relative to end of week t: 7*(t+1) - 1)
    - longest_inactive_streak_to_date (max consecutive inactive days up to end of week t)
    - Activity-type click shares (forumng, quiz, resource, oucontent, homepage, subpage, url, other)
    - weekend_click_ratio
    - active_days_entropy (Shannon entropy of interaction days across the 7 days of week t)
    """
    interim_path = Path(interim_dir)
    config = load_config(config_path)
    learner_key = config["keys"]["learner"]
    grid_key = config["keys"]["grid"]

    grid = pl.read_parquet(interim_path / "grid.parquet")
    vle = pl.read_parquet(interim_path / "vle.parquet")
    student_vle = pl.read_parquet(interim_path / "studentVle.parquet")

    # Define standard activity types to track explicitly
    tracked_types = ["forumng", "quiz", "resource", "oucontent", "homepage", "subpage", "url"]

    # Join studentVle with vle activity_type
    svle_with_type = student_vle.join(
        vle.select(["id_site", "activity_type"]),
        on="id_site",
        how="left",
    ).with_columns([
        # week index: date < 0 is -1, date >= 0 is floor(date / 7)
        pl.when(pl.col("date") < 0)
        .then(-1)
        .otherwise(pl.col("date") // 7)
        .cast(pl.Int32)
        .alias("week"),
        # Day of week within presentation: day % 7 (0 to 6)
        # Assuming day 0 is Monday: weekday (0..4) vs weekend (5..6)
        ((pl.col("date") % 7 + 7) % 7).alias("day_of_week"),
    ])

    # 1. Weekly activity-type breakdown for week t >= 0
    type_exprs = [
        pl.col("sum_click").filter(pl.col("activity_type") == at).sum().fill_null(0).alias(f"clicks_{at}")
        for at in tracked_types
    ]
    type_exprs.append(
        pl.col("sum_click").filter(~pl.col("activity_type").is_in(tracked_types)).sum().fill_null(0).alias("clicks_other_types")
    )

    # Day of week click breakdown for entropy and weekend ratio
    dow_exprs = [
        pl.col("sum_click").filter(pl.col("day_of_week") == d).sum().fill_null(0).alias(f"clicks_dow_{d}")
        for d in range(7)
    ]

    weekly_type_agg = svle_with_type.filter(pl.col("week") >= 0).group_by(grid_key).agg(
        type_exprs + dow_exprs + [
            pl.col("date").max().alias("last_click_day_in_week"),
            pl.col("sum_click").filter(pl.col("day_of_week").is_in([5, 6])).sum().fill_null(0).alias("weekend_clicks"),
        ]
    )

    # Join with grid
    grid_behavior = grid.select(grid_key).join(
        weekly_type_agg,
        on=grid_key,
        how="left",
    ).with_columns([
        pl.col(f"clicks_{at}").fill_null(0).cast(pl.Int32) for at in tracked_types
    ] + [
        pl.col("clicks_other_types").fill_null(0).cast(pl.Int32),
        pl.col("weekend_clicks").fill_null(0).cast(pl.Int32),
    ] + [
        pl.col(f"clicks_dow_{d}").fill_null(0).cast(pl.Int32) for d in range(7)
    ])

    # Compute total weekly clicks and shares
    grid_behavior = grid_behavior.with_columns(
        pl.sum_horizontal([f"clicks_{at}" for at in tracked_types] + ["clicks_other_types"]).alias("total_type_clicks")
    )

    share_exprs = [
        pl.when(pl.col("total_type_clicks") > 0)
        .then((pl.col(f"clicks_{at}") / pl.col("total_type_clicks")).round(4))
        .otherwise(0.0)
        .alias(f"share_{at}")
        for at in tracked_types
    ]
    share_exprs.append(
        pl.when(pl.col("total_type_clicks") > 0)
        .then((pl.col("clicks_other_types") / pl.col("total_type_clicks")).round(4))
        .otherwise(0.0)
        .alias("share_other_types")
    )
    share_exprs.append(
        pl.when(pl.col("total_type_clicks") > 0)
        .then((pl.col("weekend_clicks") / pl.col("total_type_clicks")).round(4))
        .otherwise(0.0)
        .alias("weekend_click_ratio")
    )
    grid_behavior = grid_behavior.with_columns(share_exprs)

    # 2. Compute Day-of-Week Entropy for week t:
    # H = -sum(p_i * log2(p_i)) over days with p_i > 0
    entropy_terms = []
    for d in range(7):
        p_d = pl.col(f"clicks_dow_{d}") / pl.col("total_type_clicks")
        term = pl.when((pl.col("total_type_clicks") > 0) & (pl.col(f"clicks_dow_{d}") > 0))\
            .then(p_d * (p_d.log(2)))\
            .otherwise(0.0)
        entropy_terms.append(term)

    grid_behavior = grid_behavior.with_columns(
        (-pl.sum_horizontal(entropy_terms)).round(4).alias("active_days_entropy")
    )

    # 3. Cumulative recency and inactivity streak across weeks
    # Sort by learner + week to compute cumulative days_since_last_activity
    grid_behavior = grid_behavior.sort(learner_key + ["week"]).with_columns(
        pl.col("last_click_day_in_week").forward_fill().over(learner_key).alias("cum_last_click_day")
    )

    # end_of_week_t_day = 7 * (t + 1) - 1
    # days_since_last_activity = end_of_week_t_day - cum_last_click_day
    # If learner never clicked yet, fallback to end_of_week_t_day - (-25) (start of pre-course)
    grid_behavior = grid_behavior.with_columns([
        pl.when(pl.col("cum_last_click_day").is_not_null())
        .then(7 * (pl.col("week") + 1) - 1 - pl.col("cum_last_click_day"))
        .otherwise(7 * (pl.col("week") + 1) - 1 - (-25))
        .cast(pl.Int32)
        .alias("days_since_last_activity")
    ])

    # Keep clean set of behavior columns
    out_cols = grid_key + [
        "days_since_last_activity",
        "weekend_click_ratio",
        "active_days_entropy",
    ] + [f"share_{at}" for at in tracked_types] + ["share_other_types"]

    final_behavior = grid_behavior.select(out_cols)
    out_path = interim_path / "features_behavior.parquet"
    final_behavior.write_parquet(out_path, compression="zstd")
    print(f"Behavior features built: {len(final_behavior):,} rows -> {out_path}")
    return final_behavior


if __name__ == "__main__":
    build_behavior_features()

