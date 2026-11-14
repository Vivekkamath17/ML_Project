"""
Temporal Features module for OULAD data preparation pipeline.
Calculates week-over-week changes, rolling means (2 & 4 weeks), linear slope,
EWMA, and engagement decay rates.
Uses strictly current and past weeks (no future data).
Saves to data/interim/features_temporal.parquet.
"""
from pathlib import Path
from typing import Dict, Any
import polars as pl
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_temporal_features(
    interim_dir: str = "data/interim",
    config_path: str = "config/config.yaml",
) -> pl.DataFrame:
    """
    Computes temporal trajectory features per learner-week:
    - wow_change_clicks = clicks_t - clicks_{t-1}
    - wow_change_active_days = active_days_t - active_days_{t-1}
    - rolling_mean_clicks_2w (mean over [t-1, t])
    - rolling_mean_clicks_4w (mean over [t-3, ..., t])
    - rolling_mean_active_days_2w, rolling_mean_active_days_4w
    - clicks_slope_4w (linear regression slope of clicks over last 4 weeks)
    - clicks_ewma (exponentially weighted moving average with alpha=0.3)
    - engagement_decay_rate (ratio of rolling_mean_clicks_2w to rolling_mean_clicks_4w)
    """
    interim_path = Path(interim_dir)
    config = load_config(config_path)
    learner_key = config["keys"]["learner"]
    grid_key = config["keys"]["grid"]

    grid = pl.read_parquet(interim_path / "grid.parquet").sort(learner_key + ["week"])

    # 1. Week-over-week changes
    df = grid.select(grid_key + ["weekly_clicks", "weekly_active_days"]).with_columns([
        pl.col("weekly_clicks").cast(pl.Float32),
        pl.col("weekly_active_days").cast(pl.Float32),
        (
            pl.col("weekly_clicks") - pl.col("weekly_clicks").shift(1).over(learner_key).fill_null(0.0)
        ).alias("wow_change_clicks"),
        (
            pl.col("weekly_active_days") - pl.col("weekly_active_days").shift(1).over(learner_key).fill_null(0.0)
        ).alias("wow_change_active_days"),
    ])

    # 2. Rolling means over 2 and 4 weeks (using shifts over learner_key)
    # 2 weeks: t and t-1
    c0 = pl.col("weekly_clicks")
    c1 = pl.col("weekly_clicks").shift(1).over(learner_key)
    c2 = pl.col("weekly_clicks").shift(2).over(learner_key)
    c3 = pl.col("weekly_clicks").shift(3).over(learner_key)

    d0 = pl.col("weekly_active_days")
    d1 = pl.col("weekly_active_days").shift(1).over(learner_key)
    d2 = pl.col("weekly_active_days").shift(2).over(learner_key)
    d3 = pl.col("weekly_active_days").shift(3).over(learner_key)

    df = df.with_columns([
        # 2w mean
        pl.when(c1.is_not_null())
        .then((c0 + c1) / 2.0)
        .otherwise(c0)
        .round(4)
        .alias("rolling_mean_clicks_2w"),
        pl.when(d1.is_not_null())
        .then((d0 + d1) / 2.0)
        .otherwise(d0)
        .round(4)
        .alias("rolling_mean_active_days_2w"),
        # 4w mean
        pl.when(c3.is_not_null())
        .then((c0 + c1 + c2 + c3) / 4.0)
        .when(c2.is_not_null())
        .then((c0 + c1 + c2) / 3.0)
        .when(c1.is_not_null())
        .then((c0 + c1) / 2.0)
        .otherwise(c0)
        .round(4)
        .alias("rolling_mean_clicks_4w"),
        pl.when(d3.is_not_null())
        .then((d0 + d1 + d2 + d3) / 4.0)
        .when(d2.is_not_null())
        .then((d0 + d1 + d2) / 3.0)
        .when(d1.is_not_null())
        .then((d0 + d1) / 2.0)
        .otherwise(d0)
        .round(4)
        .alias("rolling_mean_active_days_4w"),
    ])

    # 3. Linear slope over last 4 weeks:
    # x in [0, 1, 2, 3] with mean 1.5, sum((x - 1.5)^2) = 5.0
    # slope = (-1.5*c3 - 0.5*c2 + 0.5*c1 + 1.5*c0) / 5.0 = (3*(c0 - c3) + (c1 - c2)) / 10.0
    df = df.with_columns(
        pl.when(c3.is_not_null())
        .then((3.0 * (c0 - c3) + (c1 - c2)) / 10.0)
        .when(c2.is_not_null())
        .then((c0 - c2) / 2.0)
        .when(c1.is_not_null())
        .then(c0 - c1)
        .otherwise(0.0)
        .round(4)
        .alias("clicks_slope_4w")
    )

    # 4. Engagement decay rate: rolling_2w / (rolling_4w + 1e-5)
    df = df.with_columns(
        (pl.col("rolling_mean_clicks_2w") / (pl.col("rolling_mean_clicks_4w") + 0.0001))
        .round(4)
        .alias("engagement_decay_rate")
    )

    # 5. EWMA (alpha = 0.3): ewma_t = 0.3 * c_t + 0.7 * ewma_{t-1}
    # Polars ewma can be computed over learner_key
    df = df.with_columns(
        pl.col("weekly_clicks").ewm_mean(alpha=0.3).over(learner_key).round(4).alias("clicks_ewma")
    )

    out_cols = grid_key + [
        "wow_change_clicks",
        "wow_change_active_days",
        "rolling_mean_clicks_2w",
        "rolling_mean_clicks_4w",
        "rolling_mean_active_days_2w",
        "rolling_mean_active_days_4w",
        "clicks_slope_4w",
        "clicks_ewma",
        "engagement_decay_rate",
    ]
    final_temporal = df.select(out_cols)
    out_path = interim_path / "features_temporal.parquet"
    final_temporal.write_parquet(out_path, compression="zstd")
    print(f"Temporal features built: {len(final_temporal):,} rows -> {out_path}")
    return final_temporal


if __name__ == "__main__":
    build_temporal_features()

