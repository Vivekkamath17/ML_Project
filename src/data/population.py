"""
Learner Population module for OULAD data preparation pipeline.
Constructs the filtered learner population and saves to data/processed/learners.parquet.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import polars as pl
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_learner_population(
    interim_dir: str = "data/interim",
    processed_dir: str = "data/processed",
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Constructs the filtered learner population.
    Rules applied:
    1. Join studentRegistration, studentInfo, and courses on learner key / course key.
    2. Drop pre-course withdrawals: date_unregistration < 0 (2,678 learners).
    3. Identify zero-click non-withdrawn learners (377 learners) and flag as never_active_learner = True.
    4. Identify zero-click withdrawn learners with real dates (756 learners) and flag as is_zero_click_withdrawal = True.
    5. Exclude withdrawn learners with no date_unregistration and zero VLE clicks ever (19 learners).
    6. Compute proxy unregistration date for Withdrawn learners lacking date_unregistration (74 learners):
       proxy_date = min(last_vle_date + median_lag, module_presentation_length).
       Sets label_is_proxy = True.
    7. Retain learners who unregistered after course end with date_unregistration.
    8. Impute date_registration missing values with median, add has_missing_date_registration flag.
    9. Save to data/processed/learners.parquet.
    """
    interim_path = Path(interim_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    learner_key = config["keys"]["learner"]

    info = pl.read_parquet(interim_path / "studentInfo.parquet")
    reg = pl.read_parquet(interim_path / "studentRegistration.parquet")
    courses = pl.read_parquet(interim_path / "courses.parquet")
    vle_lf = pl.scan_parquet(interim_path / "studentVle.parquet")

    # 1. Join info + reg + courses
    df = (
        reg.join(info, on=learner_key, how="inner")
        .join(courses, on=["code_module", "code_presentation"], how="left")
    )
    initial_count = len(df)

    # 2. Drop pre-course withdrawals (date_unregistration < 0)
    df_active = df.filter(
        (pl.col("date_unregistration").is_null()) | (pl.col("date_unregistration") >= 0)
    )
    precourse_unreg_dropped = initial_count - len(df_active)

    # 3. Compute last_vle_date and total VLE clicks per learner
    vle_stats = (
        vle_lf.group_by(learner_key)
        .agg([
            pl.col("date").max().alias("last_vle_date"),
            pl.col("sum_click").sum().alias("total_vle_clicks"),
        ])
        .collect()
    )
    df_active = df_active.join(vle_stats, on=learner_key, how="left")

    # 4. Flags: never_active_learner (377) and is_zero_click_withdrawal (756)
    df_active = df_active.with_columns([
        (
            (pl.col("final_result") != "Withdrawn")
            & (pl.col("total_vle_clicks").is_null() | (pl.col("total_vle_clicks") == 0))
        ).alias("never_active_learner"),
        (
            (pl.col("final_result") == "Withdrawn")
            & (pl.col("date_unregistration").is_not_null())
            & (pl.col("total_vle_clicks").is_null() | (pl.col("total_vle_clicks") == 0))
        ).alias("is_zero_click_withdrawal"),
    ])

    # 5. Identify Withdrawn learners without date_unregistration and zero clicks (19 learners)
    w_no_date_no_clicks = df_active.filter(
        (pl.col("final_result") == "Withdrawn")
        & (pl.col("date_unregistration").is_null())
        & (pl.col("last_vle_date").is_null())
    )
    zero_click_no_date_w_dropped = len(w_no_date_no_clicks)

    # Exclude those 19 zero-click no-date withdrawn learners
    df_filtered = df_active.filter(
        ~(
            (pl.col("final_result") == "Withdrawn")
            & (pl.col("date_unregistration").is_null())
            & (pl.col("last_vle_date").is_null())
        )
    )

    # 6. Compute median lag = date_unregistration - last_vle_date over Withdrawn learners with real dates >= 0
    withdrawn_with_real_date = df_filtered.filter(
        (pl.col("final_result") == "Withdrawn")
        & (pl.col("date_unregistration").is_not_null())
        & (pl.col("last_vle_date").is_not_null())
    ).with_columns(
        (pl.col("date_unregistration") - pl.col("last_vle_date")).alias("lag")
    )
    median_lag = int(withdrawn_with_real_date["lag"].median())

    # 7. Impute proxy unregistration date for Withdrawn learners lacking date_unregistration (74 learners)
    df_populated = df_filtered.with_columns([
        (
            (pl.col("final_result") == "Withdrawn") & (pl.col("date_unregistration").is_null())
        ).alias("label_is_proxy"),
        pl.when(
            (pl.col("final_result") == "Withdrawn") & (pl.col("date_unregistration").is_null())
        )
        .then(
            pl.min_horizontal(
                pl.col("last_vle_date") + median_lag,
                pl.col("module_presentation_length"),
            )
        )
        .otherwise(pl.col("date_unregistration"))
        .alias("effective_date_unregistration"),
    ])

    # 8. Impute median date_registration
    median_reg_date = df_populated["date_registration"].median()
    df_populated = df_populated.with_columns([
        pl.col("date_registration").is_null().alias("has_missing_date_registration"),
        pl.col("date_registration").fill_null(median_reg_date).alias("imputed_date_registration"),
        (-pl.col("date_registration").fill_null(median_reg_date)).alias("registration_lead_time"),
    ])

    # Save to data/processed/learners.parquet
    output_path = processed_path / "learners.parquet"
    df_populated.write_parquet(output_path, compression="zstd")

    proxy_count = len(df_populated.filter(pl.col("label_is_proxy")))
    never_active_count = len(df_populated.filter(pl.col("never_active_learner")))
    zero_click_w_count = len(df_populated.filter(pl.col("is_zero_click_withdrawal")))
    proxy_by_presentation = (
        df_populated.filter(pl.col("label_is_proxy"))
        .group_by("code_presentation")
        .agg(pl.len().alias("count"))
        .sort("code_presentation")
        .to_dicts()
    )

    report = {
        "initial_learners": initial_count,
        "precourse_unreg_dropped": precourse_unreg_dropped,
        "zero_click_no_date_w_dropped": zero_click_no_date_w_dropped,
        "never_active_learners": never_active_count,
        "is_zero_click_withdrawal": zero_click_w_count,
        "final_learners_count": len(df_populated),
        "median_lag": median_lag,
        "proxy_unreg_count": proxy_count,
        "proxy_by_presentation": proxy_by_presentation,
        "output_path": str(output_path),
    }

    print(f"Population built: {len(df_populated):,} learners saved to {output_path}")
    print(f"Flagged {zero_click_w_count} is_zero_click_withdrawal learners")
    print(f"Flagged {never_active_count} never_active_learners")

    return report


if __name__ == "__main__":
    build_learner_population()
