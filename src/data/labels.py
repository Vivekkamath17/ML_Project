"""
Labels module for OULAD data preparation pipeline.
Computes multi-horizon dropout labels (label_dropout_k{2,3,4}),
inactivity onset labels (label_inactive_k{2,3,4}), persistent inactivity (label_inactive_persistent_k{2,3,4}),
right-censoring indicators, and performance targets (final_score, grade_band).
Saves to data/processed/labels.parquet and generates dropout/inactivity rate curves.
"""
from pathlib import Path
from typing import Dict, Any, List
import polars as pl
import matplotlib.pyplot as plt
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_labels(
    interim_dir: str = "data/interim",
    processed_dir: str = "data/processed",
    figures_dir: str = "docs/figures",
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Computes labels and performance targets:
    1. Dropout labels for k in [2, 3, 4]:
       label_dropout_k{k} = 1 if effective_date_unregistration in (7*(t+1), 7*(t+1+k)], else 0.
       Eligible rows: is_enrolled_at_week_t == True.
       Censored rows: is_censored_k{k} = True if 7*(t+1+k) > module_presentation_length.
    2. Inactivity ONSET labels for k in [2, 3, 4]:
       label_inactive_k{k} = 1 if learner has 0 VLE clicks in all weeks t+1..t+k, else 0.
       Eligible only when weekly_clicks > 0 and is_enrolled_at_week_t == True.
       Subject to same right-censoring rule (null if censored).
    3. Persistent Inactivity diagnostic label:
       label_inactive_persistent_k{k} = 1 if 0 clicks in t+1..t+k regardless of clicks at week t.
    4. Post assessment period:
       is_post_assessment_period indicates if week t > last assessment deadline.
    5. Performance targets:
       final_score = sum(score * weight) / presentation_total_weight.
       grade_band = Distinction / Pass / Fail; null for Withdrawn and late unregistration learners.
    6. Save to data/processed/labels.parquet.
    """
    interim_path = Path(interim_dir)
    processed_path = Path(processed_dir)
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    learner_key = config["keys"]["learner"]
    grid_key = config["keys"]["grid"]
    horizons: List[int] = config["horizons"]["values"]

    grid = pl.read_parquet(interim_path / "grid.parquet").sort(learner_key + ["week"])
    assessments = pl.read_parquet(interim_path / "assessments.parquet")
    student_assessments = pl.read_parquet(interim_path / "studentAssessment.parquet")

    # 1. Multi-horizon dropout labels, censoring, and inactivity labels
    label_exprs = []
    for k in horizons:
        lower_bound = 7 * (pl.col("week") + 1)
        upper_bound = 7 * (pl.col("week") + 1 + k)

        censored_expr = (
            upper_bound > pl.col("module_presentation_length")
        ).alias(f"is_censored_k{k}")

        dropout_expr = (
            pl.when(~pl.col("is_enrolled_at_week_t"))
            .then(None)
            .when(
                pl.col("effective_date_unregistration").is_not_null()
                & (pl.col("effective_date_unregistration") > lower_bound)
                & (pl.col("effective_date_unregistration") <= upper_bound)
            )
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"label_dropout_k{k}")
        )

        # Future clicks in weeks t+1..t+k
        shifts = [pl.col("weekly_clicks").shift(-i).over(learner_key).fill_null(0) for i in range(1, k + 1)]
        future_clicks = pl.sum_horizontal(shifts)

        # Inactivity ONSET: eligible only when weekly_clicks > 0
        onset_expr = (
            pl.when(~pl.col("is_enrolled_at_week_t") | (upper_bound > pl.col("module_presentation_length")) | (pl.col("weekly_clicks") == 0))
            .then(None)
            .when(future_clicks == 0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"label_inactive_k{k}")
        )

        # Persistent Inactivity diagnostic
        persistent_expr = (
            pl.when(~pl.col("is_enrolled_at_week_t") | (upper_bound > pl.col("module_presentation_length")))
            .then(None)
            .when(future_clicks == 0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias(f"label_inactive_persistent_k{k}")
        )

        label_exprs.extend([censored_expr, dropout_expr, onset_expr, persistent_expr])

    labels_df = grid.select(
        grid_key + [
            "module_presentation_length",
            "effective_date_unregistration",
            "is_enrolled_at_week_t",
            "is_post_assessment_period",
            "label_is_proxy",
            "never_active_learner",
            "is_zero_click_withdrawal",
            "final_result",
            "weekly_clicks",
        ]
    ).with_columns(label_exprs)

    # 2. Performance targets: final_score normalized by recorded assessment weight
    # Many modules (e.g. AAA, BBB, EEE, FFF) have unrecorded exams (weight 100 in assessments.csv
    # but 0 records in studentAssessment.csv). Normalizing by recorded assessment weight ensures
    # the target is on a true 0-100 scale rather than being capped at 50%.
    active_ass_ids = student_assessments.filter(pl.col("score").is_not_null())["id_assessment"].unique().to_list()
    ass_with_records = assessments.filter(pl.col("id_assessment").is_in(active_ass_ids))
    pres_weights = ass_with_records.group_by(["code_module", "code_presentation"]).agg(
        pl.col("weight").sum().alias("recorded_weight")
    )

    sa_weighted = student_assessments.join(
        assessments.select(["id_assessment", "code_module", "code_presentation", "weight"]),
        on="id_assessment",
        how="inner",
    )

    learner_scores = sa_weighted.group_by(learner_key).agg([
        (pl.col("score").fill_null(0.0) * pl.col("weight")).sum().alias("sum_score_weight"),
        pl.col("score").mean().alias("mean_unweighted_score"),
    ]).join(
        pres_weights, on=["code_module", "code_presentation"], how="left"
    ).with_columns(
        pl.when(pl.col("recorded_weight") > 0)
        .then(pl.col("sum_score_weight") / pl.col("recorded_weight"))
        .otherwise(pl.col("mean_unweighted_score"))
        .round(4)
        .alias("final_score")
    )

    # 3. Join final_score and compute grade_band
    labels_df = labels_df.join(
        learner_scores.select(learner_key + ["final_score"]),
        on=learner_key,
        how="left",
    ).with_columns(
        pl.col("final_score").fill_null(0.0),
        pl.when(
            (pl.col("final_result") == "Withdrawn")
            | (
                pl.col("effective_date_unregistration").is_not_null()
                & (pl.col("effective_date_unregistration") > pl.col("module_presentation_length"))
            )
        )
        .then(None)
        .otherwise(
            pl.when(pl.col("final_result") == "Distinction")
            .then(pl.lit("Distinction"))
            .when(pl.col("final_result") == "Pass")
            .then(pl.lit("Pass"))
            .when(pl.col("final_result") == "Fail")
            .then(pl.lit("Fail"))
            .otherwise(None)
        )
        .alias("grade_band"),
    )

    # 4. Save to data/processed/labels.parquet
    output_path = processed_path / "labels.parquet"
    labels_df.write_parquet(output_path, compression="zstd")

    # 5. Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k in horizons:
        k_df = (
            labels_df.filter(
                pl.col("is_enrolled_at_week_t") & ~pl.col(f"is_censored_k{k}")
            )
            .group_by("week")
            .agg([
                pl.col(f"label_dropout_k{k}").mean().alias("dropout_rate"),
                pl.col(f"label_inactive_k{k}").mean().alias("onset_rate"),
                pl.len().alias("n_eligible_dropout"),
                pl.col(f"label_inactive_k{k}").drop_nulls().len().alias("n_eligible_onset"),
            ])
            .sort("week")
        )

        weeks = k_df["week"].to_numpy()
        d_rates = k_df["dropout_rate"].to_numpy()
        o_rates = k_df["onset_rate"].to_numpy()

        overall_dropout = labels_df.filter(
            pl.col("is_enrolled_at_week_t") & ~pl.col(f"is_censored_k{k}")
        )[f"label_dropout_k{k}"].mean()

        overall_onset = labels_df.filter(
            pl.col(f"label_inactive_k{k}").is_not_null()
        )[f"label_inactive_k{k}"].mean()

        axes[0].plot(weeks, d_rates * 100, marker="o", markersize=3, label=f"k={k} (Avg: {overall_dropout*100:.2f}%)")
        axes[1].plot(weeks, o_rates * 100, marker="s", markersize=3, label=f"k={k} (Avg: {overall_onset*100:.2f}%)")

    axes[0].set_title("Dropout Rate by Week (t, t + k]")
    axes[0].set_xlabel("Course Week (t)")
    axes[0].set_ylabel("Dropout Rate (%)")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].set_title("Disengagement ONSET Rate (t+1..t+k=0 | clicks_t>0)")
    axes[1].set_xlabel("Course Week (t)")
    axes[1].set_ylabel("Onset Rate (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_file = fig_path / "dropout_and_inactivity_by_horizon.png"
    plt.savefig(fig_file, dpi=150)
    plt.close()

    print(f"Labels built: {len(labels_df):,} rows -> {output_path}")
    print(f"Saved plot -> {fig_file}")

    return {"output_path": str(output_path)}


if __name__ == "__main__":
    build_labels()
