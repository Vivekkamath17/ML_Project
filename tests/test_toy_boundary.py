"""
Unit tests for Step 5 boundary and censoring conditions on toy examples.
"""
import pytest
import polars as pl


def compute_toy_labels(df: pl.DataFrame, k: int = 3) -> pl.DataFrame:
    """Helper to compute labels on toy learner-week dataframe."""
    lower_bound = 7 * (pl.col("week") + 1)
    upper_bound = 7 * (pl.col("week") + 1 + k)

    return df.with_columns([
        (upper_bound > pl.col("module_presentation_length")).alias(f"is_censored_k{k}"),
        (
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
        ),
    ])


def test_dropout_boundary_half_open_interval():
    """
    Test the (7*(t+1), 7*(t+1+k)] boundary for t=0, k=3:
    lower_bound = 7 * 1 = 7.
    upper_bound = 7 * 4 = 28.
    Interval: (7, 28].
    - Day 7: exactly on lower boundary -> 0 (not in interval)
    - Day 8: inside interval -> 1
    - Day 28: exactly on upper boundary -> 1 (inclusive)
    - Day 29: past upper boundary -> 0
    """
    toy = pl.DataFrame({
        "week": [0, 0, 0, 0],
        "module_presentation_length": [240, 240, 240, 240],
        "is_enrolled_at_week_t": [True, True, True, True],
        "effective_date_unregistration": [7, 8, 28, 29],
    })

    res = compute_toy_labels(toy, k=3)
    labels = res["label_dropout_k3"].to_list()
    assert labels == [0, 1, 1, 0], f"Expected [0, 1, 1, 0] on boundary test, got {labels}"


def test_censoring_against_course_length():
    """
    Test right-censoring when 7*(t+1+k) > module_presentation_length.
    Example: course length = 240 days (~34.2 weeks, max_week=34).
    For k=3:
    - At week 30: 7 * (30 + 1 + 3) = 7 * 34 = 238 <= 240 -> False
    - At week 31: 7 * (31 + 1 + 3) = 7 * 35 = 245 > 240 -> True
    """
    toy = pl.DataFrame({
        "week": [30, 31],
        "module_presentation_length": [240, 240],
        "is_enrolled_at_week_t": [True, True],
        "effective_date_unregistration": [None, None],
    })

    res = compute_toy_labels(toy, k=3)
    censored = res["is_censored_k3"].to_list()
    assert censored == [False, True], f"Expected [False, True], got {censored}"

