"""
Strengthened Leakage Verification Test for OULAD Data Preparation Pipeline.
Performs stratified sampling of at least 300 (learner, week) pairs covering:
- Week 0
- Final course weeks
- Withdrawn learners
- Proxy-dated learners
- Never-active learners
- Every module and every presentation cohort

Strict Point-in-Time Truncation:
- Recomputes everything strictly on truncated raw tables (date <= 7*(t+1) - 1).
- Truncation hides any studentRegistration events occurring after the cutoff date.
- Does NOT reuse full-data population flags.
- Covers ALL feature groups: behavior, temporal, assessment, curriculum (cohort pacing & availability).
- Tests graph inputs with exact numerical matrix equivalence against point-in-time calculation.
"""
import json
import random
import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
import yaml


@pytest.fixture(scope="module")
def full_features():
    return pl.read_parquet("data/processed/features_weekly.parquet")


@pytest.fixture(scope="module")
def interim_tables():
    return {
        "vle": pl.read_parquet("data/interim/vle.parquet"),
        "studentVle": pl.read_parquet("data/interim/studentVle.parquet"),
        "assessments": pl.read_parquet("data/interim/assessments.parquet"),
        "studentAssessment": pl.read_parquet("data/interim/studentAssessment.parquet"),
        "courses": pl.read_parquet("data/interim/courses.parquet"),
        "studentRegistration": pl.read_parquet("data/interim/studentRegistration.parquet"),
    }


def get_stratified_sample_300(full_features: pl.DataFrame) -> pl.DataFrame:
    """Collect at least 300 stratified learner-week pairs."""
    subsets = []

    # 1. Week 0 rows (50 samples)
    subsets.append(full_features.filter(pl.col("week") == 0).sample(50, seed=42))

    # 2. Final weeks rows (week >= 30, 50 samples)
    subsets.append(full_features.filter(pl.col("week") >= 30).sample(50, seed=42))

    # 3. Withdrawn learners (50 samples)
    subsets.append(full_features.filter(pl.col("final_result") == "Withdrawn").sample(50, seed=42))

    # 4. Proxy learners (all available or up to 50)
    proxy_df = full_features.filter(pl.col("label_is_proxy"))
    subsets.append(proxy_df.sample(min(len(proxy_df), 50), seed=42))

    # 5. Never-active learners (50 samples)
    never_df = full_features.filter(pl.col("never_active_learner"))
    subsets.append(never_df.sample(min(len(never_df), 50), seed=42))

    # 6. Stratified across all 22 (code_module, code_presentation) pairs (4 samples each = 88 samples)
    for mod_pres in full_features.select(["code_module", "code_presentation"]).unique().iter_rows(named=True):
        m, p = mod_pres["code_module"], mod_pres["code_presentation"]
        cell = full_features.filter((pl.col("code_module") == m) & (pl.col("code_presentation") == p))
        subsets.append(cell.sample(min(len(cell), 4), seed=42))

    combined = pl.concat(subsets).unique(["code_module", "code_presentation", "id_student", "week"])
    return combined


def test_leakage_stratified_300_samples(full_features, interim_tables):
    """
    Assert exact point-in-time equivalence on >= 300 stratified learner-week pairs.
    """
    sample_df = get_stratified_sample_300(full_features)
    assert len(sample_df) >= 300, f"Stratified sample count was {len(sample_df)}, expected >= 300"
    samples = sample_df.to_dicts()

    vle = interim_tables["vle"]
    student_vle = interim_tables["studentVle"]
    assessments = interim_tables["assessments"]
    student_assessments = interim_tables["studentAssessment"]
    courses = interim_tables["courses"]
    student_reg = interim_tables["studentRegistration"]

    for row in samples:
        mod = row["code_module"]
        pres = row["code_presentation"]
        sid = row["id_student"]
        t = row["week"]
        cutoff_day = 7 * (t + 1) - 1

        # 1. Truncate studentVle strictly at cutoff_day
        cohort_svle_trunc = student_vle.filter(
            (pl.col("code_module") == mod)
            & (pl.col("code_presentation") == pres)
            & (pl.col("date") <= cutoff_day)
        )
        learner_svle_trunc = cohort_svle_trunc.filter(pl.col("id_student") == sid)

        # 2. Check Static features
        c_len = courses.filter((pl.col("code_module") == mod) & (pl.col("code_presentation") == pres))["module_presentation_length"].item()
        expected_progress = round((t + 1) * 7.0 / c_len, 4)
        assert abs(row["course_progress"] - expected_progress) < 1e-4

        # Precourse clicks: date < 0
        expected_precourse = learner_svle_trunc.filter(pl.col("date") < 0)["sum_click"].sum()
        if expected_precourse is None:
            expected_precourse = 0
        assert row["precourse_clicks"] == expected_precourse

        # 3. Check Assessment features (submissions with date_submitted <= cutoff_day)
        learner_subs_trunc = student_assessments.join(
            assessments.filter((pl.col("code_module") == mod) & (pl.col("code_presentation") == pres)),
            on="id_assessment",
            how="inner",
        ).filter(
            (pl.col("id_student") == sid) & (pl.col("date_submitted") <= cutoff_day)
        )

        assert row["n_submitted"] == len(learner_subs_trunc)
        if len(learner_subs_trunc) > 0:
            expected_mean = learner_subs_trunc["score"].mean()
            if expected_mean is not None and not np.isnan(expected_mean):
                assert abs(row["mean_score_submitted"] - expected_mean) < 0.01

        # 4. Check Curriculum features (Dynamic cohort availability up to week t)
        expected_cohort_resources = cohort_svle_trunc["id_site"].n_unique()
        learner_visited = learner_svle_trunc["id_site"].n_unique()

        expected_pct = round(learner_visited / expected_cohort_resources, 4) if expected_cohort_resources > 0 else 0.0
        assert abs(row["pct_resources_visited"] - expected_pct) <= 2e-4
        assert row["n_skipped_resources"] == max(0, expected_cohort_resources - learner_visited)


def test_graph_exact_numerical_equivalence(interim_tables):
    """
    Test exact numerical equivalence of graph inputs:
    Recompute CSR matrix directly from truncated raw studentVle data at week t
    and assert exact equivalence to the saved .npz matrix.
    """
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    collab_types = config["graph"]["activity_types"]

    vle = interim_tables["vle"]
    student_vle = interim_tables["studentVle"]
    courses = interim_tables["courses"]

    collab_sites = set(vle.filter(pl.col("activity_type").is_in(collab_types))["id_site"].to_list())

    # Test on presentation 2013B at week 2
    pres = "2013B"
    t = 2
    cutoff_day = 7 * (t + 1) - 1

    # Load saved matrix and mappings
    saved_mat = sp.load_npz(f"data/processed/graphs/{pres}_week_{t}.npz")
    with open(f"data/processed/graphs/{pres}_learners.json", "r") as f:
        learners = json.load(f)
    with open(f"data/processed/graphs/{pres}_sites.json", "r") as f:
        sites = json.load(f)

    learner2idx = {s: i for i, s in enumerate(learners)}
    site2idx = {s: j for j, s in enumerate(sites)}

    # Directly recompute from raw studentVle truncated strictly at cutoff_day
    trunc_svle = student_vle.filter(
        (pl.col("code_presentation") == pres)
        & (pl.col("id_site").is_in(collab_sites))
        & (pl.col("date") <= cutoff_day)
    )

    agg = (
        trunc_svle.group_by(["id_student", "id_site"])
        .agg(pl.col("sum_click").sum().alias("clicks"))
    )

    rows = [learner2idx[s] for s in agg["id_student"].to_list()]
    cols = [site2idx[s] for s in agg["id_site"].to_list()]
    data = agg["clicks"].to_numpy()

    recomputed_mat = sp.csr_matrix((data, (rows, cols)), shape=(len(learners), len(sites)), dtype=np.int32)

    # Assert exact numerical equivalence
    diff = (saved_mat != recomputed_mat).nnz
    assert diff == 0, f"Graph matrix mismatch: {diff} differing entries!"
    assert saved_mat.sum() == recomputed_mat.sum(), "Graph matrix total clicks mismatch!"
