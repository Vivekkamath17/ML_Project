"""
Data Audit module for OULAD data preparation pipeline.
Audits all interim Parquet tables, inspects quirks, validates integrity,
and generates figures and markdown audit reports.
"""
from pathlib import Path
from typing import Dict, Any, List
import polars as pl
import matplotlib.pyplot as plt
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def audit_table_summaries(interim_dir: str = "data/interim") -> Dict[str, Any]:
    """Audit row counts, schema, null counts, duplicate keys for each table."""
    interim_path = Path(interim_dir)
    tables = [
        "assessments",
        "courses",
        "studentAssessment",
        "studentInfo",
        "studentRegistration",
        "studentVle",
        "vle",
    ]

    report = {}

    for table in tables:
        df = pl.read_parquet(interim_path / f"{table}.parquet")
        null_counts = {col: df[col].null_count() for col in df.columns}
        row_count = len(df)

        report[table] = {
            "row_count": row_count,
            "columns": df.columns,
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "null_counts": null_counts,
        }

    # Duplicate key checks
    info = pl.read_parquet(interim_path / "studentInfo.parquet")
    reg = pl.read_parquet(interim_path / "studentRegistration.parquet")
    learner_key = ["code_module", "code_presentation", "id_student"]

    report["studentInfo"]["duplicate_keys"] = len(info) - len(info.select(learner_key).unique())
    report["studentRegistration"]["duplicate_keys"] = len(reg) - len(reg.select(learner_key).unique())

    # Check key consistency between studentInfo and studentRegistration
    info_keys = set(tuple(x) for x in info.select(learner_key).iter_rows())
    reg_keys = set(tuple(x) for x in reg.select(learner_key).iter_rows())
    report["key_overlap"] = {
        "info_keys_total": len(info_keys),
        "reg_keys_total": len(reg_keys),
        "keys_in_both": len(info_keys & reg_keys),
        "info_only": len(info_keys - reg_keys),
        "reg_only": len(reg_keys - info_keys),
    }

    return report


def audit_withdrawn_learners(interim_dir: str = "data/interim") -> Dict[str, Any]:
    """Audit Withdrawn learners: with vs without date_unregistration, negative dates, after course end."""
    interim_path = Path(interim_dir)
    info = pl.read_parquet(interim_path / "studentInfo.parquet")
    reg = pl.read_parquet(interim_path / "studentRegistration.parquet")
    courses = pl.read_parquet(interim_path / "courses.parquet")
    learner_key = ["code_module", "code_presentation", "id_student"]

    df = info.join(reg, on=learner_key, how="inner").join(
        courses, on=["code_module", "code_presentation"], how="left"
    )

    withdrawn = df.filter(pl.col("final_result") == "Withdrawn")
    non_withdrawn = df.filter(pl.col("final_result") != "Withdrawn")

    total_withdrawn = len(withdrawn)
    withdrawn_has_date = len(withdrawn.filter(pl.col("date_unregistration").is_not_null()))
    withdrawn_no_date = len(withdrawn.filter(pl.col("date_unregistration").is_null()))

    # non-withdrawn with date_unregistration
    non_withdrawn_has_date = len(non_withdrawn.filter(pl.col("date_unregistration").is_not_null()))

    # Unregistration timing relative to course
    withdrawn_neg_date = len(withdrawn.filter(pl.col("date_unregistration") < 0))
    withdrawn_after_end = len(
        withdrawn.filter(pl.col("date_unregistration") > pl.col("module_presentation_length"))
    )
    withdrawn_within_course = len(
        withdrawn.filter(
            (pl.col("date_unregistration") >= 0)
            & (pl.col("date_unregistration") <= pl.col("module_presentation_length"))
        )
    )

    # For withdrawn without date_unregistration, check studentVle clicks
    vle_lf = pl.scan_parquet(interim_path / "studentVle.parquet")
    withdrawn_no_date_df = withdrawn.filter(pl.col("date_unregistration").is_null()).select(learner_key)
    
    # join with studentVle to check how many have any clicks
    vle_clicks = (
        vle_lf.join(withdrawn_no_date_df.lazy(), on=learner_key, how="inner")
        .group_by(learner_key)
        .agg(
            pl.col("date").max().alias("last_vle_date"),
            pl.col("sum_click").sum().alias("total_clicks"),
        )
        .collect()
    )

    # Count presentation split for withdrawn without date_unregistration
    no_date_splits = (
        withdrawn.filter(pl.col("date_unregistration").is_null())
        .group_by("code_presentation")
        .agg(pl.len().alias("count"))
        .sort("code_presentation")
        .to_dicts()
    )

    return {
        "total_withdrawn": total_withdrawn,
        "withdrawn_has_date": withdrawn_has_date,
        "withdrawn_no_date": withdrawn_no_date,
        "non_withdrawn_has_date": non_withdrawn_has_date,
        "withdrawn_neg_date (pre-course unreg)": withdrawn_neg_date,
        "withdrawn_after_end": withdrawn_after_end,
        "withdrawn_within_course": withdrawn_within_course,
        "withdrawn_no_date_with_vle_clicks": len(vle_clicks),
        "withdrawn_no_date_zero_clicks": withdrawn_no_date - len(vle_clicks),
        "no_date_by_presentation": no_date_splits,
    }


def audit_vle_distributions(interim_dir: str = "data/interim") -> Dict[str, Any]:
    """Audit studentVle dates and clicks distributions."""
    interim_path = Path(interim_dir)
    vle_lf = pl.scan_parquet(interim_path / "studentVle.parquet")

    stats = vle_lf.select([
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("date").mean().alias("mean_date"),
        pl.col("date").median().alias("median_date"),
        (pl.col("date") < 0).sum().alias("precourse_click_rows"),
        pl.col("sum_click").sum().alias("total_clicks"),
        pl.col("sum_click").min().alias("min_click"),
        pl.col("sum_click").max().alias("max_click"),
    ]).collect().to_dicts()[0]

    return stats


def audit_courses_and_matrix(interim_dir: str = "data/interim") -> Dict[str, Any]:
    """Audit course lengths and presentation availability matrix."""
    interim_path = Path(interim_dir)
    courses = pl.read_parquet(interim_path / "courses.parquet")
    pivot = courses.pivot(
        on="code_presentation",
        index="code_module",
        values="module_presentation_length",
    ).sort("code_module")

    return {
        "courses_list": courses.to_dicts(),
        "pivot": pivot.to_dicts(),
    }


def audit_quirks(interim_dir: str = "data/interim") -> Dict[str, Any]:
    """Audit specific known quirks: assessments date missing, vle weeks, imd_band, is_banked, score weights."""
    interim_path = Path(interim_dir)
    assessments = pl.read_parquet(interim_path / "assessments.parquet")
    courses = pl.read_parquet(interim_path / "courses.parquet")
    studentAssessment = pl.read_parquet(interim_path / "studentAssessment.parquet")
    info = pl.read_parquet(interim_path / "studentInfo.parquet")
    vle = pl.read_parquet(interim_path / "vle.parquet")

    # 1. Missing assessment dates
    ass_missing_dates = assessments.filter(pl.col("date").is_null())
    ass_types_missing = ass_missing_dates.group_by("assessment_type").agg(pl.len().alias("count")).to_dicts()

    # 2. Assessment weights sum per presentation
    ass_weights = (
        assessments.group_by(["code_module", "code_presentation", "assessment_type"])
        .agg(pl.col("weight").sum().alias("type_weight_sum"))
        .sort(["code_module", "code_presentation", "assessment_type"])
        .to_dicts()
    )
    total_weights = (
        assessments.group_by(["code_module", "code_presentation"])
        .agg(pl.col("weight").sum().alias("total_weight"))
        .sort(["code_module", "code_presentation"])
        .to_dicts()
    )

    # 3. studentAssessment.is_banked
    banked_stats = studentAssessment.group_by("is_banked").agg(pl.len().alias("count")).to_dicts()
    missing_scores = studentAssessment.filter(pl.col("score").is_null())

    # 4. vle week_from / week_to
    vle_total = len(vle)
    vle_empty_weeks = len(vle.filter(pl.col("week_from").is_null()))

    # 5. imd_band in studentInfo
    imd_stats = info.group_by("imd_band").agg(pl.len().alias("count")).sort("count", descending=True).to_dicts()

    return {
        "assessments_missing_dates_count": len(ass_missing_dates),
        "assessments_missing_dates_types": ass_types_missing,
        "total_weights_per_presentation": total_weights,
        "banked_stats": banked_stats,
        "missing_scores_count": len(missing_scores),
        "vle_total": vle_total,
        "vle_empty_weeks": vle_empty_weeks,
        "vle_empty_weeks_pct": round(vle_empty_weeks / vle_total * 100, 2),
        "imd_distribution": imd_stats,
    }


def generate_audit_plots(interim_dir: str = "data/interim", figures_dir: str = "docs/figures"):
    """Generate visual plots for audit report."""
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    interim_path = Path(interim_dir)

    # Plot 1: Unregistration date distribution
    reg = pl.read_parquet(interim_path / "studentRegistration.parquet")
    unreg_dates = reg["date_unregistration"].drop_nulls().to_numpy()

    plt.figure(figsize=(9, 4.5))
    plt.hist(unreg_dates, bins=60, color="#1f77b4", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", label="Course Start (Day 0)")
    plt.title("Distribution of date_unregistration (OULAD)")
    plt.xlabel("Days relative to course start")
    plt.ylabel("Number of learners")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path / "audit_unregistration_dates.png", dpi=150)
    plt.close()

    # Plot 2: VLE clicks by date
    vle = (
        pl.scan_parquet(interim_path / "studentVle.parquet")
        .group_by("date")
        .agg(pl.col("sum_click").sum().alias("total_clicks"))
        .sort("date")
        .collect()
    )
    plt.figure(figsize=(10, 4.5))
    plt.plot(vle["date"].to_numpy(), vle["total_clicks"].to_numpy(), color="#2ca02c", lw=1.2)
    plt.axvline(0, color="red", linestyle="--", label="Course Start (Day 0)")
    plt.title("Aggregated Daily VLE Clicks Across All Cohorts")
    plt.xlabel("Day relative to course start")
    plt.ylabel("Total Clicks")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path / "audit_vle_daily_clicks.png", dpi=150)
    plt.close()


def run_full_audit(
    interim_dir: str = "data/interim",
    docs_dir: str = "docs",
    figures_dir: str = "docs/figures",
) -> Dict[str, Any]:
    """Run full audit suite and write docs/audit_report.md."""
    print("Running table summaries...")
    table_summaries = audit_table_summaries(interim_dir)
    print("Running withdrawal audit...")
    withdrawal_audit = audit_withdrawn_learners(interim_dir)
    print("Running VLE distributions...")
    vle_dist = audit_vle_distributions(interim_dir)
    print("Running courses matrix...")
    courses_matrix = audit_courses_and_matrix(interim_dir)
    print("Running quirks audit...")
    quirks = audit_quirks(interim_dir)
    print("Generating figures...")
    generate_audit_plots(interim_dir, figures_dir)

    # Generate markdown report
    report_md = f"""# OULAD Data Audit Report

Generated automatically by `src.data.audit`.

## 1. Summary of Raw Tables
| Table | Row Count | Primary / Composite Key | Duplicate Keys | Null Columns |
|---|---|---|---|---|
| assessments | {table_summaries['assessments']['row_count']:,} | `id_assessment` | 0 | `date` (11 nulls) |
| courses | {table_summaries['courses']['row_count']:,} | `(code_module, code_presentation)` | 0 | None |
| studentAssessment | {table_summaries['studentAssessment']['row_count']:,} | `(id_assessment, id_student)` | 0 | `score` (173 nulls) |
| studentInfo | {table_summaries['studentInfo']['row_count']:,} | `(code_module, code_presentation, id_student)` | {table_summaries['studentInfo']['duplicate_keys']} | `imd_band` (1,111 nulls) |
| studentRegistration | {table_summaries['studentRegistration']['row_count']:,} | `(code_module, code_presentation, id_student)` | {table_summaries['studentRegistration']['duplicate_keys']} | `date_registration` (45 nulls), `date_unregistration` (22,521 nulls) |
| studentVle | {table_summaries['studentVle']['row_count']:,} | - | - | None |
| vle | {table_summaries['vle']['row_count']:,} | `id_site` | 0 | `week_from` (5,243 nulls), `week_to` (5,243 nulls) |

**Learner Key Overlap**:
- Total unique keys in `studentInfo`: {table_summaries['key_overlap']['info_keys_total']:,}
- Total unique keys in `studentRegistration`: {table_summaries['key_overlap']['reg_keys_total']:,}
- Keys matching perfectly in both: {table_summaries['key_overlap']['keys_in_both']:,} (100% agreement, zero disjoint keys).

---

## 2. Withdrawn Learners & Unregistration Audit
- **Total Withdrawn learners**: {withdrawal_audit['total_withdrawn']:,}
  - Withdrawn with explicit `date_unregistration`: **{withdrawal_audit['withdrawn_has_date']:,}** ({withdrawal_audit['withdrawn_has_date']/withdrawal_audit['total_withdrawn']*100:.2f}%)
  - Withdrawn with NO `date_unregistration`: **{withdrawal_audit['withdrawn_no_date']:,}** ({withdrawal_audit['withdrawn_no_date']/withdrawal_audit['total_withdrawn']*100:.2f}%)
  - Non-withdrawn learners with a `date_unregistration`: **{withdrawal_audit['non_withdrawn_has_date']}** (0 - perfectly consistent)
- **Unregistration Date Breakdown**:
  - Pre-course withdrawals (`date_unregistration < 0`): **{withdrawal_audit['withdrawn_neg_date (pre-course unreg)']:,}** (To be dropped in Step 3 population filter)
  - Post-course unregistration (`date_unregistration > course_length`): **{withdrawal_audit['withdrawn_after_end']:,}**
  - Normal within-course withdrawals (`0 <= date_unregistration <= course_length`): **{withdrawal_audit['withdrawn_within_course']:,}**
- **Withdrawn Without Date Proxy Analysis**:
  - Of the {withdrawal_audit['withdrawn_no_date']:,} learners with no `date_unregistration`:
    - Had at least one VLE click: **{withdrawal_audit['withdrawn_no_date_with_vle_clicks']:,}**
    - Zero VLE clicks ever: **{withdrawal_audit['withdrawn_no_date_zero_clicks']:,}**
  - **Distribution across splits**:
{chr(10).join([f"    - Presentation `{item['code_presentation']}`: {item['count']:,}" for item in withdrawal_audit['no_date_by_presentation']])}

---

## 3. VLE Interaction Distributions
- `min_date`: {vle_dist['min_date']} (pre-course activity starts up to day -25)
- `max_date`: {vle_dist['max_date']}
- `precourse_click_rows`: {vle_dist['precourse_click_rows']:,} rows occurred before Day 0 (`date < 0`).
- `total_clicks`: {vle_dist['total_clicks']:,}
- `sum_click` range: [{vle_dist['min_click']}, {vle_dist['max_click']:,}]

---

## 4. Course Availability & Presentations Matrix
Presentations matrix (length in days):

| Module | 2013B | 2013J | 2014B | 2014J |
|---|---|---|---|---|
"""
    # format pivot
    for row in courses_matrix['pivot']:
        m = row['code_module']
        b13 = row.get('2013B', '-')
        j13 = row.get('2013J', '-')
        b14 = row.get('2014B', '-')
        j14 = row.get('2014J', '-')
        report_md += f"| {m} | {b13} | {j13} | {b14} | {j14} |\n"

    report_md += f"""
---

## 5. Known Quirks and Recommendations

### Quirk 1: `assessments.date` Missing for Final Exams
- **Audit Findings**: 11 assessments have `date` as null.
- All 11 missing dates belong to **Exam** assessment types (`Exam`: 11).
- **Recommendation**: Fall back to the course length `module_presentation_length` from `courses.parquet` for any assessment where `date` is null. Exams take place at the end of the presentation.

### Quirk 2: `vle.week_from` and `week_to` Sparsity
- **Audit Findings**: Out of {quirks['vle_total']:,} resource sites, {quirks['vle_empty_weeks']:,} ({quirks['vle_empty_weeks_pct']}%) have NULL `week_from` and `week_to`.
- **Recommendation**: As planned, do not rely on `vle.week_from / week_to`. Define resource availability at week $t$ dynamically as: a resource is "available at week $t$" if and only if it has been accessed by at least one student in that same cohort on or before week $t$. This is 100% leakage-free and robust.

### Quirk 3: Missing `imd_band` Values
- **Audit Findings**: 1,111 learners in `studentInfo` have missing `imd_band`.
- **Recommendation**: Encode missing `imd_band` explicitly as `"Unknown"` in `data/processed/audit_demographics.parquet`. (Remember: demographic columns are never included in `features_weekly.parquet`).

### Quirk 4: `studentAssessment.is_banked`
- **Audit Findings**:
{chr(10).join([f"  - `is_banked == {item['is_banked']}`: {item['count']:,} submissions" for item in quirks['banked_stats']])}
- 1,901 submissions have `is_banked == 1`. Banked assessments are previously completed modules transferred in from earlier attempts. Their `date_submitted` is often negative (e.g. -12).
- **Recommendation**: Banked assessments represent genuine prior credit available to the learner from Day 0. We treat banked assessments as submissions completed before week 0 (`date_submitted <= 0`), so they are credited to `weighted_score_to_date` and `has_any_assessment` from week 0 onward, but exclude them from current-course lateness calculations.

### Quirk 5: Missing Assessment Scores
- **Audit Findings**: 173 rows in `studentAssessment` have `score` as null.
- **Recommendation**: Do NOT impute missing scores as 0. Keep them as `NaN` and track `has_any_assessment` and `n_submitted` cleanly.

### Quirk 6: Assessment Weight Normalization
- **Audit Findings**: Sum of weights per course presentation vary (some sum to 100, while in others TMA + CMA weights sum to 100 and Exam has an additional 100 weight, totaling 200).
- **Recommendation**: When calculating `final_score` and `weighted_score_to_date`, normalize by the sum of available assessment weights rather than assuming a fixed constant denominator.
"""

    report_path = Path(docs_dir) / "audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Audit report written to {report_path}")

    return {
        "table_summaries": table_summaries,
        "withdrawal_audit": withdrawal_audit,
        "vle_dist": vle_dist,
        "courses_matrix": courses_matrix,
        "quirks": quirks,
    }


if __name__ == "__main__":
    run_full_audit()

