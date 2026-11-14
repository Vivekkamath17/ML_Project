# OULAD Data Audit Report

Generated automatically by `src.data.audit`.

## 1. Summary of Raw Tables
| Table | Row Count | Primary / Composite Key | Duplicate Keys | Null Columns |
|---|---|---|---|---|
| assessments | 206 | `id_assessment` | 0 | `date` (11 nulls) |
| courses | 22 | `(code_module, code_presentation)` | 0 | None |
| studentAssessment | 173,912 | `(id_assessment, id_student)` | 0 | `score` (173 nulls) |
| studentInfo | 32,593 | `(code_module, code_presentation, id_student)` | 0 | `imd_band` (1,111 nulls) |
| studentRegistration | 32,593 | `(code_module, code_presentation, id_student)` | 0 | `date_registration` (45 nulls), `date_unregistration` (22,521 nulls) |
| studentVle | 10,655,280 | - | - | None |
| vle | 6,364 | `id_site` | 0 | `week_from` (5,243 nulls), `week_to` (5,243 nulls) |

**Learner Key Overlap**:
- Total unique keys in `studentInfo`: 32,593
- Total unique keys in `studentRegistration`: 32,593
- Keys matching perfectly in both: 32,593 (100% agreement, zero disjoint keys).

---

## 2. Withdrawn Learners & Unregistration Audit
- **Total Withdrawn learners**: 10,156
  - Withdrawn with explicit `date_unregistration`: **10,063** (99.08%)
  - Withdrawn with NO `date_unregistration`: **93** (0.92%)
  - Non-withdrawn learners with a `date_unregistration`: **9** (0 - perfectly consistent)
- **Unregistration Date Breakdown**:
  - Pre-course withdrawals (`date_unregistration < 0`): **2,676** (To be dropped in Step 3 population filter)
  - Post-course unregistration (`date_unregistration > course_length`): **1**
  - Normal within-course withdrawals (`0 <= date_unregistration <= course_length`): **7,386**
- **Withdrawn Without Date Proxy Analysis**:
  - Of the 93 learners with no `date_unregistration`:
    - Had at least one VLE click: **74**
    - Zero VLE clicks ever: **19**
  - **Distribution across splits**:
    - Presentation `2013B`: 1
    - Presentation `2013J`: 2
    - Presentation `2014B`: 3
    - Presentation `2014J`: 87

---

## 3. VLE Interaction Distributions
- `min_date`: -25 (pre-course activity starts up to day -25)
- `max_date`: 269
- `precourse_click_rows`: 688,988 rows occurred before Day 0 (`date < 0`).
- `total_clicks`: 39,605,099
- `sum_click` range: [1, 6,977]

---

## 4. Course Availability & Presentations Matrix
Presentations matrix (length in days):

| Module | 2013B | 2013J | 2014B | 2014J |
|---|---|---|---|---|
| AAA | None | 268 | None | 269 |
| BBB | 240 | 268 | 234 | 262 |
| CCC | None | None | 241 | 269 |
| DDD | 240 | 261 | 241 | 262 |
| EEE | None | 268 | 241 | 269 |
| FFF | 240 | 268 | 241 | 269 |
| GGG | None | 261 | 241 | 269 |

---

## 5. Known Quirks and Recommendations

### Quirk 1: `assessments.date` Missing for Final Exams
- **Audit Findings**: 11 assessments have `date` as null.
- All 11 missing dates belong to **Exam** assessment types (`Exam`: 11).
- **Recommendation**: Fall back to the course length `module_presentation_length` from `courses.parquet` for any assessment where `date` is null. Exams take place at the end of the presentation.

### Quirk 2: `vle.week_from` and `week_to` Sparsity
- **Audit Findings**: Out of 6,364 resource sites, 5,243 (82.39%) have NULL `week_from` and `week_to`.
- **Recommendation**: As planned, do not rely on `vle.week_from / week_to`. Define resource availability at week $t$ dynamically as: a resource is "available at week $t$" if and only if it has been accessed by at least one student in that same cohort on or before week $t$. This is 100% leakage-free and robust.

### Quirk 3: Missing `imd_band` Values
- **Audit Findings**: 1,111 learners in `studentInfo` have missing `imd_band`.
- **Recommendation**: Encode missing `imd_band` explicitly as `"Unknown"` in `data/processed/audit_demographics.parquet`. (Remember: demographic columns are never included in `features_weekly.parquet`).

### Quirk 4: `studentAssessment.is_banked`
- **Audit Findings**:
  - `is_banked == 0`: 172,003 submissions
  - `is_banked == 1`: 1,909 submissions
- 1,901 submissions have `is_banked == 1`. Banked assessments are previously completed modules transferred in from earlier attempts. Their `date_submitted` is often negative (e.g. -12).
- **Recommendation**: Banked assessments represent genuine prior credit available to the learner from Day 0. We treat banked assessments as submissions completed before week 0 (`date_submitted <= 0`), so they are credited to `weighted_score_to_date` and `has_any_assessment` from week 0 onward, but exclude them from current-course lateness calculations.

### Quirk 5: Missing Assessment Scores
- **Audit Findings**: 173 rows in `studentAssessment` have `score` as null.
- **Recommendation**: Do NOT impute missing scores as 0. Keep them as `NaN` and track `has_any_assessment` and `n_submitted` cleanly.

### Quirk 6: Assessment Weight Normalization
- **Audit Findings**: Sum of weights per course presentation vary (some sum to 100, while in others TMA + CMA weights sum to 100 and Exam has an additional 100 weight, totaling 200).
- **Recommendation**: When calculating `final_score` and `weighted_score_to_date`, normalize by the sum of available assessment weights rather than assuming a fixed constant denominator.
