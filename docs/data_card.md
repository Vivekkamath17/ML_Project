# OULAD Data Preparation Pipeline - Data Card & Assumptions Log

This document records all architectural choices, data engineering specifications, assumption logs, and leakage safeguards established during the preparation of the Open University Learning Analytics Dataset (OULAD).

---

## 1. Pipeline Overview & Key Architecture
- **Raw Data Directory**: `data/raw/` (Read-only inputs: 7 CSV files). Never modified.
- **Interim Directory**: `data/interim/` (Type-explicit Parquet tables, intermediate grid, feature group parquets).
- **Processed Directory**: `data/processed/` (Final unified `features_weekly.parquet`, `learners.parquet`, `labels.parquet`, `audit_demographics.parquet`, `splits.yaml`, and `graphs/`).
- **Feature Manifest**: `config/features.yaml` (Exhaustive, machine-enforced whitelist of sanctioned model features).
- **Deterministic Orchestration**: Complete pipeline regenerated from scratch via `python -m src.data.build_all` in under 30 seconds.

---

## 2. Core Definitions & Global Rules

### 2.1 Unique Learner Key
- The composite identifier `(code_module, code_presentation, id_student)` is used everywhere. `id_student` alone is not unique across presentations or course offerings.

### 2.2 Relative Temporal Scale & Week Binning
- All OULAD dates represent days relative to course start (Day 0).
- `week = floor(date / 7)`.
- Pre-course activity (`date < 0`) is aggregated into week `-1` and preserved as a static baseline `precourse_clicks`.
- The evaluation grid starts at `week = 0`.
- Tunable parameter `min_week` (default 2 in `config/config.yaml`) defines the training/evaluation start point while preserving weeks 0–1 in the table for trajectory feature computation.

### 2.3 Strict Point-in-Time Leakage Rule
- For any learner at week $t$, features use **ONLY** data with interaction date:
  $$\text{day} \le 7(t + 1) - 1$$
- No information beyond this boundary (future clicks, future registrations, subsequent submissions, future unregistrations) is accessible.

### 2.4 Strict Separation of Demographic Attributes and Targets
- Model feature tables contain zero target columns (`final_result`, `final_score`, `grade_band`, `label_*`) and zero demographic columns (`gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`).
- Demographic attributes are strictly isolated in `data/processed/audit_demographics.parquet` for downstream algorithmic fairness audits.

---

## 3. Assumptions, Quirk Handlings & Population Funnel

### 3.1 Population Funnel & Drop Reasons
From the raw 32,593 records in `studentRegistration` $\cap$ `studentInfo`:
1. **Pre-Course Withdrawals (`date_unregistration < 0`)**:
   - **2,678 learners** unregistered before course start (2,676 Withdrawn, 2 Fail).
   - *Action*: Dropped in Step 3 population filtering as they never participated in the active course cohort.
2. **Withdrawn Learners Lacking Dates with Zero Clicks**:
   - **19 learners** had `final_result == 'Withdrawn'`, null `date_unregistration`, and zero VLE clicks ever.
   - *Action*: Dropped in Step 3 because an interaction-based proxy date cannot be established.
   - *Assumption Note*: This is a minor filter using overall dataset presence (zero clicks ever); fully documented here.
3. **Withdrawn Learners with Proxy Unregistration Dates**:
   - **74 learners** had `final_result == 'Withdrawn'`, null `date_unregistration`, but logged VLE clicks.
   - Empirical lag between unregistration and last VLE click across 6,631 learners with verified dates yielded $\text{median\_lag} = 5\text{ days}$.
   - Assigned $\text{proxy\_date} = \min(\text{last\_vle\_date} + 5, \text{module\_presentation\_length})$ and flagged `label_is_proxy = True`.
   - **12 learners** whose proxy date fell before Day 0 had no eligible weeks in the grid and were pruned in Step 4.
4. **Final Filtered Population**: Exactly **29,896 learners** across 22 presentations.
5. **Dense Week Grid**: Evaluates all active weeks from week 0 to course length, resulting in **925,209 learner-week rows** across 29,884 eligible learners.

### 3.2 The 9 Non-Withdrawn Learners with Unregistration Dates
- Exactly 9 learners have `final_result == 'Fail'` but recorded a `date_unregistration`:
  - 2 unregistered pre-course (Day -7, Day -4) and were dropped in Step 3.
  - 6 unregistered on Day 0 and 1 on Day 166.
  - *Action*: Kept in active population; their administrative unregistration dates govern grid termination and dropout labeling, while retaining `final_result = 'Fail'` and `grade_band = 'Fail'`.

### 3.3 Zero-Click Learners & Population Flags
- **`never_active_learner`**: 377 learners (374 Fail, 3 Pass) never touched the VLE during the course but remained registered until course completion. Flagged with `never_active_learner = True` (13,829 grid rows). Config option `exclude_never_active_from_training: true` allows modelers to exclude or include them cleanly.
- **`is_zero_click_withdrawal`**: 756 learners withdrew during the course with an explicit unregistration date without ever logging VLE clicks. Flagged explicitly.

### 3.4 Dynamic Curriculum Availability & Cohort Pacing
- Over 82% of `vle.csv` entries have null `week_from / week_to`.
- *Action*: A resource is defined as "available at week $t$" if and only if it was accessed by at least one student in that same presentation cohort on or before week $t$.
- Pacing is defined relative to the cumulative cohort median clicks at week $t$.

### 3.5 Missing Assessment Dates & Banked Transfer Credits
- 11 assessments (all `Exam`) had null `date`. Imputed using the presentation length `module_presentation_length` (exams conclude the course).
- `is_banked == 1` submissions (1,901 rows) represent verified transfer credit from prior modules. Credited to cumulative performance (`weighted_score_to_date`, `has_any_assessment`, `n_banked`) starting from week 0, but excluded from current course submission lateness metrics.

### 3.6 Assessment Coverage Quirks & Target Score Normalization
- **Exam Recording Discrepancy**: In OULAD `studentAssessment.csv`, exam submissions exist **ONLY** for modules **CCC** (2014B: 747, 2014J: 1,168) and **DDD** (all 4 presentations: 524 to 968 submissions).
- **Exam Absence Across Other Modules**: In **AAA, BBB, EEE, FFF, and GGG**, exam records are **completely absent** from `studentAssessment.csv` across all presentations, despite an exam assessment entry with weight 100 existing in the catalog. (FFF 2014B and 2014J equally lack exam submissions, correcting earlier claims).
- **Target Score Normalization**: In `src/data/labels.py`, `final_score` is normalized by the sum of weights of assessments that actually have recorded scores in that module-presentation. Dividing by total catalog weight (typically 200) erroneously capped learner scores at 50% for exam-missing modules. For GGG (where TMAs have catalog weight 0 and exam submissions are absent), the average submitted assessment score is used. This aligns final scores with academic grading bands: Distinction mean = 85.7%, Pass mean = 70.1%, Fail mean = 24.0%.

---

## 4. Multi-Horizon Targets & Censoring Rules

### 4.1 Administrative Dropout Target (`label_dropout_k{2,3,4}`)
$$\text{label\_dropout\_k} = 1 \iff 7(t + 1) < \text{effective\_date\_unregistration} \le 7(t + 1 + k)$$
- Eligible rows: `is_enrolled_at_week_t == True`.
- Censored rows: `is_censored_k == True` when $7(t + 1 + k) > \text{course\_length}$.

### 4.2 Disengagement Onset Target (`label_inactive_k{2,3,4}`)
$$\text{label\_inactive\_k} = 1 \iff (\text{weekly\_clicks}_t > 0) \land \left(\sum_{w=t+1}^{t+k} \text{weekly\_clicks}_w == 0\right)$$
- Eligible rows: `is_enrolled_at_week_t == True`, non-censored, and actively clicking in week $t$.
- Monotonicity verified: On rows eligible for all horizons, $\text{label\_inactive\_k4} \le \text{label\_inactive\_k3} \le \text{label\_inactive\_k2}$ with 0 violations.
- Post-assessment period tagged with `is_post_assessment_period` (computed from assessment deadlines only).

---

## 5. Temporal Splits & Cohort Coverage Warnings

| Split | Presentations | Total Rows | Learners | Unique Students | Modules Present |
|---|---|---|---|---|---|
| **Train** | 2013B, 2013J | 403,137 | 12,554 | 12,418 | AAA, BBB, DDD, EEE, FFF, GGG |
| **Validation** | 2014B | 204,118 | 6,994 | 6,622 | BBB, CCC, DDD, EEE, FFF, GGG |
| **Test** | 2014J | 317,954 | 10,336 | 9,877 | AAA, BBB, CCC, DDD, EEE, FFF, GGG |

### Cohort Coverage & Leakage Caveats:
1. **Module CCC**: Present in Validation (2014B) and Test (2014J), but **unseen in Train** (2013B, 2013J). Models must generalize across unseen course structures.
2. **Modules AAA & EEE**: Present in Train and Test, but **absent from Validation** (2014B).
3. **Cross-Split Repeat Students**:
   - 2,125 unique students appear in multiple presentations across splits.
   - In the Test split (2014J), 1,442 students (42,732 rows) overlap with Train/Val.
   - A clean cold-start evaluation removing these repeat learners leaves 275,222 strictly unseen test rows.
4. **StandardScaler**: Fitted strictly on the 403,137 Train rows for 39 continuous features and serialized to `models/preprocessing/train_scaler.joblib`.
