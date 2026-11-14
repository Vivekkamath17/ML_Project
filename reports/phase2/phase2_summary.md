# Phase 2 Final Summary: Baseline Models & System Verification (OULAD)

**Date**: 2026-11-14  
**Pipeline**: OULAD Baseline Dropout & Forecasting Benchmarks  
**Quarantined Split**: `2014J` strictly unaccessed (`load_split` guard enforced).  
**Validation Split**: `2014B` (6,474 enrolled learners at week $\ge 2$, 164,799 rows, base rate 2.367%).  

---

## 1. Summary of Model Performance & Selection

### 1.1 Official Baseline: LightGBM v3 (38 Features)
- **Official Model**: LightGBM v3 (38 features = 39 sanctioned features minus `n_skipped_resources`).
- **Hyperparameters (Train-Only Tuned)**: Tuned strictly on `2013B` and `2013J` via 3-fold inner `GroupKFold` (grouped by `id_student` alone) over 30 Optuna trials:
  - `max_depth = 3`, `num_leaves = 50`, `learning_rate = 0.0666`, `n_estimators = 463`, `min_child_samples = 352`.
  - Zero exposure to `2014B` during hyperparameter selection or early stopping.
- **Validation Performance ($k=3$, week $\ge 2$)**:
  - ROC-AUC: **0.7757**
  - PR-AUC: **0.0786** (3.32× base rate of 0.0237)
  - 5-Seed Headline Mean: ROC-AUC = **0.7724** ($\pm 0.0012$), PR-AUC = **0.0776** ($\pm 0.0007$).
  - Frozen-threshold MCC: **0.1444** at $\theta = 0.035$ (Note: `mcc_is_in_sample = True`, threshold chosen on val).

### 1.2 Model Selection, Selection Optimism & LOPO Context
- **Candidate Comparison**:
  - **v1 (All 39 features, 285 trees)**: Val ROC = 0.7736, PR = 0.0735 | LOPO mean ROC = 0.7905, PR = **0.0813**
  - **v2 (35 features, curriculum group dropped, 329 trees)**: Val ROC = 0.7731, PR = 0.0764 | LOPO mean ROC = 0.7891, PR = 0.0782
  - **v3 (38 features, `n_skipped_resources` dropped, 463 trees)**: Val ROC = 0.7757, PR = 0.0786 | LOPO mean ROC = 0.7915, PR = **0.0828**
- **Selection Optimism Note**: v3 was selected from the three candidate variants using validation performance on `2014B`. Consequently, its validation metrics carry modest selection optimism.
- **Statistical Significance & Direct Paired Bootstrap Comparison**:
  - Across the 3 cross-presentation LOPO folds from the same run, v3 minus v1 mean differences are $\Delta\text{ROC} = \mathbf{+0.0010}$ (0.7915 vs 0.7905) and $\Delta\text{PR} = \mathbf{+0.0015}$ (0.0828 vs 0.0813).
  - Direct paired learner-bootstrap comparison on `2014B` (500 resamples clustered by `id_student`):
    - $\Delta\text{ROC-AUC} = \mathbf{+0.0021}$ [95% CI: -0.0015, +0.0052] (95% CI spans zero, demonstrating no statistically significant ROC difference).
    - $\Delta\text{PR-AUC} = \mathbf{+0.0051}$ [95% CI: +0.0015, +0.0087].
  - **Per-Fold LOPO Deltas (v3 minus v1)**:
    - `2013B`: v1 (0.7921 ROC / 0.0861 PR) vs v3 (0.8007 ROC / 0.0983 PR) $\to$ $\Delta\text{ROC} = \mathbf{+0.0087}$, $\Delta\text{PR} = \mathbf{+0.0121}$
    - `2013J`: v1 (0.7861 ROC / 0.0628 PR) vs v3 (0.7924 ROC / 0.0687 PR) $\to$ $\Delta\text{ROC} = \mathbf{+0.0063}$, $\Delta\text{PR} = \mathbf{+0.0059}$
    - `2014B`: v1 (0.7563 ROC / 0.0673 PR) vs v3 (0.7703 ROC / 0.0760 PR) $\to$ $\Delta\text{ROC} = \mathbf{+0.0141}$, $\Delta\text{PR} = \mathbf{+0.0086}$
  - LOPO serves strictly as a cross-presentation **stability range** across course structures, not as an alternative validation score.

### 1.3 Empirical Justification for Pruning `n_skipped_resources`: Module Mix & Feature Proxies
Pruning `n_skipped_resources` is justified by drift in **module mix** across presentations, not performance optimization:
- **Resource Count Definition & Module Mix Drift**:
  - *Definition*: Catalog resource count is defined as the total number of VLE activity entries registered in raw `vle.csv` for each module-presentation.
  - Per-module counts vary across presentations (e.g., BBB: 315 in 2013B, 321 in 2013J, 311 in 2014B, 207 in 2014J; DDD: 428 in 2013B, 462 in 2013J, 453 in 2014B, 365 in 2014J).
  - The severe presentation drift is driven by **presentation module mix**: `2013B` contained only 3 large-catalog STEM modules (BBB 315, DDD 428, FFF 508; mean 278.7 skipped), whereas `2014B` introduced low-catalog modules CCC (196 resources), EEE (109 resources), and GGG (124 resources). (Note: module AAA was offered only in J presentations [2013J: 211, 2014J: 202] and never existed in either 2013B or 2014B).
  - Total VLE catalog entries reflect this mix: 2013B (1,251), 2013J (1,772), 2014B (1,671), 2014J (1,670).
- **Curriculum Feature Population Stability Index (PSI)**:
  - `pct_resources_visited`: $\text{PSI} = \mathbf{0.0129}$ (stable, $< 0.10$)
  - `n_skipped_resources`: $\text{PSI} = \mathbf{0.4808}$ (**severe population shift**, $> 0.25$; Train mean = 249.94 vs Val mean = 189.87)
  - `pacing_vs_cohort_median`: $\text{PSI} = \mathbf{0.0052}$ (stable, $< 0.10$)
  - `prerequisite_gap`: $\text{PSI} = \mathbf{0.0130}$ (stable, $< 0.10$)
- **Module Identity Proxy Audit (vs Majority-Class Baseline)**:
  - In Validation `2014B`, the largest module cohort is BBB (1,396 learners), giving a naive **majority-class baseline accuracy of 21.6%**.
  - `n_skipped_resources`: acts strongly as an unintended **module-identity proxy**, achieving **48.4% validation accuracy (51.8% balanced accuracy)** across the 6 modules—more than 2.2× higher than majority-class chance.
  - `studied_credits`: is disclosed as a **partial module proxy**, achieving **34.3% accuracy (33.8% balanced accuracy)** because standard credit requirements differ by faculty (e.g. 30 vs 60 credits).
  - `course_progress`: exhibits no module proxy capacity (**20.4% accuracy, 16.7% balanced accuracy**, below majority-class chance level).
  - Pruning `n_skipped_resources` eliminates this structural module-proxy artifact while preserving normalized curriculum metrics.

### 1.4 Paired Learner-Bootstrap Baseline Comparison
Paired bootstrap (500 learner-level resamples on 2014B, week $\ge 2$):
- **LightGBM v3 minus LR-Unweighted**:
  - $\Delta\text{ROC-AUC} = \mathbf{+0.0387}$ [95% CI: +0.0312, +0.0475]
  - $\Delta\text{PR-AUC} = \mathbf{+0.0169}$ [95% CI: +0.0120, +0.0221]
- **LightGBM v3 minus LR-Balanced**:
  - $\Delta\text{ROC-AUC} = \mathbf{+0.0319}$ [95% CI: +0.0245, +0.0406]
  - $\Delta\text{PR-AUC} = \mathbf{+0.0148}$ [95% CI: +0.0091, +0.0203]

---

## 2. Reconciled Operational Policies & Lead-Time Analysis (Official v3, $k=3$)

Evaluated on Validation `2014B` at week $\ge 2$:
- **Eligible Enrolled Cohort**: 6,474 unique students (denominator for flagged population).
- **Withdrawn Cohort**: 1,395 unique students (denominator for attrition capture; 21.55% base rate).
- **Reconciliation Note**: $5,079\text{ (active non-withdrawn)} + 1,395\text{ (active withdrawn)} = \mathbf{6,474}$ unique learners. The earlier 1,394 figure arose from filtering on `date_unregistration.notna()`, which omitted learner `(DDD, 2014B, 227826)` who had a proxy unregistration date (`label_is_proxy = True`).
- **Cost $r=10$ Threshold Note**: The cost optimization grid search identified $\theta = 0.1124$ as the exact operating minimum (flagging 1,232 learners and catching 494 dropouts). Applying the rounded threshold $\theta = 0.1120$ flags 1,243 learners (19.2%) and catches 498 dropouts (35.7%), capturing 11 additional borderline students in $[0.1120, 0.1124)$ (4 of whom withdrew). Both points exhibit identical learner precision (40.1%).

| Policy | Threshold $\theta$ | Flagged Learners (Denom=6,474) | Withdrawn Caught (Denom=1,395) | Learner Precision | Random Exp | Lift Over Random | Share of ALL WD within 3w | Share of ALL WD $\ge 2$w | Lead Time Q1 / Med / Q3 (Final Run) | Flagged Learners per Catch | Flag-Weeks per Catch | MCC Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MCC-Optimal** | 0.035 | 3,653 (56.4%) | 1,099 (78.8%) | 30.1% | 56.4% | 1.40× | 898 (64.4%) | 1,017 (72.9%) | 3.0 / 5.0 / 12.0 wks | 3.3 | 28.1 | in-sample on val (threshold chosen on val) |
| **Cost $r=10$** | 0.112 | 1,243 (19.2%) | 498 (35.7%) | 40.1% | 19.2% | 1.86× | 309 (22.2%) | 447 (32.0%) | 2.0 / 5.0 / 11.0 wks | 2.5 | 9.8 | N/A (cost-optimal) |
| **Capacity Top-10%/wk** | Dynamic | 2,269 (35.0%) | 777 (55.7%) | 34.2% | 35.0% | 1.59× | 616 (44.2%) | 706 (50.6%) | 2.0 / 5.0 / 10.0 wks | 2.9 | 21.2 | N/A (capacity-optimal) |

> **Advisor Summary Translation**: Under the MCC-optimal policy, advisors flag 56.4% of enrolled students across the term to catch 78.8% of eventual dropouts with an average of **3.3 flagged learners per successful catch** (28.1 flag-weeks generated per catch); under weekly resource constraints, the Top-10%/week capacity policy captures 55.7% of dropouts contacting 35.0% of students (**2.9 flagged learners per catch**, 21.2 flag-weeks per catch), while the Cost $r=10$ policy catches 35.7% of dropouts contacting 19.2% of students (**2.5 flagged learners per catch**, 9.8 flag-weeks per catch). Lead times from the final continuous alert run show early warning: Median is 5.0 weeks for all three policies, with upper quartiles (Q3) reaching 10.0 to 12.0 weeks.

---

## 3. Shuffled-Label Target Leakage Distribution (20 Permutations)
- 20 independent learner-level label permutations on `2014B`:
  - Mean ROC-AUC = **0.4984** ($\pm 0.0264$), Median = **0.4940**, Min = 0.4637, Max = 0.5573.
  - Centered exactly at 0.500 chance level, confirming zero structural target leakage.

---

## 4. Assessment Target & Normalization Audit

### 4.1 Cohort Audit & The 112 Never-Active Learners
- In Validation (2014B), 5,191 non-withdrawn students were enrolled at course start.
- **Audit Findings**: Exactly 112 learners have `never_active_learner == True` (0 course VLE clicks).
- **100% of these 112 students finished with `final_result == 'Fail'`** (0 Pass, 0 Distinction).
- **Unified Cohort Rule for Score Forecasting**: We adopt the unified rule: **exclude never-active learners** ($N = 5,079$ unique active non-withdrawn learners across weeks 2–33; breakdown: 2,574 Pass, 1,721 Fail, 784 Distinction). This rule applies consistently to all regressions and baseline tables.

### 4.2 Module Presentation Assessment Structure & CCC Exam Verification
- **CCC Assessment Breakdown**: In the raw catalog (`assessments.csv`), CCC lists 10 assessments totaling 300.0 catalog weight (4 CMAs [weight 25], 4 TMAs [weight 75], and 2 Exams [each weight 100: IDs 24290 and 40087]).
  - Inspection of `studentAssessment.csv` confirms that **only one exam (ID 24290) is recorded** (747 submissions in 2014B); exam 40087 has 0 submissions. Recorded weight is therefore 200.0.
- **Other Modules**:
  - **DDD**: Catalog weight 200.0 (100 continuous + 100 exam); both continuous and exam submissions are recorded in `studentAssessment.csv` (recorded weight = 200.0).
  - **GGG**: Catalog weight 100.0 (10 TMAs with weight 0 + 1 exam with weight 100 unrecorded $\to$ recorded weight = 0.0; evaluated via unweighted TMA mean).
  - **AAA, BBB, EEE, FFF**: Catalog weight 200.0 (100 continuous recorded + 100 exam unrecorded $\to$ recorded weight = 100.0).

### 4.3 Validation 2014B Performance by Module (Snapshot Weeks 8 and 12)
Evaluated on the exact same 5,079 active learners (superseding all earlier pooled week 8/12 figures):

| Module | Active $n$ | Cohort Mean Score | Exam Recorded? | Dist Mean | Pass Mean | Fail Mean | W8 Mod-Fit Baseline MAE ($R^2$) | W8 LightGBM MAE ($R^2$) | W12 Mod-Fit Baseline MAE ($R^2$) | W12 LightGBM MAE ($R^2$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BBB** | 1,088 | 56.3% | False | 82.0% | 66.9% | 28.0% | 15.23 (0.402) | **11.65 (0.607)** | 14.94 (0.405) | **9.48 (0.743)** |
| **CCC** | 1,018 | 49.4% | True (1 of 2) | 84.9% | 62.5% | 12.6% | 18.05 (0.475) | **16.33 (0.537)** | 17.36 (0.506) | **15.07 (0.579)** |
| **DDD** | 728 | 51.1% | True (1 of 1) | 85.5% | 65.3% | 14.1% | 18.28 (0.453) | **13.52 (0.656)** | 17.36 (0.498) | **10.61 (0.765)** |
| **EEE** | 512 | 63.7% | False | 90.8% | 76.1% | 28.2% | **13.48 (0.532)** | 13.83 (0.523)* | 12.75 (0.588) | **8.96 (0.784)** |
| **FFF** | 1,026 | 58.4% | False | 90.3% | 73.4% | 27.2% | 16.78 (0.446) | **11.63 (0.692)** | 16.79 (0.455) | **8.66 (0.821)** |
| **GGG** | 707 | 71.1% | False (Unwtd TMA) | 85.0% | 80.0% | 49.9% | 12.72 (0.423) | **11.86 (0.496)** | 9.75 (0.684) | **8.53 (0.766)** |
| **Overall** | **5,079** | **57.4%** | - | **85.7%** | **70.1%** | **24.0%** | **15.69 (0.457)** | **12.59 (0.638)** | **14.80 (0.521)** | **9.81 (0.781)** |

*\*Note on EEE at Week 8*: LightGBM MAE (13.83) is slightly worse than the per-module fitted `mean_score_submitted` baseline (13.48) due to small cohort size ($n=512$) and high early submission scores prior to complex assignments. By Week 12, LightGBM substantially outperforms the baseline (8.96 vs 12.75 MAE, $R^2 = 0.784$ vs $0.588$). Earlier pooled snapshot numbers are officially superseded by these per-module figures.
