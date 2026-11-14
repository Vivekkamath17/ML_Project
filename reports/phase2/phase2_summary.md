# Phase 2 Final Summary: Baseline Models & System Verification (OULAD)

**Date**: 2026-11-14  
**Pipeline**: OULAD Baseline Dropout & Forecasting Benchmarks  
**Quarantined Split**: `2014J` strictly unaccessed (`load_split` guard enforced).  
**Validation Split**: `2014B` (6,474 enrolled learners, 164,799 rows, base rate 2.367%).  

---

## 1. Summary of Model Performance

### 1.1 Champion v1 (39 Features) & Champion v2 (35 Features without Curriculum)
- **Champion v1 (39 features, 285 trees)**: Val ROC-AUC = **0.7736**, PR-AUC = **0.0735** (3.11× base rate).
- **Champion v2 (35 features, 329 trees, excluding high-PSI curriculum group)**: Val ROC-AUC = **0.7731**, PR-AUC = **0.0764** (3.23× base rate).
  - Removing the curriculum group (`pct_resources_visited`, `n_skipped_resources`, `pacing_vs_cohort_median`, `prerequisite_gap`) resolves presentation drift from `n_skipped_resources` (PSI = 0.4808) and boosts validation PR-AUC by **+0.0029**.
  - LOPO cross-presentation mean: ROC-AUC = 0.7891 ± 0.0116, PR-AUC = 0.0782 ± 0.0084.
- **Fresh Paired Learner-Bootstrap (500 resamples)**:
  - LightGBM minus LR-Unweighted: $\Delta$ROC = **+0.0366** [95% CI: +0.0294, +0.0453], $\Delta$PR = **+0.0117** [95% CI: +0.0073, +0.0163] (statistically significant).
  - LightGBM minus LR-Balanced: $\Delta$ROC = **+0.0298** [95% CI: +0.0227, +0.0384], $\Delta$PR = **+0.0096** [95% CI: +0.0043, +0.0146].

### 1.2 Proper Nested Cross-Validation (Zero 2014B Leakage)
- Evaluated strictly on `2013B` and `2013J` using 3-fold inner `GroupKFold` grouped by `id_student`, searching over `max_depth` $\in [2, 6]$, `num_leaves` $\in [7, 63]$, `n_estimators` $\in [150, 600]$ across 30 Optuna trials.
- Selected: `max_depth = 3`, `num_leaves = 50`, `learning_rate = 0.0666`, `n_estimators = 463`, `min_child_samples = 352`.
- Scored once on `2014B`: ROC-AUC = **0.7726**, PR-AUC = **0.0745**.
- Differs from the champion by only $\Delta\text{ROC} = -0.0010$ and $\Delta\text{PR} = +0.0010$, proving zero validation tuning inflation.

---

## 2. Reconciled Operational Policies & Lead-Time Analysis

- **Total Enrolled Eligible Learners**: 6,474 unique students at week $\ge 2$.
- **Total Withdrawn Learners**: 1,394 unique students (21.53% attrition base rate).

| Policy | Threshold $\theta$ | Flagged Learners (% of 6,474) | Withdrawn Recall (% of 1,394) | Random Recall Ref | Learner Precision | Random Prec Ref | Share of ALL WD within 3w of Unreg | Share of ALL WD $\ge 2$w Before Unreg | Median Lead (Final Run) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MCC-Optimal v1** | 0.025 | 4,108 (63.5%) | 1,144 (82.1%) | 63.5% | 27.8% | 21.5% | 980 (70.3%) | 1,064 (76.3%) | 6.0 wks |
| **MCC-Optimal v2** | 0.025 | 4,176 (64.5%) | 1,167 (83.7%) | 64.5% | 27.9% | 21.5% | 989 (70.9%) | 1,088 (78.0%) | 6.0 wks |
| **Cost $r=10$** | 0.095 | 816 (12.6%) | 322 (23.1%) | 12.6% | 39.5% | 21.5% | 187 (13.4%) | 295 (21.2%) | 5.0 wks |
| **Capacity Top-10%/wk** | Dynamic | 2,304 (35.6%) | 776 (55.7%) | 35.6% | 33.7% | 21.5% | 611 (43.8%) | 711 (51.0%) | 5.0 wks |

---

## 3. Shuffled-Label Target Leakage Distribution (20 Permutations)
- 20 independent learner-level label permutations:
  - Mean ROC-AUC = **0.4984** ($\pm 0.0264$), Median = **0.4940**, Min = 0.4637, Max = 0.5573.
  - Centered exactly at 0.500 chance level, confirming zero structural leakage.

---

## 4. Assessment Target & Normalization Audit
- Verified across OULAD: Exam submissions exist **only** for CCC (2014B, 2014J) and DDD (all 4 cohorts). AAA, BBB, EEE, FFF, GGG have 0 recorded exam submissions in `studentAssessment.csv`.
- `final_score` normalized by recorded assessment weight. Distinction mean = 85.7%, Pass mean = 70.1%, Fail mean = 24.0%, Withdrawn mean = 15.6%.
- Snapshot-fitted baseline (`mean_score_submitted` fitted per snapshot week on train non-withdrawn):
  - Week 4: Baseline MAE = 23.53, $R^2 = 0.109$ $\to$ LGBM MAE = **16.19**, $R^2 = \mathbf{0.467}$.
  - Week 8: Baseline MAE = 17.37, $R^2 = 0.414$ $\to$ LGBM MAE = **13.10**, $R^2 = \mathbf{0.618}$.
  - Week 12: Baseline MAE = 15.92, $R^2 = 0.511$ $\to$ LGBM MAE = **10.41**, $R^2 = \mathbf{0.743}$.
