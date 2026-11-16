# Algorithmic Fairness & Demographic Disparity Audit: lgbm_v3

**Evaluation Split**: `2014B` (Validation Cohort: 6,474 enrolled learners, 164,799 learner-weeks)
**Target**: `label_dropout_k3` ($k=3$ weeks ahead, weeks $\ge 2$)
**Operational Policy**: Capacity Top-10% per week (whole-cohort cutoff)
**Confidence Intervals**: 95% Learner-Clustered Bootstrap (500 resamples by `id_student`)

---

## 1. Headline Fairness Gaps by Demographic Attribute

Below are the observed demographic disparities across all 6 evaluated attributes under the official LightGBM v3 model. Disparities are evaluated under the project default capacity policy (Top-10% weekly cohort caseload).

| Attribute | Groups | Dem. Parity Diff (Δ Selection Rate) | Equalized Odds Diff (max(ΔTPR, ΔFPR)) | Max-Min TPR Gap | Max-Min FPR Gap | Max-Min ROC Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **imd_band** | 11 | 0.0558 | 0.2005 | 0.2005 | 0.0576 | 0.1130 |
| **region** | 13 | 0.0383 | 0.1768 | 0.1768 | 0.0388 | 0.1079 |
| **highest_education** | 5 | 0.0622 | 0.1212 | 0.1212 | 0.0590 | 0.0753 |
| **age_band** | 3 | 0.0699 | 0.1022 | 0.1022 | 0.0712 | 0.0280 |
| **gender** | 2 | 0.0411 | 0.0982 | 0.0982 | 0.0380 | 0.0011 |
| **disability** | 2 | 0.0338 | 0.0845 | 0.0845 | 0.0293 | 0.0098 |

---

## 2. Detailed Subgroup Breakdown (Official Model: LightGBM v3)

Subgroups marked with `*` indicate small-sample caution ($N < 300$ learners or $< 30$ positive dropouts). Learner-level metrics reflect unique students flagged $\ge 1$ time over the term and unique eventual dropouts caught.

### Attribute: `imd_band`

| Group | Learners (Rows) | Positives (Base Rate) | Unstable? | Selection Rate [95% CI] | Recall / TPR [95% CI] | Precision [95% CI] | ROC-AUC [95% CI] | Calib Ratio | Learner Caught / Flagged (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20-30%** | 691 (18,076) | 508 (2.81%) | No | 11.3% [0.098, 0.129] | 37.0% [0.306, 0.434] | 9.2% [0.073, 0.114] | 0.749 [0.716, 0.782] | 0.91x | 109/283 (59.2%) |
| **60-70%** | 543 (14,390) | 391 (2.72%) | No | 10.2% [0.085, 0.121] | 38.4% [0.315, 0.456] | 10.2% [0.078, 0.126] | 0.791 [0.756, 0.825] | 0.79x | 84/194 (60.4%) |
| **0-10%** | 627 (16,431) | 428 (2.60%) | No | 11.8% [0.102, 0.137] | 40.7% [0.329, 0.479] | 9.0% [0.070, 0.112] | 0.759 [0.718, 0.794] | 0.98x | 89/275 (58.2%) |
| **40-50%** | 589 (15,438) | 382 (2.47%) | No | 8.7% [0.073, 0.101] | 30.4% [0.233, 0.374] | 8.6% [0.066, 0.111] | 0.770 [0.734, 0.806] | 0.83x | 67/195 (48.9%) |
| **50-60%** | 581 (15,474) | 367 (2.37%) | No | 11.0% [0.093, 0.126] | 35.4% [0.277, 0.432] | 7.6% [0.055, 0.101] | 0.785 [0.743, 0.821] | 1.01x | 75/227 (57.7%) |
| **10-20** | 633 (16,989) | 397 (2.34%) | No | 12.6% [0.108, 0.145] | 30.5% [0.236, 0.381] | 5.6% [0.041, 0.073] | 0.730 [0.694, 0.765] | 1.09x | 74/253 (52.9%) |
| **30-40%** | 672 (17,888) | 411 (2.30%) | No | 8.8% [0.075, 0.101] | 32.6% [0.260, 0.400] | 8.5% [0.063, 0.112] | 0.788 [0.755, 0.823] | 0.95x | 76/247 (51.4%) |
| **70-80%** | 532 (14,372) | 323 (2.25%) | No | 9.9% [0.082, 0.116] | 31.6% [0.237, 0.402] | 7.2% [0.052, 0.094] | 0.748 [0.703, 0.791] | 0.95x | 60/191 (50.8%) |
| **80-90%** | 560 (15,253) | 334 (2.19%) | No | 7.0% [0.057, 0.087] | 34.1% [0.260, 0.420] | 10.6% [0.078, 0.141] | 0.806 [0.768, 0.842] | 0.82x | 68/167 (58.1%) |
| **Unknown** | 233 (6,410) | 121 (1.89%) | ⚠️ Yes | 9.4% [0.072, 0.120] | 50.4% [0.356, 0.652] | 10.1% [0.059, 0.148] | 0.843 [0.771, 0.902] | 1.01x | 26/82 (63.4%) |
| **90-100%** | 509 (14,078) | 239 (1.70%) | No | 8.3% [0.068, 0.100] | 42.3% [0.329, 0.522] | 8.6% [0.063, 0.115] | 0.811 [0.769, 0.856] | 1.10x | 49/155 (55.7%) |

### Attribute: `region`

| Group | Learners (Rows) | Positives (Base Rate) | Unstable? | Selection Rate [95% CI] | Recall / TPR [95% CI] | Precision [95% CI] | ROC-AUC [95% CI] | Calib Ratio | Learner Caught / Flagged (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Yorkshire Region** | 391 (10,065) | 291 (2.89%) | No | 9.5% [0.077, 0.116] | 42.3% [0.342, 0.515] | 12.9% [0.091, 0.168] | 0.818 [0.773, 0.856] | 0.75x | 64/148 (60.4%) |
| **East Midlands Region** | 468 (12,151) | 337 (2.77%) | No | 9.0% [0.073, 0.108] | 29.4% [0.217, 0.368] | 9.1% [0.066, 0.121] | 0.728 [0.691, 0.767] | 0.76x | 56/158 (45.9%) |
| **Scotland** | 625 (16,714) | 456 (2.73%) | No | 11.4% [0.097, 0.130] | 35.7% [0.289, 0.424] | 8.6% [0.065, 0.109] | 0.766 [0.726, 0.795] | 0.91x | 104/276 (64.2%) |
| **Ireland** | 184 (4,773) | 126 (2.64%) | ⚠️ Yes | 8.0% [0.057, 0.105] | 27.8% [0.166, 0.396] | 9.1% [0.049, 0.151] | 0.762 [0.686, 0.833] | 0.75x | 20/61 (45.5%) |
| **North Western Region** | 555 (14,608) | 363 (2.48%) | No | 10.9% [0.091, 0.128] | 33.3% [0.247, 0.413] | 7.6% [0.055, 0.101] | 0.745 [0.707, 0.784] | 0.96x | 72/204 (53.3%) |
| **South East Region** | 403 (10,628) | 258 (2.43%) | No | 9.4% [0.075, 0.113] | 33.3% [0.250, 0.418] | 8.7% [0.060, 0.120] | 0.773 [0.726, 0.814] | 0.85x | 52/139 (57.1%) |
| **South West Region** | 448 (12,175) | 283 (2.32%) | No | 9.4% [0.075, 0.114] | 34.3% [0.253, 0.447] | 8.4% [0.060, 0.113] | 0.776 [0.739, 0.817] | 0.91x | 49/142 (50.5%) |
| **West Midlands Region** | 483 (12,910) | 300 (2.32%) | No | 9.3% [0.076, 0.110] | 39.0% [0.306, 0.479] | 9.7% [0.072, 0.126] | 0.787 [0.745, 0.823] | 0.94x | 61/187 (57.5%) |
| **Wales** | 396 (10,544) | 244 (2.31%) | No | 10.9% [0.089, 0.129] | 39.8% [0.302, 0.508] | 8.4% [0.060, 0.116] | 0.757 [0.705, 0.803] | 1.07x | 49/166 (57.6%) |
| **London Region** | 613 (16,529) | 366 (2.21%) | No | 11.9% [0.100, 0.137] | 33.3% [0.266, 0.405] | 6.2% [0.046, 0.080] | 0.766 [0.724, 0.805] | 1.10x | 76/235 (58.0%) |
| **North Region** | 350 (9,558) | 209 (2.19%) | No | 9.7% [0.078, 0.120] | 45.5% [0.349, 0.566] | 10.2% [0.074, 0.141] | 0.836 [0.793, 0.874] | 0.97x | 46/135 (62.2%) |
| **East Anglian Region** | 639 (17,214) | 365 (2.12%) | No | 9.3% [0.078, 0.112] | 33.2% [0.257, 0.405] | 7.6% [0.056, 0.097] | 0.789 [0.752, 0.822] | 1.00x | 71/223 (54.6%) |
| **South Region** | 615 (16,930) | 303 (1.79%) | No | 9.3% [0.079, 0.110] | 38.0% [0.295, 0.470] | 7.3% [0.054, 0.098] | 0.799 [0.762, 0.837] | 1.14x | 57/195 (50.9%) |

### Attribute: `highest_education`

| Group | Learners (Rows) | Positives (Base Rate) | Unstable? | Selection Rate [95% CI] | Recall / TPR [95% CI] | Precision [95% CI] | ROC-AUC [95% CI] | Calib Ratio | Learner Caught / Flagged (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lower Than A Level** | 2,428 (63,139) | 1,687 (2.67%) | No | 10.8% [0.099, 0.117] | 36.8% [0.329, 0.402] | 9.1% [0.079, 0.104] | 0.776 [0.758, 0.793] | 0.88x | 340/939 (56.2%) |
| **No Formal quals** | 71 (1,844) | 48 (2.60%) | ⚠️ Yes | 15.0% [0.099, 0.207] | 41.7% [0.188, 0.663] | 7.2% [0.027, 0.136] | 0.794 [0.697, 0.887] | 1.06x | 12/37 (66.7%) |
| **HE Qualification** | 897 (24,253) | 558 (2.30%) | No | 9.8% [0.084, 0.112] | 34.6% [0.291, 0.411] | 8.2% [0.065, 0.102] | 0.775 [0.746, 0.803] | 0.95x | 108/305 (54.0%) |
| **A Level or Equivalent** | 2,692 (73,188) | 1,564 (2.14%) | No | 9.3% [0.086, 0.102] | 34.8% [0.313, 0.383] | 8.0% [0.070, 0.090] | 0.775 [0.756, 0.794] | 1.00x | 309/961 (55.6%) |
| **Post Graduate Qualification** | 82 (2,375) | 44 (1.85%) | ⚠️ Yes | 8.8% [0.052, 0.135] | 29.5% [0.128, 0.517] | 6.2% [0.020, 0.132] | 0.719 [0.590, 0.855] | 0.95x | 8/27 (50.0%) |

### Attribute: `age_band`

| Group | Learners (Rows) | Positives (Base Rate) | Unstable? | Selection Rate [95% CI] | Recall / TPR [95% CI] | Precision [95% CI] | ROC-AUC [95% CI] | Calib Ratio | Learner Caught / Flagged (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0-35** | 4,348 (116,747) | 2,846 (2.44%) | No | 10.3% [0.098, 0.110] | 37.2% [0.346, 0.400] | 8.8% [0.080, 0.097] | 0.775 [0.761, 0.789] | 0.95x | 593/1712 (58.3%) |
| **35-55** | 1,792 (47,222) | 1,043 (2.21%) | No | 9.2% [0.083, 0.101] | 31.4% [0.272, 0.354] | 7.6% [0.064, 0.090] | 0.779 [0.758, 0.801] | 0.91x | 181/543 (48.5%) |
| **55<=** | 30 (830) | 12 (1.45%) | ⚠️ Yes | 16.1% [0.077, 0.256] | 41.7% [0.000, 1.000] | 3.7% [0.000, 0.122] | 0.751 [0.507, 0.960] | 1.95x | 3/14 (60.0%) |

### Attribute: `gender`

| Group | Learners (Rows) | Positives (Base Rate) | Unstable? | Selection Rate [95% CI] | Recall / TPR [95% CI] | Precision [95% CI] | ROC-AUC [95% CI] | Calib Ratio | Learner Caught / Flagged (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M** | 3,449 (93,247) | 2,483 (2.66%) | No | 11.8% [0.110, 0.125] | 39.2% [0.361, 0.422] | 8.9% [0.080, 0.098] | 0.774 [0.758, 0.789] | 0.93x | 522/1483 (58.5%) |
| **F** | 2,721 (71,552) | 1,418 (1.98%) | No | 7.7% [0.070, 0.084] | 29.4% [0.262, 0.333] | 7.6% [0.067, 0.086] | 0.773 [0.754, 0.792] | 0.97x | 255/786 (50.8%) |

### Attribute: `disability`

| Group | Learners (Rows) | Positives (Base Rate) | Unstable? | Selection Rate [95% CI] | Recall / TPR [95% CI] | Precision [95% CI] | ROC-AUC [95% CI] | Calib Ratio | Learner Caught / Flagged (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Y** | 611 (15,536) | 514 (3.31%) | No | 13.1% [0.113, 0.149] | 43.0% [0.361, 0.497] | 10.9% [0.089, 0.129] | 0.783 [0.754, 0.814] | 0.81x | 114/259 (61.3%) |
| **N** | 5,559 (149,263) | 3,387 (2.27%) | No | 9.7% [0.091, 0.102] | 34.5% [0.322, 0.370] | 8.1% [0.073, 0.088] | 0.773 [0.761, 0.785] | 0.96x | 663/2010 (54.8%) |

---

## 3. Demographic Proxy Analysis

To test whether the model inadvertently learns demographic proxies despite excluding demographic columns, we trained 5-fold learner-grouped cross-validated models predicting each demographic attribute using only the 38 sanctioned features and behavioral aggregates, and computed the Spearman rank correlation of predicted risk with IMD deprivation deciles.

| Demographic Attribute | Predictor Features | CV Balanced Acc | Dummy Baseline Acc | CV Macro-F1 | Dummy Macro-F1 | F1 Lift | Potential Proxy Risk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **gender** | 15 features | 77.4% | 50.0% | 0.781 | 0.365 | 2.14x | Elevated |
| **age_band** | 15 features | 40.6% | 33.3% | 0.288 | 0.277 | 1.04x | Minimal |
| **highest_education** | 15 features | 29.5% | 20.0% | 0.140 | 0.122 | 1.15x | Moderate |
| **imd_band** | 15 features | 12.6% | 8.9% | 0.083 | 0.029 | 2.85x | Elevated |
| **disability** | 15 features | 58.4% | 50.0% | 0.483 | 0.474 | 1.02x | Minimal |
| **region** | 15 features | 12.0% | 7.2% | 0.071 | 0.025 | 2.88x | Elevated |

**IMD Socio-Economic Deprivation Rank Correlation**:
- Spearman $\rho = -0.1041$ ($p = 9.12e-16$, evaluated on unique validation students).
- *Finding*: The correlation between predicted dropout risk and socio-economic deprivation decile is close to zero ($\rho = -0.1041$), indicating that the model is not systematically assigning higher risk scores solely to learners from lower-income postcodes.

---

## 4. Intersectional Disparities

Intersectional analysis examines compound vulnerability across multiple sensitive attributes simultaneously. Below are highlights for the primary intersection `gender × age_band` and `gender × highest_education`:

| Intersection Pair | Subgroup | Learners | Positives | Selection Rate | Recall (TPR) | Precision | Small-Sample Warning |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **gender_x_age_band** | F × 0-35 | 1,853 | 999 | 8.2% | 29.2% | 7.4% | None |
| **gender_x_age_band** | F × 35-55 | 865 | 417 | 6.6% | 29.7% | 8.2% | None |
| **gender_x_age_band** | F × 55<= | 3 | 2 | 18.1% | 50.0% | 4.8% | Learners < 300 (found 3); Positives < 30 (found 2) |
| **gender_x_age_band** | M × 0-35 | 2,495 | 1,847 | 11.8% | 41.5% | 9.5% | None |
| **gender_x_age_band** | M × 35-55 | 927 | 626 | 11.5% | 32.6% | 7.2% | None |
| **gender_x_age_band** | M × 55<= | 27 | 10 | 15.8% | 40.0% | 3.5% | Learners < 300 (found 27); Positives < 30 (found 10) |
| **gender_x_imd_band** | F × 0-10% | 323 | 170 | 8.4% | 27.6% | 6.6% | None |
| **gender_x_imd_band** | F × 10-20 | 307 | 161 | 9.2% | 21.7% | 4.7% | None |
| **gender_x_imd_band** | F × 20-30% | 348 | 245 | 9.6% | 33.1% | 9.7% | None |
| **gender_x_imd_band** | F × 30-40% | 301 | 156 | 7.2% | 28.8% | 7.8% | None |

---

## 5. In-Sample Mitigation Experiment (Trade-Off Analysis)

We tested whether adjusting decision thresholds per demographic group can mitigate observed disparities. A 50/50 learner split of `val` was used: group-specific thresholds were calibrated on the tuning half and evaluated out-of-sample on the remaining half. Scientific note: Disparities are **measured and reduced**, never 'fixed', and carry the caveats of in-sample presentation selection.

| Policy | Target Attribute | Selection Rate | Overall Recall | Overall Precision | Dem. Parity Diff | Equalized Odds Diff | Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Global Baseline (Top-10% Cohort)** | highest_education | 10.0% | 34.2% | 8.1% | 0.0488 | 0.4612 | Standard whole-cohort capacity cut |
| **Group-Specific Thresholds (Mitigated)** | highest_education | 11.2% | 36.8% | 7.8% | 0.0528 | 0.3035 | Tuned on 50% val split, evaluated out-of-sample on remaining 50% (measured and reduced, in-sample on val presentation) |

---

## 6. Critical Limitations and Methodological Caveats

1. **Single Validation Presentation (`2014B`)**:
   All audit numbers are derived from presentation `2014B`. Because demographic compositions, regional distributions, and unregistration patterns vary between presentations (e.g. February 'B' vs October 'J' intakes), these findings reflect a point estimate rather than universal course invariant behavior.
2. **Small Subgroup Sample Sizes**:
   Certain categories (e.g. `No Formal quals`, `Post Graduate Qualification`, age `55<=`) contain fewer than 300 learners or fewer than 30 positive dropout events. Confidence intervals for these groups are wide, and their metrics must be interpreted with caution.
3. **Mathematical Impossibility of Simultaneous Fairness Criteria**:
   Because baseline dropout rates differ systematically across demographic groups (e.g. learners without formal education withdraw at higher empirical rates than those with higher education), Kleinberg's theorem applies: **Equal True Positive Rate (Equal Opportunity) and Equal Precision (Predictive Parity) cannot simultaneously hold** unless the classifier achieves 100% accuracy.
4. **Audit Does Not Prove Absence of Bias**:
   Observational parity in selection rates or TPR does not guarantee an absence of disparate impact in downstream academic interventions. If human advisors respond differently to alerts for different demographic cohorts, the operational impact will diverge from model predictions.
