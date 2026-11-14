# Phase 3: Model Zoo Comparison for Early Dropout Prediction (OULAD)

## 1. Executive Summary & Protocol Overview
This phase benchmarks multiple model families on strictly identical footing against the official Phase 2 champion baseline (**LightGBM v3**, 38 features, train-only tuned).
- **Target:** `label_dropout_k3` (k=3, min_week=2, drop_never_active=True).
- **Train split:** 330,940 weekly observations across 11,835 learners (positive rate 1.850%).
- **Val split (2014B):** 164,799 weekly observations across 6,474 learners (positive rate 2.367%).
- **Test split (2014J):** **STRICTLY UNTOUCHED (`allow_test=True` forbidden)**.
- **Protocol:**
  - 30-trial Optuna inner CV (5-fold `GroupKFold` by `id_student` on Train only).
  - Metric: Inner-CV PR-AUC (`average_precision_score`), defined as the **mean over folds** (same definition for all models).
  - Final tree / iteration count: median of optimal early-stopping iterations across the 5 inner folds.
  - Multi-seed training: 5 fixed seeds `[42, 101, 2024, 7, 999]`.
  - Headline metric: 5-seed mean predictions evaluated on Val (2014B).
  - Paired comparison: Clustered bootstrap paired difference vs the **LightGBM v3 SEED MEAN** (500 resamples, clustered strictly by `id_student`, seed=42). Models whose 95% CI contains zero are marked as statistically tied (`*`).
- **Omissions:** A tabular MLP and ExtraTrees were omitted for time. ExtraTrees is the same bagged-tree family as Random Forest, and neural models are covered in the separate sequence-model phase.
- **Final Model Set:** Majority prior, recency-only LR, 39-feature LR (unweighted), LightGBM v3, XGBoost, CatBoost, Random Forest.

---

## 2. Checkpoint B: Full Single-Model Comparison Table

All models are evaluated on the exact same validation split (2014B, $N=164,799$, base rate 2.37%). 
The table is ranked by **seed-mean Val PR-AUC** (descending). Paired deltas are computed against the **LightGBM v3 5-seed mean predictions**.

| Rank | Model | 5-seed PR Mean ± Std | 5-seed ROC Mean ± Std | Val PR-AUC [95% CI] | PR Multiple | Val ROC-AUC [95% CI] | Brier Score | Paired ΔPR vs v3 Seed Mean [95% CI] | Paired ΔROC vs v3 Seed Mean [95% CI] | Tied? | Train Time (s/seed) | Latency (ms/1k) | Model Size |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **LightGBM v3 ★ (Official Champion)** | 0.0783 ± 0.0003 | 0.7751 ± 0.0007 | 0.0786 [0.0723, 0.0862] | 3.32x | 0.7757 [0.7643, 0.7869] | 0.0225 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | **Reference** | 3.3s | 13.51 ms | 0.43 MB |
| 2 | **CatBoost** | 0.0781 ± 0.0005 | 0.7773 ± 0.0006 | 0.0786 [0.0722, 0.0864] | 3.32x | 0.7782 [0.7676, 0.7895] | 0.0225 | -0.0000 [-0.0027, +0.0022] | +0.0025 [-0.0005, +0.0055] | **Yes (*)** | 9.4s | 1.52 ms | 0.85 MB |
| 3 | **Random Forest** | 0.0776 ± 0.0002 | 0.7759 ± 0.0003 | 0.0778 [0.0716, 0.0848] | 3.28x | 0.7763 [0.7652, 0.7879] | 0.0305 | -0.0008 [-0.0039, +0.0022] | +0.0006 [-0.0036, +0.0051] | **Yes (*)** | 123.7s | 54.22 ms | 28.43 MB |
| 4 | **XGBoost** | 0.0776 ± 0.0010 | 0.7747 ± 0.0007 | 0.0781 [0.0719, 0.0858] | 3.30x | 0.7756 [0.7647, 0.7867] | 0.0225 | -0.0005 [-0.0020, +0.0011] | -0.0001 [-0.0016, +0.0014] | **Yes (*)** | 5.8s | 0.25 ms | 0.36 MB |
| 5 | **39-Feature LR (unweighted)** | 0.0617 ± 0.0000 | 0.7370 ± 0.0000 | 0.0617 [0.0566, 0.0680] | 2.61x | 0.7370 [0.7247, 0.7492] | 0.0228 | -0.0169 [-0.0229, -0.0110] | -0.0387 [-0.0466, -0.0296] | No | 1.2s | 0.08 ms | 0.01 MB |
| 6 | **Recency-Only LR** | 0.0334 ± 0.0000 | 0.6272 ± 0.0000 | 0.0334 [0.0310, 0.0362] | 1.41x | 0.6272 [0.6140, 0.6401] | 0.0231 | -0.0453 [-0.0525, -0.0380] | -0.1484 [-0.1628, -0.1312] | No | 0.2s | 0.02 ms | 0.001 MB |
| 7 | **Majority Prior** | 0.0237 ± 0.0000 | 0.5000 ± 0.0000 | 0.0237 [0.0225, 0.0248] | 1.00x | 0.5000 [0.5000, 0.5000] | 0.0231 | -0.0549 [-0.0614, -0.0481] | -0.2757 [-0.2862, -0.2634] | No | 0.0s | 0.001 ms | 0.00 MB |

> **Notes on Table:**
> - `*` marks models whose 95% paired bootstrap CI for $\Delta\text{PR}$ includes zero, establishing statistical equivalence with LightGBM v3. **Every single tree model (CatBoost, Random Forest, XGBoost) is statistically tied with LightGBM v3 on both PR-AUC and ROC-AUC.**
> - CIs are 95% bootstrap confidence intervals (500 resamples clustered by `id_student`, seed=42) evaluated on the **seed-averaged predicted probabilities**.
> - Random Forest Brier score (0.0305) reflects probability distortion from `use_class_weight=True` (effective positive class weight ~5.0). Trees without class weighting maintain well-calibrated uncalibrated probabilities (Brier = 0.0225).

---

## 3. LightGBM v3 5-Seed Protocol Reference
LightGBM v3 was re-evaluated under the exact 5-seed protocol (`seeds = [42, 101, 2024, 7, 999]`) using the official frozen train-only hyperparameters:
- Seed 42: Val PR = 0.0786, Val ROC = 0.7757, Fit time = 3.21s
- Seed 101: Val PR = 0.0786, Val ROC = 0.7750, Fit time = 3.55s
- Seed 2024: Val PR = 0.0784, Val ROC = 0.7760, Fit time = 3.24s
- Seed 7: Val PR = 0.0778, Val ROC = 0.7740, Fit time = 3.41s
- Seed 999: Val PR = 0.0781, Val ROC = 0.7747, Fit time = 3.12s
- **5-Seed Summary:** PR-AUC = $0.0783 \pm 0.0003$, ROC-AUC = $0.7751 \pm 0.0007$.
- **Seed-Averaged Headline:** Val PR-AUC = **0.0786**, Val ROC-AUC = **0.7757**. Used as the baseline reference for all paired bootstrap comparisons.

---

## 4. Inner-CV vs Validation Gap Analysis
Inner-CV PR-AUC is defined consistently for all models as the **mean over the 5 folds** of learner-grouped CV on Train (`learner_group_cv`, GroupKFold by `id_student`, objective = `average_precision_score`):
- **LightGBM v3:** Inner-CV PR = 0.1038 $\to$ Val PR = 0.0786 ($\Delta = -0.0252$) [Reference gap]
- **XGBoost:** Inner-CV PR = 0.1033 $\to$ Val PR = 0.0781 ($\Delta = -0.0252$, deviation from reference = **0.0000**)
- **CatBoost:** Inner-CV PR = 0.1027 $\to$ Val PR = 0.0786 ($\Delta = -0.0242$, deviation from reference = **+0.0010**)
- **Random Forest:** Inner-CV PR = 0.1010 $\to$ Val PR = 0.0778 ($\Delta = -0.0232$, deviation from reference = **+0.0020**)

*Sanity Trigger Verdict:* **Zero models trigger the anomaly alert.** All four tree model gaps fall within a tight band ($-0.0232$ to $-0.0252$). This uniform drop of ~0.025 is driven entirely by the natural cross-presentation cohort shift between Train (2013B + 2013J) and Val (2014B), confirming that no model suffered idiosyncratic overfitting.

---

## 5. Random Forest Specifics & Subsample Tuning Audit
- **Chosen Hyperparameters (Optuna Trial #21):**
  - `n_estimators`: 300
  - `max_depth`: 15
  - `min_samples_leaf`: 129
  - `max_features`: 0.3716
  - `use_class_weight`: True (effective class weight: `{0: 1.0, 1: 5.0}`)
- **Tuning Protocol:** 30 trials using a **25% learner-grouped subsample** (`get_learner_subsample_mask`, zero student overlap across folds). Total tuning time: **1,629.0s (27.1 minutes)**.
- **Full Train Refit:** The final model was trained on the 100% full Train dataset across all 5 seeds (mean train time: 123.7s/seed).
- **Comparison Against Default Configuration (Full Train Refit):**
  - *Tuned RF (Optuna on 25% subsample):* Seed-mean PR = $0.0776 \pm 0.0002$ | Seed-mean ROC = $0.7759 \pm 0.0003$ | Headline PR = 0.0778
  - *Default RF (`min_samples_leaf=100`, `max_depth=12`, `n_est=200`, `max_feat=0.5`):* Seed-mean PR = $0.0778 \pm 0.0004$ | Seed-mean ROC = $0.7724 \pm 0.0009$ | Headline PR = 0.0781
  - *Tuning Impact (Tuned minus Default):* $\Delta\text{PR} = -0.0001$ ($\Delta\text{ROC} = +0.0035$).
  - *Finding:* Tuning on a 25% subsample yielded zero PR-AUC advantage over a sensible default configuration, confirming that Random Forest performance is primarily dictated by the underlying feature representation rather than tree hyperparameter fine-tuning.

---

## 6. Input Sets & Preprocessing Audit
| Model Family | Feature Count | Input Columns & Handling |
| :--- | :---: | :--- |
| **Linear Models (LR)** | 39 | 39 continuous features from `config/features.yaml` (including `n_skipped_resources`). Train-median imputed, standardized via `StandardScaler`. **No missingness indicators.** |
| **Gradient Boosting (LGBM, XGB, CatBoost)** | 38 | 38 sanctioned v3 features (the 39 minus `n_skipped_resources`). Raw numerical inputs. **Missing values handled natively** via optimal directional splitting. |
| **Random Forest (RF)** | 43 | 38 sanctioned v3 features (train-median imputed and scaled) **PLUS 5 missingness indicators** for assessment features: <br>1. `mean_score_submitted_is_missing`<br>2. `last_score_is_missing`<br>3. `score_slope_is_missing`<br>4. `mean_lateness_is_missing`<br>5. `banked_score_mean_is_missing`<br>Missing indicators are binary $\{0, 1\}$ and left unscaled. |

*Key Difference:* LR imputes medians directly, inadvertently treating "no submission yet" as average performance. RF uses missingness indicators to isolate unsubmitted coursework as a discrete risk state. Tree models split on missingness natively without imputation artifacts.

---

## 7. Permutation Feature Importance & Group Analysis
Permutation feature importance was evaluated on a single **fixed validation subsample of 5,000 rows** (`seed=42`, 3 repeats, metric = drop in PR-AUC). RF importances were collapsed back to the 38 shared raw features for direct alignment.

### Importance by Feature Group (Sum of PR-AUC Drops)
| Feature Group | LightGBM v3 | XGBoost | CatBoost | Random Forest | Group Characterization |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Assessment** | **0.02695** | **0.05272** | **0.05124** | **0.03359** | **Overwhelmingly dominates in all 4 models** |
| **Static** | 0.00049 | 0.02218 | 0.01547 | 0.00515 | Moderate impact (registration timing, credits) |
| **Behavior** | 0.00087 | 0.01717 | 0.01035 | 0.00641 | Activity recency and frequency |
| **Curriculum** | -0.00400 | 0.01742 | -0.00173 | 0.00452 | Resource visited percentage and pacing |
| **Temporal** | 0.00276 | 0.00347 | 0.00514 | 0.00375 | Rolling trends and click slopes |

### Consensus Top-10 Features (Rank Averaged Across All 4 Models)
1. `weighted_score_to_date` (Avg Normalized Rank = **1.000** — #1 feature across **every single model**)
2. `days_since_last_activity` (0.875)
3. `studied_credits` (0.868)
4. `mean_score_submitted` (0.855)
5. `precourse_clicks` (0.849)
6. `mean_lateness` (0.849)
7. `last_score` (0.809)
8. `registration_lead_time` (0.796)
9. `weekly_clicks` (0.776)
10. `prerequisite_gap` (0.776)

> **Scientific Disclosure on Proxy Signals:**
> As established in the Phase 2 audit, `studied_credits` and `course_progress` act partially as **module and calendar proxies**:
> - `studied_credits` strongly correlates with course module enrollment patterns (e.g., 60 vs 120 credit paths), embedding module baseline difficulty.
> - `course_progress` captures academic calendar progression rather than pure individualized engagement.
> Despite this, assessment features (`weighted_score_to_date`, `mean_score_submitted`, `mean_lateness`) remain the primary driver of actionable predictive power across all model architectures.

---

## 8. Prediction Agreement & Interchangeability
Pairwise prediction agreement was computed on the full validation cohort ($N=164,799$):

### Spearman Rank Correlation Matrix
| | LightGBM v3 | XGBoost | CatBoost | Random Forest |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM v3** | 1.0000 | **0.9896** | **0.9647** | 0.9116 |
| **XGBoost** | **0.9896** | 1.0000 | **0.9651** | 0.9217 |
| **CatBoost** | **0.9647** | **0.9651** | 1.0000 | 0.9207 |
| **Random Forest** | 0.9116 | 0.9217 | 0.9207 | 1.0000 |

### Top-10% Flagged Learner Overlap ($k = 16,479$ rows)
- LightGBM v3 & XGBoost: **90.25%** (14,873 learners in common)
- LightGBM v3 & CatBoost: **84.62%** (13,944 learners in common)
- XGBoost & CatBoost: **84.90%** (13,991 learners in common)
- CatBoost & Random Forest: **85.01%** (14,009 learners in common)
- XGBoost & Random Forest: **83.41%** (13,745 learners in common)
- LightGBM v3 & Random Forest: **81.53%** (13,436 learners in common)
- **All 4 Models Unanimous Overlap:** **73.26%** (12,073 learners flagged by every model)

*Scientific Finding:* All gradient boosting models correlate at $\rho > 0.96$ with 85–90% top-decile overlap. **LightGBM, XGBoost, and CatBoost are functionally interchangeable.** Differences between them are well within sampling error.

---

## 9. Reproducibility & Sanity Checks
- **Compute Hardware:** Intel Core i7 10-core / 16-thread CPU, 15.65 GB RAM, NVIDIA GeForce RTX 4050 Laptop GPU (Windows 11).
- **Optuna Studies:** 30 trials completed and persisted for XGBoost (`xgb_optuna_study.joblib`), CatBoost (`catboost_optuna_study.joblib`), and Random Forest (`rf_optuna_study.joblib`).
- **Determinism Audit:**
  - XGBoost: Identical seed fit yields bit-exact probabilities (max diff = $0.0$).
  - CatBoost: Setting `eval_metric='PRAUC'` ensures bit-exact reproducibility across seeds (max diff = $0.0$).
  - Random Forest: Fixed seed yields bit-exact predictions (max diff = $0.0$).
- **Sanity Threshold Verification:**
  - *Rule 1 (ROC-AUC $\le 0.80$):* Highest observed is CatBoost at $0.7782$. **PASSED.**
  - *Rule 2 (Max Feature Importance $\le 50\%$):* Highest permutation PR drop is `weighted_score_to_date` on XGBoost at $0.0328$ (~31% of total model drop). **PASSED.**
  - *Rule 3 (Divergent Same-Seed Predictions):* Max divergence across all verified models is $0.0$. **PASSED.**
- **Test Suite Health:** Full automated pytest suite executed: **44/44 passed in 14.99s**.
