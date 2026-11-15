# Executive Technical Summary: Forecasting Comparison, Probability Calibration & Model Finalists

**Author:** Niranjan Patil | **Branch:** `niranjan-forecast-calibration` | **Date:** November 2026

## 1. Overview
This report covers (a) final-score forecasting and grade-band classification at weeks 4, 8 and 12, (b) out-of-fold calibration of the dropout models (k=3), and (c) the finalist list for the final test run. The test split (2014J) was never used.

## 2. Forecasting Comparison (Phase 5)

### 2.1 Setup
- **Cohort:** non-withdrawn, non-never-active learners. Train (2013B + 2013J): 9,618 learners (28,854 rows). Validation (2014B): 5,079 learners (15,237 rows = 5,079 x 3 snapshots). One learner set at all three snapshots, identical for every model. Train mean final_score 59.27, validation mean 57.39.
- **Inputs:** the 38 v3 features. Targets (`final_score`, `grade_band`) come from `labels.parquet` and are never used as features.
- **Tuning:** 15 Optuna trials per model, train only, 5-fold GroupKFold by `id_student`. Validation was never used for tuning or selection.
- **Evaluation:** 500 learner-clustered bootstrap resamples. Stochastic models report seed mean ± std over seeds 42/101/2024/7/999; CIs and paired deltas use seed-averaged predictions. Deterministic models (train mean, linear baselines, Ridge, multinomial LR, majority class) are marked as such. A paired delta is called "tied" when its 95% CI includes zero.
- **Regression models:** train mean, per-snapshot linear fit on `mean_score_submitted` (pooled and per-module), Ridge, Random Forest, LightGBM, XGBoost, CatBoost.
- **Classification models:** majority class, multinomial LR, Random Forest, LightGBM, XGBoost, CatBoost.
- **Hardware:** Intel Core CPU (4 logical cores), 8 GB RAM, Windows x64, no GPU. Timings are comparable only within this machine.

### 2.2 final_score regression (validation; headline = seed mean)

| Week | Model | MAE | RMSE | R² | Paired ΔR² vs LightGBM |
|---|---|---|---|---|---|
| 4 | Train mean | 24.09 | 29.53 | -0.004 | -0.474 (worse) |
| 4 | Per-snapshot linear (pooled) | 20.84 | 26.39 | 0.198 | -0.272 (worse) |
| 4 | Per-snapshot linear (per-module) | 19.91 | 25.83 | 0.232 | -0.238 (worse) |
| 4 | Ridge | 17.97 | 23.20 | 0.381 | -0.090 (worse) |
| 4 | Random Forest | 16.60 | 21.73 | 0.457 | -0.012 (worse) |
| 4 | LightGBM | 16.37 | 21.55 | 0.466 | reference |
| 4 | XGBoost | 16.50 | 21.57 | 0.464 | tied |
| 4 | CatBoost | 16.46 | 21.51 | 0.467 | tied |
| 8 | Train mean | 24.09 | 29.53 | -0.004 | -0.657 (worse) |
| 8 | Per-snapshot linear (pooled) | 20.63 | 26.28 | 0.205 | -0.448 (worse) |
| 8 | Per-snapshot linear (per-module) | 18.89 | 24.98 | 0.282 | -0.371 (worse) |
| 8 | Ridge | 15.04 | 19.65 | 0.556 | -0.098 (worse) |
| 8 | Random Forest | 12.84 | 17.43 | 0.651 | tied |
| 8 | LightGBM | 12.87 | 17.44 | 0.650 | reference |
| 8 | XGBoost | 12.91 | 17.40 | 0.651 | tied |
| 8 | CatBoost | 12.96 | 17.50 | 0.648 | tied |
| 12 | Train mean | 24.09 | 29.53 | -0.004 | -0.768 (worse) |
| 12 | Per-snapshot linear (pooled) | 19.66 | 25.40 | 0.257 | -0.507 (worse) |
| 12 | Per-snapshot linear (per-module) | 18.18 | 24.37 | 0.316 | -0.448 (worse) |
| 12 | Ridge | 14.56 | 19.55 | 0.560 | -0.204 (worse) |
| 12 | Random Forest | 10.46 | 14.39 | 0.762 | tied |
| 12 | LightGBM | 10.36 | 14.40 | 0.761 | reference |
| 12 | XGBoost | 10.61 | 14.41 | 0.761 | tied |
| 12 | CatBoost | 10.70 | 14.50 | 0.758 | tied |

**Findings**
1. **Forecasts improve as the course progresses.** For the tree models R² rises from about 0.46 (week 4) to about 0.65 (week 8) to about 0.76 (week 12), and MAE falls from about 16.4 to about 10.4 points. The maximum R² is 0.7616, far below the 0.95 leakage threshold.
2. **The four tree models are tied on R²** at weeks 8 and 12, and at week 4 only Random Forest is slightly worse (ΔR² -0.012). On MAE at week 12, RF (+0.18), XGBoost (+0.25) and CatBoost (+0.35) are slightly worse than LightGBM, but these gaps are small next to a ~10-point error.
3. **Trees clearly beat linear models.** Ridge reaches R² 0.38 to 0.56 and barely improves from week 8 to week 12 (0.556 to 0.560). The `mean_score_submitted`-only baselines (R² 0.20 to 0.32) are weaker still.
4. **Caution:** `final_score` is defined differently across modules (Section 5). Part of the tree advantage may come from module offsets learned through module-proxy features (for example `studied_credits`, `course_progress`), so the tree-vs-linear gap should be read as an upper bound on the value of the behavioural signal.

### 2.3 Per-module results (validation, R²)

| Module | n | Pooled linear W8 | Ridge W8 | LightGBM W8 | Pooled linear W12 | Ridge W12 | LightGBM W12 |
|---|---|---|---|---|---|---|---|
| BBB | 1,088 | 0.189 | 0.562 | 0.623 | 0.206 | 0.655 | 0.718 |
| CCC | 1,018 | 0.314 | 0.574 | 0.646 | 0.366 | 0.432 | 0.686 |
| DDD | 728 | 0.258 | 0.555 | 0.653 | 0.307 | 0.466 | 0.740 |
| EEE | 512 | 0.261 | 0.475 | 0.594 | 0.295 | 0.473 | 0.845 |
| FFF | 1,026 | 0.216 | 0.580 | 0.700 | 0.228 | 0.633 | 0.812 |
| GGG | 707 | -0.682 | 0.268 | 0.478 | -0.428 | 0.548 | 0.754 |

LightGBM has the highest R² in every module at both weeks. GGG is the weakest at week 8 and improves strongly by week 12. GGG's score is an unweighted mean of submitted TMAs (catalog weight 0), which explains the negative R² of the pooled linear baseline there. CCC at week 12 shows a drop for Ridge (0.432) relative to week 8 (0.574), while LightGBM improves, so linear models struggle with the exam-inclusive score.

### 2.4 grade_band classification (Fail / Pass / Distinction; validation)

| Week | Model | Macro-F1 | QWK | Recall F / P / D | ΔF1 vs LightGBM |
|---|---|---|---|---|---|
| 4 | Majority class | 0.224 | 0.000 | 0.00 / 1.00 / 0.00 | worse |
| 4 | Multinomial LR | 0.463 | 0.388 | 0.69 / 0.26 / 0.68 | tied |
| 4 | Random Forest | 0.520 | 0.406 | 0.68 / 0.50 / 0.43 | better |
| 4 | LightGBM | 0.464 | 0.374 | 0.56 / 0.78 / 0.07 | reference |
| 4 | XGBoost | 0.464 | 0.384 | 0.51 / 0.83 / 0.07 | tied |
| 4 | CatBoost | 0.513 | 0.412 | 0.73 / 0.43 / 0.50 | better |
| 8 | Majority class | 0.224 | 0.000 | 0.00 / 1.00 / 0.00 | worse |
| 8 | Multinomial LR | 0.547 | 0.489 | 0.70 / 0.40 / 0.73 | tied |
| 8 | Random Forest | 0.592 | 0.538 | 0.72 / 0.54 / 0.63 | better |
| 8 | LightGBM | 0.544 | 0.491 | 0.63 / 0.83 / 0.15 | reference |
| 8 | XGBoost | 0.547 | 0.489 | 0.58 / 0.87 / 0.16 | tied |
| 8 | CatBoost | 0.572 | 0.513 | 0.75 / 0.47 / 0.62 | better |
| 12 | Majority class | 0.224 | 0.000 | 0.00 / 1.00 / 0.00 | worse |
| 12 | Multinomial LR | 0.598 | 0.556 | 0.72 / 0.47 / 0.80 | better (small) |
| 12 | Random Forest | 0.642 | 0.602 | 0.78 / 0.58 / 0.67 | better |
| 12 | LightGBM | 0.576 | 0.547 | 0.69 / 0.83 / 0.17 | reference |
| 12 | XGBoost | 0.578 | 0.548 | 0.67 / 0.85 / 0.18 | tied |
| 12 | CatBoost | 0.626 | 0.584 | 0.79 / 0.54 / 0.68 | better |

**Findings**
1. Macro-F1 and QWK rise from week 4 to week 12 (RF: 0.52 to 0.64 and 0.41 to 0.60). The maximum QWK is 0.6017, far below 0.95.
2. RF and CatBoost have higher macro-F1 than LightGBM at every snapshot (week 12: +0.067 and +0.054, CIs above zero). QWK gains are smaller (+0.050 and +0.036).
3. **Confounder:** LightGBM and XGBoost mostly predict "Pass" (Distinction recall 0.07 to 0.18), while RF and CatBoost trade Pass recall for Distinction recall. This pattern looks like a class-weighting effect rather than a model-family effect, so "RF/CatBoost are better" holds only for the configurations used here.
4. Unweighted boosters find Distinction hardest (recall at most 0.18); the weighted models find Pass hardest (recall 0.43 to 0.58).

### 2.5 Cost (regression: training time per seed / inference per 1,000 rows)
Random Forest 21.6 s / 24.6 ms; LightGBM 1.6 s / 9.5 ms; XGBoost 1.9 s / 5.7 ms; CatBoost 3.2 s / 0.9 ms; Ridge under 0.01 s / 0.08 ms. Classification: RF 1.4 s / 11.5 ms; LightGBM 2.0 s / 15.9 ms; XGBoost 4.1 s / 13.8 ms; CatBoost 8.5 s / 6.8 ms. The same timing method (predict only) is used for all models.

## 3. Calibration Analysis (Dropout, k=3)

### 3.1 Problem and method
Baseline dropout models show three issues: (1) global under-prediction (training base rate 1.85% vs validation 2.37%); (2) over-confidence above p = 0.20 (predicted 25 to 35%, observed about 11%); (3) Random Forest was trained with class weight 5, so its raw probabilities are distorted (Brier 0.0305, ECE 0.0606, not comparable with the unweighted boosters).

Protocol: 5-fold GroupKFold by `id_student` inside the validation split (2014B, 164,799 learner-week rows, 6,474 students). Calibrators fitted on 4 folds and applied to the 5th, so all reported numbers are out-of-fold. Methods: Platt (sigmoid), isotonic, temperature scaling. A final calibrator per model and method was also fitted on all validation rows and saved to `models/calibration/` for the later test run; it was never scored on validation. Brier-change CIs use learner-clustered bootstrap.

### 3.2 Results (out-of-fold, validation)

| Model | Method | Brier before | Brier after | ΔBrier [95% CI] | ECE before | ECE after | CITL before | CITL after | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM v3 | Platt | 0.02253 | 0.02244 | -0.000085 [-0.00018, -0.00001] | 0.00497 | 0.00161 | -0.0048 | +0.00002 | 0.7756 | 0.0786 |
| LightGBM v3 | Isotonic | 0.02253 | 0.02245 | -0.000083 [-0.00018, +0.00001] | 0.00497 | 0.00084 | -0.0048 | +0.00001 | 0.7736 | 0.0784 |
| LightGBM v3 | Temperature | 0.02253 | 0.02255 | +0.000027 [+0.00001, +0.00005] | 0.00497 | 0.00413 | -0.0048 | -0.0041 | 0.7757 | 0.0786 |
| XGBoost | Platt | 0.02250 | 0.02244 | -0.000056 [-0.00014, +0.00002] | 0.00458 | 0.00180 | -0.0043 | +0.00001 | 0.7755 | 0.0775 |
| XGBoost | Isotonic | 0.02250 | 0.02245 | -0.000043 [-0.00013, +0.00003] | 0.00458 | 0.00051 | -0.0043 | +0.00001 | 0.7736 | 0.0772 |
| XGBoost | Temperature | 0.02250 | 0.02252 | +0.000021 [+0.00000, +0.00004] | 0.00458 | 0.00352 | -0.0043 | -0.0035 | 0.7756 | 0.0775 |
| CatBoost | Isotonic | 0.02250 | 0.02245 | -0.000055 [-0.00015, +0.00002] | 0.00413 | 0.00047 | -0.0039 | +0.00001 | 0.7760 | 0.0792 |
| CatBoost | Platt | 0.02250 | 0.02245 | -0.000050 [-0.00014, +0.00003] | 0.00413 | 0.00271 | -0.0039 | +0.00002 | 0.7781 | 0.0792 |
| CatBoost | Temperature | 0.02250 | 0.02251 | +0.000009 [-0.00001, +0.00003] | 0.00413 | 0.00348 | -0.0039 | -0.0035 | 0.7782 | 0.0792 |
| Random Forest | Platt | 0.03049 | 0.02245 | -0.008041 [-0.00845, -0.00762] | 0.06055 | 0.00153 | +0.0592 | +0.00002 | 0.7762 | 0.0781 |
| Random Forest | Isotonic | 0.03049 | 0.02246 | -0.008030 [-0.00843, -0.00760] | 0.06055 | 0.00071 | +0.0592 | +0.00001 | 0.7735 | 0.0779 |
| Random Forest | Temperature | 0.03049 | 0.02456 | -0.005929 [-0.00632, -0.00552] | 0.06055 | 0.01828 | +0.0592 | +0.0175 | 0.7763 | 0.0781 |
| Logistic Reg. (39 feat.) | Isotonic | 0.02276 | 0.02268 | -0.000081 [-0.00016, -0.00001] | 0.00670 | 0.00060 | -0.0065 | +0.00001 | 0.7350 | 0.0617 |

### 3.3 Interpretation
- For LightGBM, XGBoost and CatBoost the Brier gains are tiny (about 0.00005 to 0.00009) and several CIs include zero. These models were already near-calibrated, so calibration helps mainly ECE and the overall bias (CITL moves to about 0).
- **The real gain is Random Forest:** Brier 0.0305 to 0.0225 and ECE 0.061 to 0.0015. The "before" value reflects class weight 5.
- Temperature scaling does not fix the global bias (CITL stays near -0.004) and is not recommended. Platt and isotonic are indistinguishable in Brier.
- ROC-AUC and PR-AUC stay stable; isotonic shifts ROC-AUC by about 0.002 because it creates ties.
- The logistic-regression row uses the 39-feature variant (including `n_skipped_resources`), so it is not on the same inputs as the other models.
- Calibrators are fitted on 2014B. Because the base rate differs between presentations, the 2014J test run is the honest check of whether they transfer.

## 4. Finalist Candidates (for the final 2014J run)
Listed in `reports/niranjan/finalists.yaml`:
1. **LightGBM v3 + Platt** (`models/calibration/lgbm_v3_platt_calibrator.joblib`): Brier 0.02244, ECE 0.0016, ROC-AUC 0.7756.
2. **CatBoost + Isotonic** (`models/calibration/catboost_isotonic_calibrator.joblib`): ECE 0.0005, ROC-AUC 0.7760, PR-AUC 0.0792.
3. **Random Forest + Platt** (`models/calibration/rf_platt_calibrator.joblib`): different architecture (bagging); probabilities repaired by calibration.

The boosters are statistically tied, so the choice between 1 and 2 rests on calibration, cost and diversity, not on a measured winner. Their differences in Brier (0.02244 vs 0.02245) are within noise.

## 5. Limitations
1. Single validation presentation (2014B); calibrators are fitted on it. The 2014J test run is the honest check.
2. `final_score` is not comparable across modules: CCC and DDD include the exam, GGG is an unweighted mean of submitted TMAs, and other modules are continuous-assessment only. Per-module results are reported for this reason.
3. The forecast population (non-withdrawn) is selected using the outcome, so forecasts are conditional on the learner staying.
4. Multinomial LR did not fully converge at max_iter=200. Ridge alpha (0.010) and LR C (0.006) landed near the lower edge of the search range.
5. Grade-band comparisons are confounded by class weighting (Section 2.4), so family-level conclusions are not supported.
6. Tuning budget was 15 Optuna trials per model. 
7. Module-proxy features (`studied_credits`, `course_progress`) may inflate tree-vs-linear gaps in forecasting.
8. Only the 39-feature LR was calibrated; it uses different inputs from the other calibrated models.
9. The test split (2014J) was untouched in all tuning, calibration and comparison.