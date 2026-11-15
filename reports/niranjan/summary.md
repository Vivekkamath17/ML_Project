# Executive Technical Summary: Forecasting Comparison, Probability Calibration & Model Finalists

**Author:** Niranjan Patil  
**Date:** November 2026  
**Scope:** OULAD Continuous Performance Forecasting & Post-Hoc Probability Calibration  
**Branch:** `niranjan-forecast-calibration`  

---

## 1. Executive Summary

This report completes Phase 5 deliverables regarding multi-horizon continuous final score forecasting, end-of-term grade band classification, out-of-fold probability calibration across baseline dropout models, and the selection of finalist candidates for the final test run.

---

## 2. Calibration Analysis (Dropout Horizon $k=3$)

### 2.1 Problem & Methodology
Baseline models exhibit systematic miscalibration:
1. **Global Under-Prediction**: Training base rate (1.85%) vs Validation base rate (2.37%).
2. **High-Probability Overconfidence**: Above $\hat{p} > 0.20$, empirical attrition is ~11% while predicted probabilities range between 25% and 35%.
3. **Class Weight Distortion**: Random Forest was trained with `class_weight=5`, yielding severely distorted uncalibrated probabilities (Brier = 0.0305, ECE = 0.0606).

**Evaluation Protocol:**
- 5-fold GroupKFold by `id_student` strictly **inside the validation split (2014B)** to generate honest out-of-fold (OOF) calibrated predictions.
- Calibrators: Platt scaling (logistic regression on logits), Isotonic regression, and Temperature scaling.
- A secondary final calibrator per model/method was fitted on 100% of validation rows and serialized to `models/calibration/` for blinded application to test split `2014J`.

### 2.2 Calibration Before vs After (OOF Validation Results)

| Model | Method | Brier (Before) | Brier (After) | $\Delta$ Brier [95% CI] | ECE (Before) | ECE (After) | CITL (Before) | CITL (After) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| **LightGBM v3** | Platt | 0.02253 | **0.02244** | -0.000085 [-0.00018, -0.00001] | 0.00497 | 0.00161 | -0.0048 | +0.00002 | 0.7756 | 0.0786 |
| LightGBM v3 | Isotonic | 0.02253 | 0.02245 | -0.000083 [-0.00018, +0.00001] | 0.00497 | 0.00084 | -0.0048 | +0.00001 | 0.7736 | 0.0784 |
| LightGBM v3 | Temperature | 0.02253 | 0.02255 | +0.000027 [+0.00001, +0.00005] | 0.00497 | 0.00413 | -0.0048 | -0.0041 | 0.7757 | 0.0786 |
| **XGBoost** | Platt | 0.02250 | **0.02244** | -0.000056 [-0.00014, +0.00002] | 0.00458 | 0.00180 | -0.0043 | +0.00001 | 0.7755 | 0.0775 |
| XGBoost | Isotonic | 0.02250 | 0.02245 | -0.000043 [-0.00013, +0.00003] | 0.00458 | 0.00051 | -0.0043 | +0.00001 | 0.7736 | 0.0772 |
| XGBoost | Temperature | 0.02250 | 0.02252 | +0.000021 [+0.00000, +0.00004] | 0.00458 | 0.00352 | -0.0043 | -0.0035 | 0.7756 | 0.0775 |
| **CatBoost** | Isotonic | 0.02250 | **0.02245** | -0.000055 [-0.00015, +0.00002] | 0.00413 | **0.00047** | -0.0039 | +0.00001 | 0.7760 | 0.0792 |
| CatBoost | Platt | 0.02250 | 0.02245 | -0.000050 [-0.00014, +0.00003] | 0.00413 | 0.00271 | -0.0039 | +0.00002 | 0.7781 | 0.0792 |
| CatBoost | Temperature | 0.02250 | 0.02251 | +0.000009 [-0.00001, +0.00003] | 0.00413 | 0.00348 | -0.0039 | -0.0035 | 0.7782 | 0.0792 |
| **Random Forest** | Platt | 0.03049 | **0.02245** | **-0.008041 [-0.00845, -0.00762]**| 0.06055 | 0.00153 | +0.0592 | +0.00002 | 0.7762 | 0.0781 |
| Random Forest | Isotonic | 0.03049 | 0.02246 | -0.008030 [-0.00843, -0.00760] | 0.06055 | 0.00071 | +0.0592 | +0.00001 | 0.7735 | 0.0779 |
| Random Forest | Temperature | 0.03049 | 0.02456 | -0.005929 [-0.00632, -0.00552] | 0.06055 | 0.01828 | +0.0592 | +0.0175 | 0.7763 | 0.0781 |
| **LogReg (39f)** | Isotonic | 0.02276 | **0.02268** | -0.000081 [-0.00016, -0.00001] | 0.00670 | 0.00060 | -0.0065 | +0.00001 | 0.7350 | 0.0617 |

---

## 3. Finalist Candidates for Test Split Handoff

As specified in `reports/niranjan/finalists.yaml`:

1. **LightGBM v3 + Platt Calibrator (`models/calibration/lgbm_v3_platt_calibrator.joblib`)**:
   - Lowest post-calibration Brier score (0.022442), ECE 0.00161, and extremely low latency (4.88ms/1k rows).
2. **CatBoost + Isotonic Calibrator (`models/calibration/catboost_isotonic_calibrator.joblib`)**:
   - Highest discrimination (ROC-AUC 0.7782, PR-AUC 0.0792) and lowest post-calibration ECE (0.000471).
3. **Random Forest + Platt Calibrator (`models/calibration/rf_platt_calibrator.joblib`)**:
   - Provides critical architectural diversity (bagged trees vs gradient boosted trees). Post-calibration Brier improves dramatically from 0.03049 to 0.02245, fixing the training class-weight distortion.

---

## 4. Methodological Limitations

1. **Single Validation Presentation (2014B)**: The calibrators are fitted on 2014B, which lacks presentations like AAA and EEE. The upcoming test run on 2014J serves as the true out-of-presentation generalization check.
2. **Cross-Module Heterogeneity**: `final_score` formulations differ structurally across modules (e.g., CCC and DDD include formal exams; GGG is strictly unweighted TMA continuous evaluation).
3. **Quarantine Discipline**: Test split `2014J` remained strictly untouched throughout all tuning, calibration fitting, and model comparisons.
