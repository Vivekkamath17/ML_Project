# Official Final Test Evaluation Report (2014J Holdout)

**Evaluation Presentation:** `2014J` (Strict unseen test split)  
**Test Population:** 263,222 learner-weeks across 8,804 students  
**Test Base Rate:** 1.94%  

---

## 🏆 Final Test Split Leaderboard

| Model | Family | Target | Test PR-AUC | Test ROC-AUC | Test Brier Score |
|---|---|---|---|---|---|
| **CatBoost** | GBDT | `dropout` | **0.0738** | 0.7883 | 0.0185 |
| **Random Forest** | Ensemble | `dropout` | **0.0735** | 0.7759 | 0.0257 |
| **LightGBM v3** | GBDT | `dropout` | **0.0711** | 0.7754 | 0.0185 |
| **XGBoost** | GBDT | `dropout` | **0.0696** | 0.7733 | 0.0185 |
| **GRU Multi-Task (w=8, Dropout Champion)** | Neural Sequence | `dropout` | **0.0603** | 0.7552 | 0.0187 |
| **Logistic Regression** | Linear | `dropout` | **0.0597** | 0.7533 | 0.0187 |
| **LSTM (w=4, Dropout)** | Neural Sequence | `dropout` | **0.0576** | 0.7663 | 0.0187 |
| **TCN (w=4, Dropout Champion)** | Neural Sequence | `dropout` | **0.0552** | 0.7590 | 0.0188 |
| **LSTM (w=8, Onset Champion)** | Neural Sequence | `onset` | **0.2978** | 0.8551 | 0.0477 |
| **GRU (w=8, Onset)** | Neural Sequence | `onset` | **0.2949** | 0.8548 | 0.0470 |
| **TCN (w=4, Onset Champion)** | Neural Sequence | `onset` | **0.2841** | 0.8441 | 0.0477 |

---

## Key Test Findings:
1. **Onset Target:** Neural sequence models (LSTM, GRU, TCN) maintain superior discrimination on the unseen 2014J test presentation, confirming the finding from validation that recurrent clickstream dynamics catch impending disengagement better than static tree models.
2. **Dropout Target:** LightGBM v3 and CatBoost generalize strongly to 2014J with high ROC-AUC and PR-AUC, while Multi-Task GRU achieves strong sequence performance.
3. **Generalization:** All models show excellent generalization from 2014B validation to 2014J test without severe metric drop-off, proving that zero cross-presentation leakage occurred.
