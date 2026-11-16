# Phase 4 Handoff Document (`reports/phase4/HANDOFF.md`)

**Date:** November 16, 2026  
**Phase:** Phase 4 — Neural Sequence Models for Student Trajectory Modeling  
**Status:** COMPLETE & VERIFIED  

---

## 1. Catalog of Deliverables

All deliverables have been generated, validated, and placed in their standardized repository locations:

### A. Predictions (`predictions/seq/`)
Every prediction file adheres strictly to the Part A schema contract:
- **Binary Classification Schema:** `learner_key, week, split, model, k, y_true, y_score`
- **Regression Schema (Final Score):** `learner_key, week, split, model, y_true, y_score`
- All 20 files verified with zero NaN values, strict $[0.0, 1.0]$ bounds, composite key uniqueness, and zero test split rows.

| File Name | Horizon / Target | Model Architecture | Rows | Val PR-AUC |
|---|---|---|---|---|
| `lstm_seq_only_w8_onset.csv` | $k=3$ Onset | LSTM ($W=8$, Seq-Only) | 164,799 | **0.3202** (Beats LGBM) |
| `gru_seq_only_w8_onset.csv` | $k=3$ Onset | GRU ($W=8$, Seq-Only) | 164,799 | **0.3181** (Beats LGBM) |
| `tcn_seq_only_w4_onset.csv` | $k=3$ Onset | TCN ($W=4$, Seq-Only) | 164,799 | **0.3156** (Beats LGBM) |
| `gru_seq_only_multi_w8_dropout.csv`| $k=3$ Dropout | GRU ($W=8$, Multi-Task) | 164,799 | **0.0711** |
| `gru_hybrid_multi_w8_dropout.csv` | $k=3$ Dropout | GRU ($W=8$, Hybrid Multi)| 164,799 | 0.0686 |
| `lstm_seq_only_w4_dropout.csv` | $k=3$ Dropout | LSTM ($W=4$, Seq-Only) | 164,799 | 0.0683 |
| `tcn_seq_only_w4_dropout.csv` | $k=3$ Dropout | TCN ($W=4$, Seq-Only) | 164,799 | 0.0680 |
| `gru_seq_only_w4_dropout.csv` | $k=3$ Dropout | GRU ($W=4$, Seq-Only) | 164,799 | 0.0676 |
| `lstm_seq_only_w8_dropout.csv` | $k=3$ Dropout | LSTM ($W=8$, Seq-Only) | 164,799 | 0.0676 |
| `tcn_seq_only_w8_dropout.csv` | $k=3$ Dropout | TCN ($W=8$, Seq-Only) | 164,799 | 0.0670 |
| `gru_hybrid_w8_dropout.csv` | $k=3$ Dropout | GRU ($W=8$, Hybrid) | 164,799 | 0.0665 |
| `gru_seq_only_w8_dropout.csv` | $k=3$ Dropout | GRU ($W=8$, Seq-Only) | 164,799 | 0.0656 |
| `gru_seq_only_multi_w8_finalscore.csv`| Continuous Score | GRU ($W=8$, Multi-Task) | 164,799 | Regression $R^2$/MSE |
| `gru_hybrid_multi_w8_finalscore.csv` | Continuous Score | GRU ($W=8$, Hybrid Multi)| 164,799 | Regression $R^2$/MSE |

### B. Summary Tables & Technical Reports (`reports/phase4/`)
- [`reports/phase4/sequence_vs_lgbm.md`](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/ML_Project/reports/phase4/sequence_vs_lgbm.md): Comprehensive report addressing all 8 mandated research questions.
- [`reports/phase4/by_model_seq.csv`](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/ML_Project/reports/phase4/by_model_seq.csv): Master results table containing 18 evaluated configurations with 5-seed metrics and paired bootstrap CIs vs LightGBM v3.
- [`reports/phase4/feature_map.md`](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/ML_Project/reports/phase4/feature_map.md): Full feature catalog of 38 features with scaling and temporal handling.
- [`reports/phase4/STATUS.md`](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/ML_Project/reports/phase4/STATUS.md): Real-time command log and checkpoint history.
- Demographic Cohort Tables:
  - `cohort_by_stem.csv`
  - `cohort_by_prev_attempts.csv`
  - `cohort_by_imd.csv`
  - `cohort_by_gender.csv`
  - `cohort_by_disability.csv`
  - `cohort_by_week.csv`
  - `cohort_overall.csv`

### C. Figures (`reports/phase4/figures/`)
- [`reports/phase4/figures/metric_vs_model_ci_dropout.png`](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/ML_Project/reports/phase4/figures/metric_vs_model_ci_dropout.png): PR-AUC with 95% bootstrap CIs comparing all models on Dropout.
- [`reports/phase4/figures/metric_vs_model_ci_onset.png`](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/ML_Project/reports/phase4/figures/metric_vs_model_ci_onset.png): PR-AUC and ROC-AUC comparisons on Onset target showing sequence models outperforming LightGBM.
- [`reports/phase4/figures/metric_by_snapshot_week_dropout.png`](file:///c:/Users/Sneha%20Sarade/OneDrive/Desktop/Github/ML_Project/reports/phase4/figures/metric_by_snapshot_week_dropout.png): Trajectory of PR-AUC and ROC-AUC across course weeks.

### D. Source Code (`src/seq/`)
- `builder.py`: Vectorized windowed sequence dataset builder with train-only `SequenceScaler`.
- `dataset.py`: PyTorch Dataset & DataLoader with pre-tensorized memory optimization.
- `models/`: PyTorch modules (`gru.py`, `lstm.py`, `tcn.py`, `transformer.py`, `heads.py`).
- `train.py`: Training loop with early stopping, gradient clipping, and plateau scheduler.
- `tune.py`: Optuna hyperparameter tuner with `MedianPruner` and learner-grouped CV.
- `predict_io.py`: Standardized serializer for classification and regression outputs.
- `evaluate.py`: Metrics harness integration and cluster-bootstrap CI calculator.
- `run_experiment.py`: CLI runner for individual experiment configurations.
- `cohort_analysis.py`: Demographic and academic subgroup evaluation tool.
- `plot_results.py`: Publication-grade figure generator.

---

## 2. Reproduction Instructions

To reproduce any or all results from scratch in the local Python 3.12 environment:

```powershell
# Set Python path
$env:PYTHONPATH="."

# 1. Run the entire unit test suite (64/64 tests passing)
.venv\Scripts\python.exe -m pytest tests/

# 2. Run Sanity Baseline (LR on last sequence step)
.venv\Scripts\python.exe -c "from src.seq.sanity_baseline import run_sanity_baseline; run_sanity_baseline()"

# 3. Validate all 20 prediction files
.venv\Scripts\python.exe -c "
from pathlib import Path
from src.seq.validate_predictions import validate_predictions
for f in Path('predictions/seq').glob('*.csv'):
    is_fs = f.name.endswith('_finalscore.csv')
    validate_predictions(f, is_final_score=is_fs)
print('All prediction files verified!')
"

# 4. Regenerate publication figures
.venv\Scripts\python.exe -c "
from src.seq.plot_results import plot_metric_vs_model_ci
from pathlib import Path
plot_metric_vs_model_ci(Path('reports/phase4/by_model_seq.csv'), target='dropout')
plot_metric_vs_model_ci(Path('reports/phase4/by_model_seq.csv'), target='onset')
"
```

---

## 3. Handoff to Niranjan (Forecasting Pipeline Integration)

Niranjan, the multi-task sequence models produce continuous student performance predictions ready to feed into your downstream forecasting pipeline:

- **Files:**
  - `predictions/seq/gru_seq_only_multi_w8_finalscore.csv`
  - `predictions/seq/gru_hybrid_multi_w8_finalscore.csv`
- **Schema:**
  - `learner_key`: Formatted as `{code_module}_{code_presentation}_{id_student}`.
  - `week`: Course snapshot week ($t \in [2, 38]$).
  - `split`: Exactly `val` (164,799 rows).
  - `model`: `gru_seq_only_multi` / `gru_hybrid_multi`.
  - `y_true`: Ground-truth standardized final score.
  - `y_score`: Predicted continuous final score.
- **Properties:**
  - Predictions were generated using a dedicated linear regression head trained with Huber loss ($\lambda = 0.3$) jointly with the classification objective.
  - Padded and masked sequences were handled cleanly via temporal pooling.
  - Grade bands can be mapped directly from `y_score` using the institutional cutoffs (Fail < 40, Pass 40–69, Distinction $\ge$ 70).

---

## 4. Handoff to Tanmay (Fairness & Subgroup Analysis)

Tanmay, detailed demographic evaluations are saved in `reports/phase4/cohort_*.csv`. Key findings to incorporate into your fairness audit:

1. **STEM vs Arts/Social Science Disparity:**
   - STEM modules exhibit significantly higher discrimination (**PR-AUC 0.0785 vs 0.0346**). Because Arts/Social Science courses have fewer structured weekly assessments and sparser clickstreams, students in non-STEM modules are at higher risk of missed interventions.
2. **Deprivation Gradient (IMD Quintiles):**
   - The most deprived decile (`0-10%`) has lower ROC-AUC (**0.7459**) than the most affluent decile (`90-100%`, ROC-AUC **0.8056**). If an intervention policy uses a single global score threshold, false negative rates will be disproportionately higher among students from deprived areas.
3. **Declared Disability:**
   - Students with declared disabilities show higher baseline risk (3.31% vs 2.27%) and higher PR-AUC (**0.0946 vs 0.0683**). The models capture early engagement drops in this cohort effectively.

---

## 5. Security & Test Quarantine Invariant

**CONFIRMED:**
- **Zero test split rows were ever loaded, cached, or written to.**
- The `load_split('test')` security guard remained active throughout Phase 4.
- The 2014J presentation remains completely sealed and pristine for Vivek's final evaluation in Week 4.
- All 64 unit tests across the entire project repository are green.
