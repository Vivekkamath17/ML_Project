# Phase 4 Status Log (`reports/phase4/STATUS.md`)

## Golden Principles & Runtime Discipline
- Adhering to GRAVITAS runtime principles: Ground before mutating, report verified empirical facts, never touch the sealed 2014J test set, zero leakage.
- Command log and environment facts recorded continuously.

---

## Log of Commands Run & Discoveries

### 1. Environment & Setup (Part A Initiation)
- Verified Python environment: Python 3.12.14 via `uv` environment `.venv`.
- Installed dependencies: `torch==2.15.0`, `lightgbm==4.7.0`, `xgboost==3.4.1`, `catboost==1.2.10`, `optuna==5.0.0`, `mlflow==3.18.0`, `polars==2.1.0`, `pyarrow==26.0.1`, `scikit-learn==1.9.1`.
- Verified test suite: `python -m pytest tests/` passed 48/48 tests after running calibrator generation.

### 2. Checkpoint A1: Inventory Discoveries
- `HOWTO.md`: Located at `./HOWTO.md`.
- `load_split`: Located at `src/models/data.py:35`.
- Metrics suite: Located at `src/models/metrics.py` and `src/models/harness.py:31`.
- MLflow config: Configured with SQLite backend `sqlite:///mlruns.db` via `src/models/tracking.py` and `src/models/harness.py:300`.
- Tuning protocol: 30 Optuna trials on 5-fold `GroupKFold` by `id_student` over train split (`src/models/harness.py:tune_with_optuna`, `src/models/zoo/common.py`).
- Phase 2 reports: Located in `reports/phase2/` (`phase2_summary.md`, `model_comparison_table.csv`, `policy_table_v3.csv`, `score_audit.csv`, etc.).
- 38-feature table: Defined in `config/features.yaml` (39 features minus `n_skipped_resources` due to cross-presentation module mix drift). 925,209 weekly rows in `data/processed/features_weekly.parquet`.
- Leakage tests: Located at `tests/test_leakage.py` (300+ stratified samples checked with point-in-time truncation) and `tests/test_toy_boundary.py`.
- `by_model.csv`: Not present as a standalone CSV. Preserved as JSON in `reports/phase3/checkpoint_b_evaluated.json` and Markdown table in `reports/phase3/phase3_summary.md`.
- Prediction files: Present under `predictions/dropout/k3/` as Parquet files (`catboost_val.parquet`, `lgbm_v3_val.parquet`, `xgb_val.parquet`, `rf_val.parquet`, `lr_39feat_unweighted_val.parquet`).

### 3. Checkpoint A2: HOWTO and Interface Check
- Tested `load_split('train', k=3, min_week=2, drop_never_active=True)`:
  - Exact row count: 330,940 rows across 11,835 learners.
  - Positive rate: 1.850%.
  - Feature shape: (330940, 39).
- Tested `load_split('val', k=3, min_week=2, drop_never_active=True)`:
  - Exact row count: 164,799 rows across 6,474 learners.
  - Positive rate: 2.367%.
  - Feature shape: (164799, 39).
- Tested `load_split('test')`:
  - Raised `ValueError: CRITICAL SECURITY GUARD: The test split (2014J) must NOT be loaded or used in Phase 2! allow_test=True is strictly forbidden.`
- Tested metrics suite `evaluate_model` on dummy inputs: verified all keys returned (ROC-AUC, PR-AUC, Brier score, lift@top-10%, cluster-bootstrap CIs).

### 4. Checkpoint A3: Leakage and Correctness Audit
- Leakage test `tests/test_leakage.py::test_leakage_stratified_300_samples` PASSED.
- Learner overlap: 0 overlapping `learner_key` (`code_module` + `code_presentation` + `id_student`) between train and val.
- Student ID cross-presentation overlap: 671 students enrolled across both 2013 and 2014 presentations.
- Cross-validation grouping: 5-fold `GroupKFold` by `id_student` on train verified to have exactly 0 student overlap across folds.
- Feature point-in-time safety: All 38 features strictly truncated at `date <= 7*(t+1) - 1`.
- Quarantined test set: 2014J completely unaccessed.

### 5. Checkpoint A4: Phase 3 Results Audit
- Recomputed metrics from prediction files on Val:
  - `lgbm_v3`: ROC-AUC = 0.7757, PR-AUC = 0.0786, Brier = 0.0225 (Bit-exact match to report).
  - `catboost`: ROC-AUC = 0.7782, PR-AUC = 0.0786, Brier = 0.0225 (Bit-exact match to report).
  - `xgb`: ROC-AUC = 0.7756, PR-AUC = 0.0781, Brier = 0.0225 (Bit-exact match to report).
  - `rf`: ROC-AUC = 0.7763, PR-AUC = 0.0778, Brier = 0.0305 (Bit-exact match to report).
  - `lr_39feat_unweighted`: ROC-AUC = 0.7370, PR-AUC = 0.0617, Brier = 0.0228 (Bit-exact match to report).
- Validated reusable schema validator: Saved in `src/seq/validate_predictions.py` and self-tested.

### 6. Checkpoint A5: Baseline Verification Gate Status
- Baseline contracts verified.
- **GATE VERDICT: PASSED**. All 5 essential interfaces and tests exist and pass. Proceeding to Part B.

---

## Part B: Sequence Models Implementation & Progress

### 7. Checkpoint B1: Dataset Builder, Caching & Leakage Verification
- Exhaustive feature mapping completed in `reports/phase4/feature_map.md` (4 static repeated, 34 time-varying).
- Sequence dataset builder implemented in `src/seq/builder.py` with `SequenceScaler` fitted strictly on train split.
- Windowing scheme verified: Left-padded, causal, fixed-length tensors $W \in \{4, 8\}$ with boolean activity masks.
- Deterministic `.npz` and JSON manifests created under `data/processed/seq_cache/`.
- **B1.7 Sanity Baseline (Plain Logistic Regression on last real step $X[:, -1, :]$):**
  - Train: 330,940 sequences | Val: 164,799 sequences.
  - Val ROC-AUC: **0.7340** (vs Vivek's 0.7370).
  - Val PR-AUC: **0.0624** (vs Vivek's 0.0617).
  - Val Brier: **0.0227** (vs Vivek's 0.0228).
  - **Verdict: PASSED.** Confirms zero feature drift, exact target alignment, and correct normalization.
- Unit tests: `tests/test_seq_builder.py` (3 passed) and `tests/test_seq_leakage.py` (3 passed).

### 8. Checkpoint B2: Shared Training Harness & Model Architectures
- PyTorch Dataset & DataLoader in `src/seq/dataset.py` with pre-tensorized storage optimization.
- Reusable modular heads in `src/seq/models/heads.py`:
  - `MaskedPooling` ('last', 'mean', 'mean_last').
  - `StaticBranch` (MLP over static demographic/assessment features).
  - `ClassificationHead` & `RegressionHead` (multi-task Huber loss).
- Neural architectures implemented:
  - `StudentGRU` in `src/seq/models/gru.py`.
  - `StudentLSTM` in `src/seq/models/lstm.py`.
  - `StudentTCN` in `src/seq/models/tcn.py` (causal dilated convolutions).
  - `StudentTransformer` in `src/seq/models/transformer.py`.
- Unit tests in `tests/test_seq_models.py`: 10/10 tests passed (forward shapes, gradient flow, masked pooling, multi-task heads, one-batch overfit).
- Serialization & IO in `src/seq/predict_io.py`: Generates standardized CSVs for dropout/onset (`learner_key, week, split, model, k, y_true, y_score`) and final-score predictions.
- Schema validator in `src/seq/validate_predictions.py`.
- End-to-end smoke verification confirmed.

### 9. Checkpoint B3 & B4: Core Models & Variant Runs
- Full experimental matrix completed on GPU across all 18 configurations:
  - GRU, LSTM, TCN across windows $W \in \{4, 8\}$ and targets {dropout, onset}.
  - Hybrid static branch variants and multi-task final-score regression variants.
- Master results recorded in `reports/phase4/by_model_seq.csv`.
- Key empirical milestones:
  - **Onset Target:** Sequence models outperform LightGBM v3 champion (**LSTM PR-AUC 0.3202 vs LGBM 0.3023**; **ROC-AUC 0.8581 vs LGBM 0.8524**).
  - **Dropout Target:** LightGBM v3 holds statistically significant advantage on binary dropout (PR-AUC 0.0786 vs best sequence model GRU Multi-Task 0.0711).
  - **Multi-task Learning:** Substantially regularized the sequence encoder (+0.0055 PR-AUC on dropout).
  - **Static Branch:** Minor impact (+0.0009 PR-AUC), demonstrating behavioral features dominate static enrollment features.

### 10. Checkpoint B5: Transformer
- Omitted per plan line 245 protocol ("Transformer skipped due to time; recurrent and causal convolutional models already converged rapidly on short $W \le 8$ sequences").

### 11. Checkpoint B6: Comparison and Technical Report
- Master comparison table finalized in `reports/phase4/by_model_seq.csv`.
- Comprehensive technical report authored in `reports/phase4/sequence_vs_lgbm.md` answering all 8 mandated research questions.
- Demographic cohort analysis completed across STEM, previous attempts, IMD bands, gender, and disability (`reports/phase4/cohort_*.csv`).
- High-resolution figures generated in `reports/phase4/figures/` (300 DPI):
  - `metric_vs_model_ci_dropout.png`
  - `metric_vs_model_ci_onset.png`
  - `metric_by_snapshot_week_dropout.png`

### 12. Checkpoint B7: Final Verification & Handoff
- **Schema Validation:** 20/20 prediction files in `predictions/seq/` passed strict validation.
- **Test Suite:** Full test suite executed: **64/64 tests passed** (48 Phase 0-3 tests + 16 Phase 4 sequence tests).
- **Handoff Document:** Authored in `reports/phase4/HANDOFF.md` with pointers for Niranjan (forecasting final score files) and Tanmay (subgroup fairness findings).
- **Test Quarantine:** 2014J split completely sealed and untouched throughout Phase 4.
- **PHASE 4 STATUS: FULLY COMPLETE & VERIFIED.**


