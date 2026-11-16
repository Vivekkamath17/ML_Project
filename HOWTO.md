# OULAD Modeling Team Guide & Developer Contract (`HOWTO.md`)

Welcome to the Open University Learning Analytics Dataset (OULAD) machine learning repository. This guide establishes the shared development contracts, environment standards, baseline targets, and APIs for building sequence models, continuous forecasting pipelines, calibration layers, and interpretability modules.

---

## 1. Team Roles & Project Responsibilities

| Role | Primary Responsibilities | Target Deliverables |
| :--- | :--- | :--- |
| **Teammate A (Sequence Modeling)** | Recurrent & temporal neural architectures on student interaction trajectories | LSTM, GRU, Transformers, Temporal Convolutional Networks (TCN), dynamic sequence masking, embedding layers for VLE activities. |
| **Teammate B (Forecasting & Interpretability)** | Continuous score forecasting, post-hoc calibration, explainability & fairness audits | Multi-horizon score regressions, Platt/Isotonic/Conformal calibration curves, TreeSHAP / KernelSHAP feature attributions, algorithmic demographic audits. |
| **Shared Team Contract** | Standardized evaluation, leak-free validation, cross-validation discipline | `src.models.data.load_split`, `src.models.predictions.write_predictions`, `src.models.harness.evaluate_model`, 5-seed headline reporting. |

---

## 2. Environment Setup

### 2.1 Python Version & Virtual Environment
The repository requires **Python 3.12**. Always use an isolated virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS (Bash)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 MLflow Tracking Server (Local SQLite Backend)
All experiment logging uses a persistent SQLite backend located at `sqlite:///mlruns.db`.  
To inspect runs, parameters, metrics, and logged artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 3. Repository Directory Structure

```
c:\ML_Project\
├── config/
│   ├── config.yaml              # Global pipeline configuration
│   └── features.yaml            # Exhaustive whitelist of 39 sanctioned features
├── data/
│   ├── raw/                     # READ-ONLY raw OULAD CSV tables (never modify!)
│   ├── interim/                 # Parquet versions of raw tables and grid
│   └── processed/
│       ├── features_weekly.parquet   # 925,209 learner-week rows across 29,884 students
│       ├── learners.parquet          # Student cohort summary & effective unregistration
│       ├── labels.parquet            # Pre-computed multi-horizon targets
│       └── audit_demographics.parquet # Demographic attributes strictly isolated for fairness audits
├── models/
│   ├── baselines/               # Official trained models & threshold configs
│   │   ├── lgbm_champion_v3_official.txt # LightGBM v3 booster
│   │   └── threshold_config.yaml         # Operational decision thresholds
│   └── preprocessing/
│       └── train_scaler.joblib  # StandardScaler fitted strictly on Train split
├── predictions/                 # Standardized Parquet prediction files
│   └── dropout/k3/
│       ├── lgbm_v3_val.parquet              # Official LightGBM baseline predictions
│       └── lr_39feat_unweighted_val.parquet # 39-feature Logistic Regression baseline
├── reports/                     # Published performance summaries and audit logs
│   └── phase2/
│       ├── phase2_summary.md    # Official Phase 2 results and findings
│       ├── policy_table_v3.csv  # Operational policy lead-time & yield table
│       └── score_audit.csv      # 22-presentation assessment normalization audit
├── src/
│   ├── data/                    # Data preparation, feature engineering & leakage guards
│   └── models/
│       ├── data.py              # load_split() data loader with strict point-in-time filtering
│       ├── harness.py           # Standard evaluation, paired bootstrap, CV & Optuna
│       ├── predictions.py       # write_predictions() and read_predictions()
│       └── tracking.py          # MLflow logging utilities
├── tests/                       # Pytest test suite (leakage, schema, harness)
└── HOWTO.md                     # This developer contract
```

---

## 4. The Four Core Prediction Targets

All models must target one of these four standardized problem formulations:

1. **Administrative Dropout (`label_dropout_k{2,3,4}`)**:
   - Binary indicator: Will the learner unregister within the next $k$ weeks ($k \in \{2, 3, 4\}$)?
   - Evaluated on enrolled students (`is_enrolled_at_week_t == True`, week $\ge 2$).
2. **Behavioral Disengagement Onset (`label_inactive_k{2,3,4}`)**:
   - Binary indicator: Given the learner clicked in week $t$ (`weekly_clicks > 0`), will they log zero clicks across the next $k$ weeks?
3. **Continuous Final Score Forecasting (`final_score`)**:
   - Continuous target in $[0.0, 100.0]$ representing cumulative assessment performance.
   - Evaluated strictly on the active non-withdrawn cohort ($N=5,079$ in validation) at snapshot weeks 4, 8, and 12.
4. **End-of-Term Grade Band (`grade_band`)**:
   - Multiclass target with 3 classes: `Fail`, `Pass`, `Distinction`.
   - Prediction file must output class probability columns: `y_score_fail`, `y_score_pass`, `y_score_distinction`.

---

## 5. How to Load Data (`src.models.data.load_split`)

Always load data through `load_split()`. It guarantees point-in-time leakage filtering, excludes post-unregistration rows, handles censoring, and enforces split isolation.

```python
from src.models.data import load_split

# 1. Classification: 3-Week Dropout Horizon (Default)
X_tr, y_tr, meta_tr = load_split("train", k=3, min_week=2, drop_never_active=True)
X_va, y_va, meta_va = load_split("val", k=3, min_week=2, drop_never_active=True)

# 2. Continuous Score Forecasting (Non-Withdrawn Students Only)
X_tr_sc, y_tr_sc, meta_tr_sc = load_split(
    "train", non_withdrawn_only=True, drop_never_active=True, target_col="final_score"
)
X_va_sc, y_va_sc, meta_va_sc = load_split(
    "val", non_withdrawn_only=True, drop_never_active=True, target_col="final_score"
)

# 3. Behavioral Disengagement Onset
X_tr_in, y_tr_in, meta_tr_in = load_split("train", target_col="label_inactive_k3")
X_va_in, y_va_in, meta_va_in = load_split("val", target_col="label_inactive_k3")

# 4. Multiclass Grade Band
X_tr_gb, y_tr_gb, meta_tr_gb = load_split("train", target_col="grade_band")
X_va_gb, y_va_gb, meta_va_gb = load_split("val", target_col="grade_band")
```

> [!CAUTION]
> **Test Split Quarantine Rule**: The test split (`2014J`) is strictly quarantined.  
> Calling `load_split("test")` raises a critical security exception.  
> **NEVER attempt to bypass test quarantine** in your code, notebooks, or scripts during model development. All model selection, tuning, and evaluation must use `train` (`2013B`, `2013J`) and `val` (`2014B`).

---

## 6. Feature Manifest & The Official 38-Feature Set

The complete list of 39 sanctioned features is whitelisted in `config/features.yaml`.  
The official Phase 2 baseline model (**LightGBM v3**) uses **38 features** by excluding `n_skipped_resources` due to presentation-level module mix drift.

```python
from src.data.features import load_feature_manifest

# Load the full 39 sanctioned features
all_features = load_feature_manifest("config/features.yaml")

# Official v3 baseline feature subset (38 features)
v3_features = [f for f in all_features if f != "n_skipped_resources"]
```

---

## 7. Baseline Numbers to Beat (Validation Split: `2014B`, $week \ge 2$)

To demonstrate a valid improvement, new architectures must beat these validated baselines:

### 7.1 Dropout Prediction ($k=3$ Weeks Ahead, Base Rate = 2.367%)
| Model | Features | Val ROC-AUC | Val PR-AUC | Val Brier Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LightGBM v3 (Official Champion)** | 38 | **0.7757** | **0.0786** | **0.0221** | Train-only tuned: 463 trees, depth 3, leaves 50. **5-seed headline mean: 0.7724 ± 0.0012**. |
| **Logistic Regression (Balanced)** | 39 | 0.7438 | 0.0638 | 0.1742 | $C=0.01$, balanced class weights |
| **Logistic Regression (Unweighted)** | 39 | 0.7370 | 0.0617 | 0.0227 | $C=0.1$, standard L2 regularization |
| **Recency-Only Baseline** | 1 | 0.6552 | 0.0418 | 0.0232 | `days_since_last_activity` single-feature model |
| **Clicks-Only Baseline** | 1 | 0.6481 | 0.0402 | 0.0233 | `weekly_clicks` single-feature model |
| **Score-Only Baseline** | 1 | 0.6449 | 0.0435 | 0.0234 | `weighted_score_to_date` single-feature model |
| **Majority Class Prior** | 0 | 0.5000 | 0.0237 | 0.0231 | Constant training prior ($p = 0.0185$) |

### 7.2 Continuous Final Score Forecasting ($N=5,079$ Active Non-Withdrawn Cohort)
| Snapshot Week | Per-Module Fitted Baseline MAE ($R^2$) | LightGBM MAE ($R^2$) | Target Metric to Beat |
| :---: | :---: | :---: | :---: |
| **Week 4** | 23.53 ($R^2 = 0.109$) | **16.19 ($R^2 = 0.467$)** | $\text{MAE} < 16.19$ |
| **Week 8** | 15.69 ($R^2 = 0.457$) | **12.59 ($R^2 = 0.638$)** | $\text{MAE} < 12.59$ |
| **Week 12** | 14.80 ($R^2 = 0.521$) | **9.81 ($R^2 = 0.781$)** | $\text{MAE} < 9.81$ |

---

## 8. Model Training & Prediction Workflow Template

Below is a complete, copy-pasteable script template demonstrating training, validation evaluation, latency measurement, and standardized prediction output:

```python
import time
import lightgbm as lgb
from src.models.data import load_split
from src.models.predictions import write_predictions, read_predictions
from src.models.harness import evaluate_model

# 1. Load data
X_tr, y_tr, meta_tr = load_split("train", k=3, min_week=2)
X_va, y_va, meta_va = load_split("val", k=3, min_week=2)

# 2. Select v3 sanctioned features (38 features)
features = [col for col in X_tr.columns if col != "n_skipped_resources"]

# 3. Fit model and record training time
t0 = time.perf_counter()
train_set = lgb.Dataset(X_tr[features], label=y_tr)
params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1, "random_state": 42}
model = lgb.train(params, train_set, num_boost_round=100)
train_time_sec = time.perf_counter() - t0

# 4. Predict probabilities on validation set and record inference latency
t1 = time.perf_counter()
val_probs = model.predict(X_va[features])
inference_latency_ms_per_1k = ((time.perf_counter() - t1) / len(X_va)) * 1000 * 1000

# 5. Evaluate against standard metrics with student-clustered CIs
metrics = evaluate_model(y_va, val_probs, cluster_ids=meta_va["id_student"], threshold=0.035, n_resamples=100)
print(f"Train time: {train_time_sec:.2f}s | Latency: {inference_latency_ms_per_1k:.2f}ms / 1k rows")
print(f"Validation: ROC-AUC = {metrics['roc_auc']:.4f}, PR-AUC = {metrics['pr_auc']:.4f}, Lift@10% = {metrics['lift_at_top10']:.2f}x")

# 6. Format and save standardized predictions
df_preds = meta_va[["id_student", "code_module", "code_presentation", "week"]].copy()
df_preds["model"] = "example_lgbm_model"
df_preds["k"] = 3
df_preds["target"] = "label_dropout_k3"
df_preds["split"] = "val"
df_preds["y_true"] = y_va
df_preds["y_prob"] = val_probs

metadata = {
    "hyperparameters": params,
    "train_time_sec": train_time_sec,
    "inference_latency_ms_per_1k": inference_latency_ms_per_1k,
    "metrics": {k: float(v) for k, v in metrics.items() if k != "ci"},
}
out_path = write_predictions(df_preds, metadata=metadata)
print(f"Successfully saved predictions to: {out_path}")
```

### Actual Output from Running the Template Script:
```
Loaded split='train' (k=3, min_week=2, drop_never_active=True): 330,940 rows across 11,835 learners. Positive rate: 1.850%
Loaded split='val' (k=3, min_week=2, drop_never_active=True): 164,799 rows across 6,474 learners. Positive rate: 2.367%
Train time: 3.82s | Latency: 4.88ms / 1k rows
Validation: ROC-AUC = 0.7616, PR-AUC = 0.0736, Lift@10% = 3.38x
Successfully saved predictions to: C:\ML_Project\predictions\dropout\k3\example_lgbm_model_val.parquet
Verified read-back shape: (164799, 10)
```

---

## 9. 5-Fold Cross-Validation, Seed Variance & Headline Reporting Rule

To ensure reproducibility and statistical validity:
1. **5-Fold Learner-Grouped CV on Training Set**:
   Always group by `id_student` (never split rows from the same student across folds):
   ```python
   from src.models.harness import learner_group_cv

   for fold, (train_idx, val_idx) in enumerate(learner_group_cv(X_tr, y_tr, groups=meta_tr["id_student"], n_splits=5)):
       # Train on train_idx, evaluate on val_idx
       ...
   ```
2. **5-Seed Evaluation Rule**:
   All headline model comparisons must run across 5 fixed random seeds: `[42, 43, 44, 45, 46]`.
   Report headline results as the **mean ± standard deviation** across the 5 seeds. Do not select or report a single lucky seed.
3. **Paired Bootstrap Against Baseline**:
   Always evaluate your model against the saved baseline using `paired_bootstrap` clustered by `id_student`:
   ```python
   from src.models.predictions import read_predictions
   from src.models.harness import paired_bootstrap

   df_base, _ = read_predictions("predictions/dropout/k3/lgbm_v3_val.parquet")
   boot_res = paired_bootstrap(
       y_true=df_preds["y_true"],
       y_prob_a=df_preds["y_prob"],
       y_prob_b=df_base["y_prob"],
       cluster_ids=df_preds["id_student"], # Clustered strictly by student ID
       n_boot=500,
       seed=42,
   )
   print(f"Delta ROC: {boot_res['delta_roc']:+.4f} [{boot_res['delta_roc_ci'][0]:+.4f}, {boot_res['delta_roc_ci'][1]:+.4f}]")
   print(f"Delta PR:  {boot_res['delta_pr']:+.4f} [{boot_res['delta_pr_ci'][0]:+.4f}, {boot_res['delta_pr_ci'][1]:+.4f}]")
   ```

---

## 10. MLflow Logging Convention (`src.models.harness.log_run`)

All runs must be logged to the local SQLite database (`sqlite:///mlruns.db`) following this strict naming and tagging standard:

- **Run Name Format**: `<family>/<model>/<variant>/seed<N>`
  - Example: `sequence/lstm/bidirectional/seed42`
  - Example: `calibration/platt/isotonic/seed42`
  - Example: `tree/lgbm/v3/seed42`
- **Authorized Experiments**: Must be one of `{'dropout', 'onset', 'score', 'grade_band', 'calibration'}`.
- **Required Tags**: `family`, `model`, `variant`, `seed`, `target`, `horizon`.

```python
from src.models.harness import log_run

run_id = log_run(
    run_name="sequence/lstm/bidirectional/seed42",
    params={"hidden_dim": 128, "num_layers": 2, "learning_rate": 0.001, "batch_size": 256, "k": 3, "target": "label_dropout_k3"},
    metrics={"roc_auc": 0.7815, "pr_auc": 0.0842, "brier_score": 0.0219},
    feature_list=v3_features,
    experiment_name="dropout",
)
```

---

## 11. Known Data Quirks & Modeling Caveats

1. **Weeks 0–1 vs Precourse Activity**:
   - Activity before Day 0 (`date < 0`) is aggregated into static feature `precourse_clicks`.
   - The official evaluation grid begins at week 2 (`min_week=2`) to allow 2 weeks of observed course trajectory for lag/momentum features. Weeks 0 and 1 are kept in the data solely for rolling history.
2. **Presentation Module Coverage**:
   - **Module CCC**: Present in Validation (2014B) and Test (2014J), but **completely absent from Train** (2013B, 2013J). Models must generalize zero-shot to CCC's curriculum structure.
   - **Module AAA**: Offered only in J presentations (2013J, 2014J); completely absent from B presentations (2013B, 2014B).
   - **Module EEE**: Absent from 2013B; present in 2013J, 2014B, 2014J.
3. **Calibration Under-Prediction vs High-Probability Over-Confidence**:
   - The model's overall predicted average probability is ~1.9%, slightly under-predicting the validation base rate (2.37%) due to the training prior (1.85%).
   - However, in high probability bins ($\hat{p} > 0.20$), models are substantially **over-confident** (predicting 24% to 34% probability while empirical attrition is ~11%). Teammate B should implement isotonic regression or temperature scaling to calibrate these probabilities.
4. **Assessment-Driven Signal Shift**:
   - Prior to week 6, dropout signal is driven primarily by click intensity, streak dynamics, and inactivity recency.
   - From week 8 onwards, assessment submission features (`weighted_score_to_date`) dominate model importance.

---

## 12. Pull Request (PR) & Collaboration Rules

Before opening a pull request or submitting experimental results:
1. **Sanctioned Features Only**: Any new feature must be whitelisted in `config/features.yaml`. Do not use undeclared features.
2. **Zero Demographic Features**: Demographic columns (`gender`, `imd_band`, `highest_education`, `age_band`, `disability`) are quarantined for algorithmic bias audits. Never feed them into model feature sets.
3. **Zero Test Split Exposure**: Never call `load_split("test")` or access presentation `2014J`.
4. **All Tests Must Pass**: Run the test suite before pushing:
   ```bash
   python -m pytest tests/
   ```
5. **Standard Output Parquet**: Ensure your prediction files are generated via `write_predictions` with all schema columns present.

---

## 13. Teammate Contract: SHAP Top-5 Feature Contributions
To enable automatic rendering in the interactive Streamlit dashboard (`app/` Learner Detail page), teammate SHAP explanations must follow this standard parquet schema:
- **Location**: `reports/shap/<model>_k3_val_top5.parquet`
- **Required Columns**:
  - `code_module` (string)
  - `code_presentation` (string)
  - `id_student` (int64)
  - `week` (int64)
  - `feature` (string)
  - `shap_value` (float64)
  - `rank` (int64, 1 to 5)
- **Degradation**: If this file is absent, the dashboard automatically renders a clean "Pending teammate delivery" status card without crashing.

---

## 14. Teammate Contract: Score and Grade Band Forecasting Predictions
To support final score regression and multi-class grade band forecasts in the dashboard and reports:
- **Continuous Score Regression**:
  - **Location**: `predictions/score/<model>_w<week>_val.parquet` or `predictions/score/snapshot/<model>_val.parquet`
  - **Required Columns**: `code_module`, `code_presentation`, `id_student`, `week`, `y_true`, `y_pred`
- **Multi-Class Grade Band Forecasting**:
  - **Location**: `predictions/grade_band/<model>_w<week>_val.parquet` or `predictions/grade_band/snapshot/<model>_val.parquet`
  - **Required Columns**: `code_module`, `code_presentation`, `id_student`, `week`, `y_true`, `y_score_fail`, `y_score_pass`, `y_score_distinction`

