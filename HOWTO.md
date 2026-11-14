# OULAD Modeling Team Guide & Developer Contract (`HOWTO.md`)

Welcome to the Open University Learning Analytics Dataset (OULAD) machine learning repository. This guide establishes the shared development contracts, environment standards, baseline targets, and APIs for building sequence models, forecasting pipelines, calibration layers, and interpretability modules.

---

## 1. Environment Setup

### 1.1 Python Version & Virtual Environment
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

### 1.2 MLflow Tracking Server (Local SQLite Backend)
All experiment logging uses a persistent SQLite backend located at `sqlite:///mlruns.db`.  
To inspect runs, parameters, metrics, and logged artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 2. Repository Directory Structure

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

## 3. How to Load Data (`src.models.data.load_split`)

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

# 3. Behavioral Disengagement Onset (Active Clicking in Week t -> 0 Clicks in Next k Weeks)
X_tr_in, y_tr_in, meta_tr_in = load_split("train", target_col="label_inactive_k3")
X_va_in, y_va_in, meta_va_in = load_split("val", target_col="label_inactive_k3")
```

> [!CAUTION]
> **Test Split Quarantine Rule**: The test split (`2014J`) is strictly quarantined.  
> Calling `load_split("test")` raises a critical security exception.  
> **NEVER attempt to bypass test quarantine** in your code, notebooks, or scripts during model development. All model selection, tuning, and evaluation must use `train` (`2013B`, `2013J`) and `val` (`2014B`).

---

## 4. Feature Manifest & The Official 38-Feature Set

The complete list of 39 sanctioned features is whitelisted in `config/features.yaml`.  
The official Phase 2 baseline model (**LightGBM v3**) uses **38 features** by excluding `n_skipped_resources` due to presentation-level module mix drift.

```python
from src.data.features import load_feature_manifest

# Load the full 39 sanctioned features
all_features = load_feature_manifest("config/features.yaml")

# Official v3 baseline feature subset (38 features)
v3_features = [f for f in all_features if f != "n_skipped_resources"]
```

### Feature Categories:
1. **Interaction Intensity & Regularity**: `weekly_clicks`, `cum_clicks`, `clicks_per_active_day`, `active_days_count`, `relative_activity_ratio`.
2. **Temporal Trend & Momentum**: `clicks_growth_rate`, `rolling_2w_diff`, `acceleration_clicks`, `streak_active_weeks`, `streak_inactive_weeks`.
3. **Assessment Performance**: `weighted_score_to_date`, `avg_submission_lateness`, `unweighted_pass_rate`, `has_any_assessment`, `n_banked`.
4. **Curriculum Pacing (Normalized)**: `pct_resources_visited`, `pacing_vs_cohort_median`, `prerequisite_gap`. *(Note: `n_skipped_resources` is excluded from v3)*.
5. **VLE Resource Mix**: Proportion of clicks across `resource`, `oucontent`, `forumng`, `quiz`, `subpage`, `homepage`, `url`.
6. **Registration & Precourse Context**: `studied_credits`, `precourse_clicks`, `days_before_start_registered`, `num_of_prev_attempts`.

---

## 5. Baseline Targets to Beat (Validation Split: `2014B`, $week \ge 2$)

To demonstrate a valid improvement, new architectures (sequence models, survival models, neural nets) must outperform these validated baselines:

### 5.1 Dropout Prediction ($k=3$ Weeks Ahead, Base Rate = 2.367%)
| Model | Features | Val ROC-AUC | Val PR-AUC | Val Brier Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LightGBM v3 (Official Champion)** | 38 | **0.7757** | **0.0786** | **0.0221** | Train-only tuned: 463 trees, depth 3, leaves 50. 5-seed mean: 0.7724. |
| **Logistic Regression (Balanced)** | 39 | 0.7438 | 0.0638 | 0.1742 | $C=0.01$, balanced class weights |
| **Logistic Regression (Unweighted)** | 39 | 0.7370 | 0.0617 | 0.0227 | $C=0.1$, standard L2 regularization |
| **Recency-Only Baseline** | 1 | 0.6552 | 0.0418 | 0.0232 | `days_since_last_activity` single-feature model |
| **Clicks-Only Baseline** | 1 | 0.6481 | 0.0402 | 0.0233 | `weekly_clicks` single-feature model |
| **Score-Only Baseline** | 1 | 0.6449 | 0.0435 | 0.0234 | `weighted_score_to_date` single-feature model |
| **Majority Class Prior** | 0 | 0.5000 | 0.0237 | 0.0231 | Constant training prior ($p = 0.0185$) |

### 5.2 Continuous Final Score Forecasting ($N=5,079$ Active Non-Withdrawn Cohort)
| Snapshot Week | Per-Module Fitted Baseline MAE ($R^2$) | LightGBM MAE ($R^2$) | Target Metric to Beat |
| :---: | :---: | :---: | :---: |
| **Week 4** | 23.53 ($R^2 = 0.109$) | **16.19 ($R^2 = 0.467$)** | $\text{MAE} < 16.19$ |
| **Week 8** | 15.69 ($R^2 = 0.457$) | **12.59 ($R^2 = 0.638$)** | $\text{MAE} < 12.59$ |
| **Week 12** | 14.80 ($R^2 = 0.521$) | **9.81 ($R^2 = 0.781$)** | $\text{MAE} < 9.81$ |

---

## 6. Model Training & Prediction Workflow Template

Below is a complete, executable 20-line template demonstrating training, validation evaluation, and standardized prediction output:

```python
import lightgbm as lgb
from src.models.data import load_split
from src.models.predictions import write_predictions
from src.models.harness import evaluate_model

# 1. Load data
X_tr, y_tr, meta_tr = load_split("train", k=3, min_week=2)
X_va, y_va, meta_va = load_split("val", k=3, min_week=2)

# 2. Select v3 sanctioned features (38 features)
features = [col for col in X_tr.columns if col != "n_skipped_resources"]

# 3. Fit model
train_set = lgb.Dataset(X_tr[features], label=y_tr)
params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1, "random_state": 42}
model = lgb.train(params, train_set, num_boost_round=100)

# 4. Predict probabilities on validation set
val_probs = model.predict(X_va[features])

# 5. Evaluate against standard metrics
metrics = evaluate_model(y_va, val_probs, threshold=0.035)
print(f"Validation: ROC-AUC = {metrics['roc_auc']:.4f}, PR-AUC = {metrics['pr_auc']:.4f}")

# 6. Format and save standardized predictions
df_preds = meta_va[["id_student", "code_module", "code_presentation", "week"]].copy()
df_preds["y_true"] = y_va
df_preds["y_prob"] = val_probs

metadata = {
    "model_name": "example_lgbm_model",
    "target": "label_dropout_k3",
    "horizon": 3,
    "split": "val",
    "features_used": features,
    "hyperparameters": params,
    "metrics": metrics,
}
write_predictions(df_preds, "predictions/dropout/k3/example_lgbm_val.parquet", metadata)
```

---

## 7. Model Evaluation & Clustered Paired Bootstrap

To verify whether your model statistically outperforms the baseline, run a paired bootstrap clustered by `id_student`:

```python
from src.models.predictions import read_predictions
from src.models.harness import evaluate_model, paired_bootstrap

# Read your model predictions and the official baseline
df_mine, meta_mine = read_predictions("predictions/dropout/k3/example_lgbm_val.parquet")
df_base, meta_base = read_predictions("predictions/dropout/k3/lgbm_v3_val.parquet")

# 1. Point metrics
metrics_mine = evaluate_model(df_mine["y_true"], df_mine["y_prob"], threshold=0.035)
metrics_base = evaluate_model(df_base["y_true"], df_base["y_prob"], threshold=0.035)

# 2. Paired bootstrap (500 resamples clustered by id_student)
boot_res = paired_bootstrap(
    y_true=df_mine["y_true"],
    y_prob_a=df_mine["y_prob"],      # Your model
    y_prob_b=df_base["y_prob"],      # Baseline
    cluster_ids=df_mine["id_student"], # Cluster strictly by student ID
    n_boot=500,
    seed=42,
)

print(f"Delta ROC: {boot_res['delta_roc']:+.4f} [95% CI: {boot_res['delta_roc_ci'][0]:+.4f}, {boot_res['delta_roc_ci'][1]:+.4f}]")
print(f"Delta PR:  {boot_res['delta_pr']:+.4f} [95% CI: {boot_res['delta_pr_ci'][0]:+.4f}, {boot_res['delta_pr_ci'][1]:+.4f}]")
```

---

## 8. Prediction File Serialization Contract (`src.models.predictions`)

All prediction files must follow this schema and naming convention:
- **Directory Path**: `predictions/<target_type>/<horizon>/<model_name>_<split>.parquet`
  - Example: `predictions/dropout/k3/lstm_sequence_val.parquet`
  - Example: `predictions/score/snapshots/transformer_w8_val.parquet`
- **Required Columns**:
  - `id_student` (integer)
  - `code_module` (string)
  - `code_presentation` (string)
  - `week` (integer)
  - `y_true` (binary integer for classification, float for regression)
  - `y_prob` (float in $[0.0, 1.0]$ for classification) OR `y_pred` (float for regression)
- **Validation Rules**:
  - `(id_student, code_module, code_presentation, week)` must form a unique key with zero duplicates.
  - Zero NaN values allowed in predictions or identifiers.
  - Automatic test quarantine check: rejects `split="test"` and presence of presentation `2014J`.
- **Required Metadata Dictionary**:
  ```python
  metadata = {
      "model_name": "lstm_sequence_v1",
      "target": "label_dropout_k3",
      "horizon": 3,
      "split": "val",
      "features_used": [...],
      "hyperparameters": {...},
      "git_commit": "...",
      "timestamp": "2026-11-14T...",
  }
  ```

---

## 9. MLflow Run Logging (`src.models.harness.log_run`)

Always log your final models, parameters, and artifact paths to MLflow:

```python
from src.models.harness import log_run

run_id = log_run(
    model_name="lstm_sequence_v1",
    params={"hidden_dim": 64, "num_layers": 2, "learning_rate": 0.001, "batch_size": 256},
    metrics={"roc_auc": 0.7812, "pr_auc": 0.0845, "brier_score": 0.0218},
    feature_list=v3_features,
    artifacts={"model_weights": "models/checkpoints/lstm_best.pt"},
    experiment_name="oulad_sequence_models"
)
print("Logged run to MLflow with Run ID:", run_id)
```

---

## 10. Hyperparameter Tuning Contract

To prevent validation tuning optimism and data leakage:
1. **Tune Inside Train Split Only**:
   - Hyperparameter search (Optuna / Grid Search) and early stopping must use **only** the `train` split (`2013B` and `2013J`).
   - Use 3-fold `GroupKFold` grouped strictly by `id_student`:
     ```python
     from src.models.harness import learner_group_cv, tune_with_optuna

     def objective(trial):
         lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
         # Evaluate across folds using learner_group_cv(X_tr, y_tr, groups=meta_tr["id_student"], n_splits=3)
         return mean_val_metric

     study = tune_with_optuna(objective, n_trials=30, seed=42)
     ```
2. **Score Held-Out Validation Once**:
   - Refit the model on the full training split using the best hyperparameters.
   - Score the model **once** on `val` (`2014B`).
   - **Never early-stop on `2014B`**.

---

## 11. Common Pitfalls to Avoid

1. **Point-in-Time Leakage**: Never compute features using future information (clicks or unregistrations after day $7(t+1)-1$). `load_split()` enforces this.
2. **Student Identity Leakage Across Folds**: A student taking multiple modules (e.g. BBB and DDD) must never have one course in train and another in validation. Always group CV by `id_student`, not by composite key.
3. **Demographic Attributes in Model Training**: Demographics (`gender`, `imd_band`, `highest_education`, `age_band`, `disability`) are strictly isolated in `data/processed/audit_demographics.parquet` for fairness evaluations. **Never include them in model feature matrices.**
4. **Modifying Raw Data Files**: Never edit files in `data/raw/`. All pipeline processing must flow deterministically through `python -m src.data.build_all`.
5. **Accidental Test Set Access**: Never call `load_split("test")` or attempt to access test data. The test split (`2014J`) is reserved exclusively for the final publication audit.

---

## 12. FAQ & Troubleshooting

- **Why is module CCC missing from training data?**  
  Module CCC was first introduced in `2014B` and does not exist in `2013B` or `2013J`. Models must generalize zero-shot to unseen course structures in validation.
- **Why are GGG assessment weights equal to 0?**  
  In the OULAD raw catalog, all 10 continuous assessments for module GGG are assigned weight `0.0`, and its final exam is unrecorded. For GGG, `final_score` is computed as the unweighted mean of submitted continuous assessments.
- **Why does the active non-withdrawn cohort have $N=5,079$ instead of 5,191?**  
  Exactly 112 students registered for course offerings in `2014B` but never logged a single VLE interaction (`never_active_learner == True`). 100% of these 112 students received a final result of `Fail`. Under our unified forecasting cohort definition, never-active learners are excluded from continuous score regressions ($5,191 - 112 = 5,079$).
- **How do I run the automated test suite?**  
  Run the test suite using pytest from the repository root:
  ```bash
  python -m pytest tests/
  ```
  Ensure all tests pass before submitting pull requests or committing code.
