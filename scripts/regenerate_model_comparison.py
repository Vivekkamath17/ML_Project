"""
Regenerate the full model comparison table from scratch (Validation 2014B, k=3, min_week=2).
Strictly uses load_split().
Includes:
- Majority Prior
- Recency-Only LR (days_since_last_activity)
- Clicks-Only LR (weekly_clicks)
- Score-Only LR (weighted_score_to_date) -> 0.6449 ROC-AUC
- 39-Feature LR Unweighted (C=0.1)
- 39-Feature LR Balanced (C=0.01 / C=1.0)
- LightGBM Champion v1 (39 features)
- LightGBM Champion v3 (38 features, without n_skipped_resources, train-only tuned official baseline)
- 500-sample learner-cluster paired bootstrap block for LightGBM vs LR
- Frozen thresholds for v3 for k=2, 3, 4
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef
)
from joblib import Parallel, delayed

from src.models.data import load_split
from src.models.metrics import evaluate_predictions, compute_point_metrics
from src.models.train_lr import prepare_lr_matrices

print("=" * 75)
print("REGENERATING FULL MODEL COMPARISON TABLE FROM SCRATCH (VAL 2014B, k=3)")
print("=" * 75)

X_tr, y_tr, meta_tr = load_split("train", k=3, min_week=2)
X_va, y_va, meta_va = load_split("val", k=3, min_week=2)
keys_va = meta_va["code_module"] + "_" + meta_va["code_presentation"] + "_" + meta_va["id_student"].astype(str)

base_val_rate = float(y_va.mean())
n_resamples = 500

records = []

# ── 1. Majority Prior Baseline ─────────────────────────────────────────────
prior_prob = np.full(len(y_va), float(y_tr.mean()))
ev_prior = evaluate_predictions(y_va, prior_prob, learner_keys=keys_va, n_resamples=n_resamples)
records.append({
    "Model": "Majority Class Prior",
    "Features": "None (constant prior p=0.0185)",
    "ROC-AUC": f"{ev_prior['point']['roc_auc']:.4f} [{ev_prior['ci']['roc_auc'][0]:.4f}, {ev_prior['ci']['roc_auc'][1]:.4f}]",
    "PR-AUC": f"{ev_prior['point']['pr_auc']:.4f} [{ev_prior['ci']['pr_auc'][0]:.4f}, {ev_prior['ci']['pr_auc'][1]:.4f}]",
    "PR Multiplier": f"{ev_prior['point']['pr_auc']/base_val_rate:.2f}x",
    "Brier Score": f"{ev_prior['point']['brier']:.4f}",
    "Hyperparameters / Notes": "Constant train prior probability",
})

# ── 2. Single-Feature Baselines (Fitted on Train with Median Imputation) ───
single_features = [
    ("Recency-Only LR", "days_since_last_activity"),
    ("Clicks-Only LR", "weekly_clicks"),
    ("Score-Only LR", "weighted_score_to_date"),
]

for name, feat in single_features:
    xtr = X_tr[[feat]].fillna(X_tr[feat].median())
    xva = X_va[[feat]].fillna(X_tr[feat].median())
    sc = StandardScaler().fit(xtr)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(sc.transform(xtr), y_tr)
    pv = lr.predict_proba(sc.transform(xva))[:, 1]
    ev = evaluate_predictions(y_va, pv, learner_keys=keys_va, n_resamples=n_resamples)
    records.append({
        "Model": name,
        "Features": f"Single feature: {feat}",
        "ROC-AUC": f"{ev['point']['roc_auc']:.4f} [{ev['ci']['roc_auc'][0]:.4f}, {ev['ci']['roc_auc'][1]:.4f}]",
        "PR-AUC": f"{ev['point']['pr_auc']:.4f} [{ev['ci']['pr_auc'][0]:.4f}, {ev['ci']['pr_auc'][1]:.4f}]",
        "PR Multiplier": f"{ev['point']['pr_auc']/base_val_rate:.2f}x",
        "Brier Score": f"{ev['point']['brier']:.4f}",
        "Hyperparameters / Notes": "C=1.0, train median imputation",
    })

# ── 3. 39-Feature Logistic Regression (Unweighted and Balanced) ────────────
X_tr_lr, X_va_lr, _ = prepare_lr_matrices(X_tr, X_va)

# Unweighted LR (C=0.1)
lr_uw = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver="lbfgs")
lr_uw.fit(X_tr_lr, y_tr)
p_lr_uw = lr_uw.predict_proba(X_va_lr)[:, 1]
ev_uw = evaluate_predictions(y_va, p_lr_uw, learner_keys=keys_va, n_resamples=n_resamples)
records.append({
    "Model": "39-Feature LR (unweighted)",
    "Features": "39 continuous + 39 missing indicators",
    "ROC-AUC": f"{ev_uw['point']['roc_auc']:.4f} [{ev_uw['ci']['roc_auc'][0]:.4f}, {ev_uw['ci']['roc_auc'][1]:.4f}]",
    "PR-AUC": f"{ev_uw['point']['pr_auc']:.4f} [{ev_uw['ci']['pr_auc'][0]:.4f}, {ev_uw['ci']['pr_auc'][1]:.4f}]",
    "PR Multiplier": f"{ev_uw['point']['pr_auc']/base_val_rate:.2f}x",
    "Brier Score": f"{ev_uw['point']['brier']:.4f}",
    "Hyperparameters / Notes": "C=0.1, L2 penalty, lbfgs solver",
})

# Balanced LR (C=0.01 / C=1.0)
lr_bal = LogisticRegression(C=0.01, max_iter=1000, random_state=42, solver="lbfgs", class_weight="balanced")
lr_bal.fit(X_tr_lr, y_tr)
p_lr_bal = lr_bal.predict_proba(X_va_lr)[:, 1]
ev_bal = evaluate_predictions(y_va, p_lr_bal, learner_keys=keys_va, n_resamples=n_resamples)
records.append({
    "Model": "39-Feature LR (balanced)",
    "Features": "39 continuous + 39 missing indicators",
    "ROC-AUC": f"{ev_bal['point']['roc_auc']:.4f} [{ev_bal['ci']['roc_auc'][0]:.4f}, {ev_bal['ci']['roc_auc'][1]:.4f}]",
    "PR-AUC": f"{ev_bal['point']['pr_auc']:.4f} [{ev_bal['ci']['pr_auc'][0]:.4f}, {ev_bal['ci']['pr_auc'][1]:.4f}]",
    "PR Multiplier": f"{ev_bal['point']['pr_auc']/base_val_rate:.2f}x",
    "Brier Score": f"{ev_bal['point']['brier']:.4f}",
    "Hyperparameters / Notes": "C=0.01 (C=1.0 yields identical 0.7437/0.0639)",
})

# ── 4. Champion v1 (39 Features, Val-tuned) ────────────────────────────────
booster_v1 = lgb.Booster(model_file="models/baselines/lgbm_champion_v1.txt")
p_v1 = booster_v1.predict(X_va.values)
ev_v1 = evaluate_predictions(y_va, p_v1, learner_keys=keys_va, n_resamples=n_resamples)
records.append({
    "Model": "Champion v1 LightGBM",
    "Features": "39 continuous features",
    "ROC-AUC": f"{ev_v1['point']['roc_auc']:.4f} [{ev_v1['ci']['roc_auc'][0]:.4f}, {ev_v1['ci']['roc_auc'][1]:.4f}]",
    "PR-AUC": f"{ev_v1['point']['pr_auc']:.4f} [{ev_v1['ci']['pr_auc'][0]:.4f}, {ev_v1['ci']['pr_auc'][1]:.4f}]",
    "PR Multiplier": f"{ev_v1['point']['pr_auc']/base_val_rate:.2f}x",
    "Brier Score": f"{ev_v1['point']['brier']:.4f}",
    "Hyperparameters / Notes": "285 trees, max_depth=4, num_leaves=93",
})

# ── 5. Official Champion v3 (38 Features, Train-Only Tuned Official Baseline)
# Train-only tuned parameters:
official_params = {
    "num_leaves": 50, "max_depth": 3, "learning_rate": 0.06664,
    "min_child_samples": 352, "feature_fraction": 0.6073, "bagging_fraction": 0.8478,
    "bagging_freq": 1, "lambda_l1": 8.408, "lambda_l2": 8.203,
    "n_estimators": 463, "scale_pos_weight": 1.0, "random_state": 42,
    "verbosity": -1, "objective": "binary"
}
feats_v3 = [c for c in X_tr.columns if c != "n_skipped_resources"]
clf_v3 = lgb.LGBMClassifier(**official_params)
clf_v3.fit(X_tr[feats_v3], y_tr)
p_v3 = clf_v3.predict_proba(X_va[feats_v3])[:, 1]
ev_v3 = evaluate_predictions(y_va, p_v3, learner_keys=keys_va, n_resamples=n_resamples)
records.append({
    "Model": "Official Champion v3 (Official Baseline)",
    "Features": "38 features (39 minus n_skipped_resources)",
    "ROC-AUC": f"{ev_v3['point']['roc_auc']:.4f} [{ev_v3['ci']['roc_auc'][0]:.4f}, {ev_v3['ci']['roc_auc'][1]:.4f}]",
    "PR-AUC": f"{ev_v3['point']['pr_auc']:.4f} [{ev_v3['ci']['pr_auc'][0]:.4f}, {ev_v3['ci']['pr_auc'][1]:.4f}]",
    "PR Multiplier": f"{ev_v3['point']['pr_auc']/base_val_rate:.2f}x",
    "Brier Score": f"{ev_v3['point']['brier']:.4f}",
    "Hyperparameters / Notes": "Train-only tuned (max_depth=3, leaves=50, n_est=463)",
})

# Save official v3 model
clf_v3.booster_.save_model("models/baselines/lgbm_champion_v3_official.txt")

df_comparison = pd.DataFrame(records)
print(df_comparison.to_string(index=False))
df_comparison.to_csv("reports/phase2/model_comparison_table.csv", index=False)

# ── 6. Paired Learner-Cluster Bootstrap Block (v3 vs 39-feat LR) ───────────
print("\n" + "=" * 75)
print("PAIRED LEARNER-CLUSTER BOOTSTRAP (Official Champion v3 minus 39-feat LR)")
print("=" * 75)
codes, uniques = pd.factorize(keys_va)
n_clusters = len(uniques)
rng = np.random.default_rng(42)
all_sampled = rng.integers(0, n_clusters, size=(500, n_clusters))

def paired_worker(b):
    counts = np.bincount(all_sampled[b], minlength=n_clusters)
    w = counts[codes]
    sw_pos = float(np.dot(w.astype(np.float64), y_va.astype(np.float64)))
    sw_tot = float(w.sum())
    if sw_pos == 0.0 or sw_pos == sw_tot:
        return None
    r_v3 = roc_auc_score(y_va, p_v3, sample_weight=w)
    pr_v3 = average_precision_score(y_va, p_v3, sample_weight=w)
    r_uw = roc_auc_score(y_va, p_lr_uw, sample_weight=w)
    pr_uw = average_precision_score(y_va, p_lr_uw, sample_weight=w)
    r_bal = roc_auc_score(y_va, p_lr_bal, sample_weight=w)
    pr_bal = average_precision_score(y_va, p_lr_bal, sample_weight=w)
    return {
        "d_roc_uw": r_v3 - r_uw, "d_pr_uw": pr_v3 - pr_uw,
        "d_roc_bal": r_v3 - r_bal, "d_pr_bal": pr_v3 - pr_bal
    }

paired_samples = Parallel(n_jobs=4, prefer="threads")(delayed(paired_worker)(b) for b in range(500))
paired_samples = [s for s in paired_samples if s is not None]

d_roc_uw = [s["d_roc_uw"] for s in paired_samples]
d_pr_uw = [s["d_pr_uw"] for s in paired_samples]
d_roc_bal = [s["d_roc_bal"] for s in paired_samples]
d_pr_bal = [s["d_pr_bal"] for s in paired_samples]

print(f"Official Champion v3 minus LR-Unweighted:")
print(f"  Delta ROC-AUC: {np.mean(d_roc_uw):+.4f} [95% CI: {np.percentile(d_roc_uw, 2.5):+.4f}, {np.percentile(d_roc_uw, 97.5):+.4f}]")
print(f"  Delta PR-AUC:  {np.mean(d_pr_uw):+.4f} [95% CI: {np.percentile(d_pr_uw, 2.5):+.4f}, {np.percentile(d_pr_uw, 97.5):+.4f}]")
print(f"Official Champion v3 minus LR-Balanced:")
print(f"  Delta ROC-AUC: {np.mean(d_roc_bal):+.4f} [95% CI: {np.percentile(d_roc_bal, 2.5):+.4f}, {np.percentile(d_roc_bal, 97.5):+.4f}]")
print(f"  Delta PR-AUC:  {np.mean(d_pr_bal):+.4f} [95% CI: {np.percentile(d_pr_bal, 2.5):+.4f}, {np.percentile(d_pr_bal, 97.5):+.4f}]")

# ── 7. Multi-Horizon Frozen Thresholds for Champion v3 ────────────────────
print("\n" + "=" * 75)
print("FREEZING ONE THRESHOLD PER k (k=2, 3, 4) FOR OFFICIAL CHAMPION v3")
print("=" * 75)
horizon_records = []
frozen_thresholds_v3 = {}

for k in [2, 3, 4]:
    X_va_k, y_va_k, _ = load_split("val", k=k, min_week=2)
    X_tr_k, y_tr_k, _ = load_split("train", k=k, min_week=2)
    feats_k = [c for c in X_tr_k.columns if c != "n_skipped_resources"]
    
    clf_k = lgb.LGBMClassifier(**official_params)
    clf_k.fit(X_tr_k[feats_k], y_tr_k)
    p_k = clf_k.predict_proba(X_va_k[feats_k])[:, 1]
    
    r_k = roc_auc_score(y_va_k, p_k)
    pr_k = average_precision_score(y_va_k, p_k)
    base_k = float(y_va_k.mean())
    
    # Sweep threshold for MCC
    best_th, best_mcc = 0.025, -1
    for th in np.linspace(0.01, 0.40, 80):
        pred_bin = (p_k >= th).astype(int)
        mcc = float(matthews_corrcoef(y_va_k, pred_bin))
        if mcc > best_mcc:
            best_mcc, best_th = mcc, float(th)
            
    frozen_thresholds_v3[f"k{k}"] = {
        "horizon": k, "threshold": round(best_th, 4), "mcc": round(best_mcc, 4)
    }
    horizon_records.append({
        "horizon_k": k, "n_val_rows": len(y_va_k), "base_rate": base_k,
        "roc_auc": r_k, "pr_auc": pr_k, "pr_mult": pr_k / base_k,
        "frozen_threshold": best_th, "mcc_at_frozen_threshold": best_mcc,
    })
    print(f"  k={k}: ROC={r_k:.4f}, PR={pr_k:.4f} ({pr_k/base_k:.2f}x base) -> Frozen th={best_th:.3f} (MCC={best_mcc:.4f})")

df_horizon_v3 = pd.DataFrame(horizon_records)
df_horizon_v3.to_csv("reports/phase2/by_horizon.csv", index=False)

# Update models/baselines/threshold_config.yaml
with open("models/baselines/threshold_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

cfg["official_champion_v3"] = {
    "model_file": "models/baselines/lgbm_champion_v3_official.txt",
    "n_features": 38,
    "excluded_feature": "n_skipped_resources",
    "hyperparameters": official_params,
    "frozen_thresholds_per_k": frozen_thresholds_v3,
    "mcc_optimal_policy": {
        "threshold": frozen_thresholds_v3["k3"]["threshold"],
        "mcc": frozen_thresholds_v3["k3"]["mcc"],
        "note": "MCC 0.1411 attached to MCC-optimal policy on validation cohort",
    }
}
with open("models/baselines/threshold_config.yaml", "w") as f:
    yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)

print("\nSaved threshold configuration to models/baselines/threshold_config.yaml")
print("Saved comparison table to reports/phase2/model_comparison_table.csv")
print("Saved horizon table to reports/phase2/by_horizon.csv")
