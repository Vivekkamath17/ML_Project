"""Script to assemble CHECKPOINT A table."""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from src.models.data import load_split, load_feature_manifest
from src.models.metrics import evaluate_predictions, compute_point_metrics

X_tr, y_tr, meta_tr = load_split("train")
X_va, y_va, meta_va = load_split("val")
keys_tr = meta_tr["code_module"] + "_" + meta_tr["code_presentation"] + "_" + meta_tr["id_student"].astype(str)
keys_va = meta_va["code_module"] + "_" + meta_va["code_presentation"] + "_" + meta_va["id_student"].astype(str)

n_resamples = 500

# ── 1. Majority Prior ──────────────────────────────────────────────────────
prior_probs = np.full(len(y_va), float(y_tr.mean()))
prior_ev = evaluate_predictions(y_va, prior_probs, learner_keys=keys_va, n_resamples=0)

# ── 2. Recency-only LR (days_since_last_activity) ─────────────────────────
feat = "days_since_last_activity"
feat_tr = X_tr[[feat]].fillna(X_tr[[feat]].median())
feat_va = X_va[[feat]].fillna(X_tr[[feat]].median())
sc = StandardScaler().fit(feat_tr)
recency_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
recency_lr.fit(sc.transform(feat_tr), y_tr)
recency_probs = recency_lr.predict_proba(sc.transform(feat_va))[:, 1]
recency_ev = evaluate_predictions(y_va, recency_probs, learner_keys=keys_va, n_resamples=n_resamples)

# ── 3. One-feature baselines ───────────────────────────────────────────────
one_feat_names = ["weekly_clicks", "weighted_score_to_date", "pct_resources_visited",
                  "wow_change_clicks", "num_of_prev_attempts"]
one_feat_ev = {}
for f in one_feat_names:
    xtr = X_tr[[f]].fillna(X_tr[[f]].median())
    xva = X_va[[f]].fillna(X_tr[[f]].median())
    sc2 = StandardScaler().fit(xtr)
    lr2 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr2.fit(sc2.transform(xtr), y_tr)
    pv = lr2.predict_proba(sc2.transform(xva))[:, 1]
    one_feat_ev[f] = evaluate_predictions(y_va, pv, learner_keys=keys_va, n_resamples=0)

# ── 4. 39-feature LR (unweighted C=0.1 and balanced C=1.0) ───────────────
from src.models.train_lr import prepare_lr_matrices
X_tr_lr, X_va_lr, lr_feat_names = prepare_lr_matrices(X_tr, X_va)

lr_uw = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver="lbfgs")
lr_uw.fit(X_tr_lr, y_tr)
pv_uw = lr_uw.predict_proba(X_va_lr)[:, 1]
lr_uw_ev = evaluate_predictions(y_va, pv_uw, learner_keys=keys_va, n_resamples=n_resamples)

lr_bal = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs", class_weight="balanced")
lr_bal.fit(X_tr_lr, y_tr)
pv_bal = lr_bal.predict_proba(X_va_lr)[:, 1]
lr_bal_ev = evaluate_predictions(y_va, pv_bal, learner_keys=keys_va, n_resamples=n_resamples)

# ── 5. LightGBM champion ──────────────────────────────────────────────────
booster = lgb.Booster(model_file="models/baselines/lgbm_champion_v1.txt")
champ_probs_va = booster.predict(X_va.values)
champ_probs_tr = booster.predict(X_tr.values)
champ_ev = evaluate_predictions(y_va, champ_probs_va, learner_keys=keys_va, n_resamples=n_resamples)
# Train point only (no CI needed for sanity)
train_pt = compute_point_metrics(y_tr, champ_probs_tr)

# ─── PRINT CHECKPOINT A TABLE ─────────────────────────────────────────────
def fmt_ci(ev, metric):
    ci = ev["ci"].get(metric, (0.0, 0.0))
    if ci == (0.0, 0.0):
        return "   n/a   "
    return f"[{ci[0]:.4f},{ci[1]:.4f}]"

def row(name, ev):
    pt = ev["point"]
    roc = f"{pt['roc_auc']:.4f} {fmt_ci(ev,'roc_auc')}"
    pr  = f"{pt['pr_auc']:.4f} ({pt['pr_auc_mult']:.2f}x) {fmt_ci(ev,'pr_auc')}"
    mcc = f"{pt['mcc']:.4f}"
    bri = f"{pt['brier']:.4f}"
    return name, roc, pr, mcc, bri

print("=" * 110)
print("CHECKPOINT A — Model Comparison Table (Val split = 2014B, base rate = 2.37%)")
print("=" * 110)
header = f"{'Model':<35} {'ROC-AUC [95% CI]':<30} {'PR-AUC (xBase) [95% CI]':<40} {'MCC':<8} {'Brier'}"
print(header)
print("-" * 110)

rows_data = [
    row("Majority Prior",          prior_ev),
    row("Recency-only LR",         recency_ev),
]
for f in one_feat_names:
    rows_data.append(row(f"  one-feat: {f}", one_feat_ev[f]))
rows_data += [
    row("LR-39feat Unweighted C=0.1", lr_uw_ev),
    row("LR-39feat Balanced   C=1.0", lr_bal_ev),
    row("LightGBM Champion (Optuna)", champ_ev),
]
for name, roc, pr, mcc, bri in rows_data:
    print(f"{name:<35} {roc:<30} {pr:<40} {mcc:<8} {bri}")

# ─── CHAMPION HYPERPARAMETERS ─────────────────────────────────────────────
best_params = {
    "num_leaves": 93, "max_depth": 4,
    "learning_rate": 0.0306, "min_child_samples": 363,
    "feature_fraction": 0.932, "bagging_fraction": 0.803,
    "lambda_l1": 8.07, "lambda_l2": 2.83,
    "scale_pos_weight": 1.0, "n_estimators": 1500,
}
print("\n" + "=" * 60)
print("CHAMPION HYPERPARAMETERS")
print("=" * 60)
for k, v in best_params.items():
    print(f"  {k:<25} = {v}")

# ─── TOP 15 FEATURES BY GAIN ──────────────────────────────────────────────
feat_names = list(X_va.columns)
gains = booster.feature_importance(importance_type="gain")
gain_df = pd.DataFrame({"feature": feat_names, "gain": gains}).sort_values("gain", ascending=False)
total_gain = gain_df["gain"].sum()
gain_df["pct_gain"] = (gain_df["gain"] / total_gain * 100).round(2)
print("\n" + "=" * 60)
print("TOP 15 FEATURES BY GAIN")
print("=" * 60)
print(f"{'Rank':<6} {'Feature':<35} {'Gain':>12} {'Pct%':>8}")
for i, r2 in enumerate(gain_df.head(15).itertuples(), 1):
    print(f"{i:<6} {r2.feature:<35} {r2.gain:>12.1f} {r2.pct_gain:>7.2f}%")

# ─── SANITY CHECKS ────────────────────────────────────────────────────────
champ_pt = champ_ev["point"]
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)
roc_ok = champ_pt["roc_auc"] < 0.90
top_ok  = gain_df.iloc[0]["pct_gain"] < 50.0
gap_ok  = abs(champ_pt["roc_auc"] - train_pt["roc_auc"]) < 0.08
print(f"  Val ROC-AUC={champ_pt['roc_auc']:.4f}  {'PASS (<0.90)' if roc_ok else 'FAIL: >0.90!'}")
print(f"  Top feat gain={gain_df.iloc[0]['pct_gain']:.2f}%  {'PASS (<50%)' if top_ok else 'FAIL: >50%!'}")
print(f"  Val-Train ROC gap={champ_pt['roc_auc'] - train_pt['roc_auc']:+.4f}  {'PASS (|gap|<0.08)' if gap_ok else 'WARN: large gap!'}")
print(f"  Train ROC-AUC={train_pt['roc_auc']:.4f}  Val ROC-AUC={champ_pt['roc_auc']:.4f}")
print(f"  Best iter (n_estimators used): stored in champion model")
print()
all_pass = roc_ok and top_ok and gap_ok
print(f"  OVERALL: {'ALL SANITY CHECKS PASS' if all_pass else 'SOME WARNINGS — see above'}")
