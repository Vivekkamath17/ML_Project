"""
Script to compute all remaining Checkpoint B items:
1. Onset target baselines:
   - LightGBM vs recency-only vs weekly_clicks-only on exact same onset rows.
   - By-week AUC curves for all 3.
   - Metrics excluding is_post_assessment_period rows.
2. Thresholds:
   - Cost curve for r=5.
   - Calibration curve/table at high probabilities.
   - Freeze one threshold per k (k=2, 3, 4) in threshold_config.yaml.
   - Lead-time metrics: flagged >= 1 and >= 2 weeks before unregistration, median weeks of warning.
   - Share of all learners flagged at least once for each policy.
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
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss, matthews_corrcoef

from src.models.data import load_split
from src.models.metrics import evaluate_predictions

# ── 1. ONSET TARGET BASELINES & POST-ASSESSMENT SLICING ──────────────────────
print("=" * 70)
print("[1/2] Onset Target (label_inactive_k3) Baselines & Post-Assessment Slicing")
print("=" * 70)

X_tr_on, y_tr_on, meta_tr_on = load_split("train", k=3, min_week=2, target_col="label_inactive_k3")
X_va_on, y_va_on, meta_va_on = load_split("val", k=3, min_week=2, target_col="label_inactive_k3")
keys_va_on = meta_va_on["code_module"] + "_" + meta_va_on["code_presentation"] + "_" + meta_va_on["id_student"].astype(str)

# LightGBM champion on onset
params_onset = {
    "num_leaves": 93, "max_depth": 4, "learning_rate": 0.0306, "min_child_samples": 363,
    "feature_fraction": 0.932, "bagging_fraction": 0.803, "bagging_freq": 1,
    "lambda_l1": 8.07, "lambda_l2": 2.83, "scale_pos_weight": 1.0, "n_estimators": 1000,
    "random_state": 42, "verbosity": -1, "objective": "binary"
}
clf_lgb_on = lgb.LGBMClassifier(**params_onset)
clf_lgb_on.fit(X_tr_on, y_tr_on, eval_X=X_va_on, eval_y=y_va_on, callbacks=[lgb.early_stopping(50, verbose=False)])
p_lgb_on = clf_lgb_on.predict_proba(X_va_on)[:, 1]

# Recency-only baseline (days_since_last_activity)
sc_rec = StandardScaler().fit(X_tr_on[["days_since_last_activity"]].fillna(0.0))
lr_rec = LogisticRegression(C=1.0, random_state=42).fit(sc_rec.transform(X_tr_on[["days_since_last_activity"]].fillna(0.0)), y_tr_on)
p_rec_on = lr_rec.predict_proba(sc_rec.transform(X_va_on[["days_since_last_activity"]].fillna(0.0)))[:, 1]

# Weekly-clicks-only baseline (weekly_clicks)
sc_clk = StandardScaler().fit(X_tr_on[["weekly_clicks"]].fillna(0.0))
lr_clk = LogisticRegression(C=1.0, random_state=42).fit(sc_clk.transform(X_tr_on[["weekly_clicks"]].fillna(0.0)), y_tr_on)
p_clk_on = lr_clk.predict_proba(sc_clk.transform(X_va_on[["weekly_clicks"]].fillna(0.0)))[:, 1]

# Overall metrics
base_on = float(y_va_on.mean())
print(f"Overall Onset Performance (n={len(y_va_on):,}, Base Rate = {base_on*100:.2f}%):")
print(f"  LightGBM (39 feat):    ROC-AUC = {roc_auc_score(y_va_on, p_lgb_on):.4f}, PR-AUC = {average_precision_score(y_va_on, p_lgb_on):.4f} ({average_precision_score(y_va_on, p_lgb_on)/base_on:.2f}x)")
print(f"  Recency-Only LR:       ROC-AUC = {roc_auc_score(y_va_on, p_rec_on):.4f}, PR-AUC = {average_precision_score(y_va_on, p_rec_on):.4f} ({average_precision_score(y_va_on, p_rec_on)/base_on:.2f}x)")
print(f"  Weekly-Clicks-Only LR: ROC-AUC = {roc_auc_score(y_va_on, p_clk_on):.4f}, PR-AUC = {average_precision_score(y_va_on, p_clk_on):.4f} ({average_precision_score(y_va_on, p_clk_on)/base_on:.2f}x)")

# Excluding is_post_assessment_period rows
mask_active_period = ~meta_va_on["is_post_assessment_period"].astype(bool)
y_va_active = y_va_on[mask_active_period]
p_lgb_active = p_lgb_on[mask_active_period]
p_rec_active = p_rec_on[mask_active_period]
p_clk_active = p_clk_on[mask_active_period]
base_active = float(y_va_active.mean())

print(f"\nExcluding Post-Assessment Period Rows (Active Assessment Period Only, n={len(y_va_active):,}, Base Rate = {base_active*100:.2f}%):")
print(f"  LightGBM (39 feat):    ROC-AUC = {roc_auc_score(y_va_active, p_lgb_active):.4f}, PR-AUC = {average_precision_score(y_va_active, p_lgb_active):.4f} ({average_precision_score(y_va_active, p_lgb_active)/base_active:.2f}x)")
print(f"  Recency-Only LR:       ROC-AUC = {roc_auc_score(y_va_active, p_rec_active):.4f}, PR-AUC = {average_precision_score(y_va_active, p_rec_active):.4f} ({average_precision_score(y_va_active, p_rec_active)/base_active:.2f}x)")
print(f"  Weekly-Clicks-Only LR: ROC-AUC = {roc_auc_score(y_va_active, p_clk_active):.4f}, PR-AUC = {average_precision_score(y_va_active, p_clk_active):.4f} ({average_precision_score(y_va_active, p_clk_active)/base_active:.2f}x)")

# By-week AUC curves for all 3
print("\nOnset Target By-Week ROC-AUC & PR-AUC (Sample Weeks):")
print(f"{'Week':<6} {'n':>6} {'Base':>7} | {'LGBM ROC':>8} {'LGBM PR':>8} | {'Rec ROC':>8} {'Rec PR':>8} | {'Clk ROC':>8} {'Clk PR':>8}")
onset_by_week = []
for w in sorted(meta_va_on["week"].unique()):
    mask_w = meta_va_on["week"] == w
    if mask_w.sum() > 50 and y_va_on[mask_w].sum() > 0:
        y_w = y_va_on[mask_w]
        pl_w = p_lgb_on[mask_w]
        pr_w = p_rec_on[mask_w]
        pc_w = p_clk_on[mask_w]
        bw = float(y_w.mean())
        roc_l = float(roc_auc_score(y_w, pl_w))
        pr_l = float(average_precision_score(y_w, pl_w))
        roc_r = float(roc_auc_score(y_w, pr_w))
        pr_r = float(average_precision_score(y_w, pr_w))
        roc_c = float(roc_auc_score(y_w, pc_w))
        pr_c = float(average_precision_score(y_w, pc_w))
        onset_by_week.append({
            "week": int(w), "n": int(mask_w.sum()), "base": bw,
            "lgb_roc": roc_l, "lgb_pr": pr_l,
            "rec_roc": roc_r, "rec_pr": pr_r,
            "clk_roc": roc_c, "clk_pr": pr_c,
        })
        if w in [2, 4, 8, 12, 16, 20, 24, 28]:
            print(f"{w:<6d} {mask_w.sum():>6d} {bw*100:>6.2f}% | {roc_l:>8.4f} {pr_l:>8.4f} | {roc_r:>8.4f} {pr_r:>8.4f} | {roc_c:>8.4f} {pr_c:>8.4f}")


# ── 2. THRESHOLDS: COST CURVE r=5, CALIBRATION, LEAD TIME, MULTI-K FREEZE ───
print("\n" + "=" * 70)
print("[2/2] Thresholds: Cost Curve r=5, Calibration, Lead-Time & Multi-k Freeze")
print("=" * 70)

X_va_k3, y_va_k3, meta_va_k3 = load_split("val", k=3, min_week=2)
booster_k3 = lgb.Booster(model_file="models/baselines/lgbm_champion_v1.txt")
probs_k3 = booster_k3.predict(X_va_k3.values)

# Cost curve for r=5 across threshold sweep
thresholds = np.linspace(0.01, 0.40, 40)
cost_curve_r5 = []
for th in thresholds:
    pred = (probs_k3 >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va_k3, pred, labels=[0, 1]).ravel()
    cost = 5.0 * fn + 1.0 * fp
    cost_curve_r5.append({"threshold": round(th, 3), "cost": cost, "fn": fn, "fp": fp, "tp": tp, "prec": tp/(tp+fp) if tp+fp>0 else 0, "rec": tp/(tp+fn)})

df_cost_r5 = pd.DataFrame(cost_curve_r5)
min_cost_idx = df_cost_r5["cost"].idxmin()
print("Cost Curve around minimum for r=5 (Missed Dropout = 5x Wasted Outreach):")
print(df_cost_r5.iloc[max(0, min_cost_idx-3):min(len(df_cost_r5), min_cost_idx+4)].to_string(index=False))

# Calibration at high probabilities
print("\nCalibration at High Predicted Probabilities:")
bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
calib_records = []
for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    mask = (probs_k3 >= low) & (probs_k3 < high)
    n_in_bin = mask.sum()
    if n_in_bin > 0:
        obs_rate = float(y_va_k3[mask].mean())
        mean_pred = float(probs_k3[mask].mean())
        calib_records.append({"bin": f"[{low:.2f}, {high:.2f})", "n": int(n_in_bin), "mean_pred": mean_pred, "observed_rate": obs_rate})
        print(f"  Bin [{low:.2f}, {high:.2f}): n={n_in_bin:>6d} | Mean Pred = {mean_pred*100:>5.2f}% | Observed Rate = {obs_rate*100:>5.2f}%")

# Lead-time metrics: for true dropouts, weeks of advance warning before date_unregistration
# Load learner unregistration dates from raw studentRegistration
raw_reg = pd.read_csv("data/raw/studentRegistration.csv")
meta_va_k3 = meta_va_k3.merge(
    raw_reg[["code_module", "code_presentation", "id_student", "date_unregistration"]],
    on=["code_module", "code_presentation", "id_student"],
    how="left"
)
meta_va_k3["prob"] = probs_k3
meta_va_k3["learner_key"] = meta_va_k3["code_module"] + "_" + meta_va_k3["code_presentation"] + "_" + meta_va_k3["id_student"].astype(str)

# True dropouts with non-null unregistration week
meta_va_k3["unreg_week"] = meta_va_k3["date_unregistration"] // 7
withdrawn_learners = meta_va_k3[meta_va_k3["date_unregistration"].notna()]["learner_key"].unique()
total_withdrawn_learners = len(withdrawn_learners)

def compute_policy_lead_time(flag_mask, policy_name):
    flagged_df = meta_va_k3[flag_mask]
    # First week flagged for each learner
    first_flag = flagged_df.groupby("learner_key").agg({"week": "min", "unreg_week": "first", "date_unregistration": "first"}).reset_index()
    
    # Unique share of ALL learners flagged
    all_learners = meta_va_k3["learner_key"].unique()
    share_all_flagged = len(first_flag) / len(all_learners)
    
    # Lead time for true withdrawn learners
    first_flag_withdrawn = first_flag[first_flag["date_unregistration"].notna()].copy()
    first_flag_withdrawn["lead_weeks"] = first_flag_withdrawn["unreg_week"] - first_flag_withdrawn["week"]
    
    n_withdrawn_flagged = len(first_flag_withdrawn)
    share_withdrawn_flagged = n_withdrawn_flagged / total_withdrawn_learners
    
    pct_ge_1w = float((first_flag_withdrawn["lead_weeks"] >= 1).mean()) if n_withdrawn_flagged > 0 else 0.0
    pct_ge_2w = float((first_flag_withdrawn["lead_weeks"] >= 2).mean()) if n_withdrawn_flagged > 0 else 0.0
    med_lead = float(first_flag_withdrawn["lead_weeks"].median()) if n_withdrawn_flagged > 0 else 0.0
    
    return {
        "policy": policy_name,
        "share_all_learners_flagged": share_all_flagged,
        "share_withdrawn_flagged": share_withdrawn_flagged,
        "flagged_ge_1w": pct_ge_1w,
        "flagged_ge_2w": pct_ge_2w,
        "median_lead_weeks": med_lead,
    }

lead_mcc = compute_policy_lead_time(meta_va_k3["prob"] >= 0.025, "MCC-Optimal (th=0.025)")
lead_r5 = compute_policy_lead_time(meta_va_k3["prob"] >= 0.265, "Cost r=5 (th=0.265)")
lead_r10 = compute_policy_lead_time(meta_va_k3["prob"] >= 0.095, "Cost r=10 (th=0.095)")

# Capacity top 10% flag mask
top10_flags = pd.Series(False, index=meta_va_k3.index)
for w in meta_va_k3["week"].unique():
    w_idx = meta_va_k3[meta_va_k3["week"] == w].index
    w_sub = meta_va_k3.loc[w_idx].sort_values(by=["prob", "id_student"], ascending=[False, True])
    n_top = int(np.ceil(0.10 * len(w_sub)))
    top_indices = w_sub.iloc[:n_top].index
    top10_flags.loc[top_indices] = True

lead_cap10 = compute_policy_lead_time(top10_flags, "Capacity Top-10% / Week")

print("\nLead-Time & Warning Metrics for Withdrawn Learners:")
df_leads = pd.DataFrame([lead_mcc, lead_r10, lead_r5, lead_cap10])
print(df_leads.to_string(index=False))

# Freeze one threshold per k (k=2, 3, 4) in threshold_config.yaml
# Load k=2 and k=4 predictions
X_va_k2, y_va_k2, m_va_k2 = load_split("val", k=2, min_week=2)
X_tr_k2, y_tr_k2, _ = load_split("train", k=2, min_week=2)
clf_k2 = lgb.LGBMClassifier(**params_onset).fit(X_tr_k2, y_tr_k2, eval_X=X_va_k2, eval_y=y_va_k2, callbacks=[lgb.early_stopping(50, verbose=False)])
p_k2 = clf_k2.predict_proba(X_va_k2)[:, 1]

X_va_k4, y_va_k4, m_va_k4 = load_split("val", k=4, min_week=2)
X_tr_k4, y_tr_k4, _ = load_split("train", k=4, min_week=2)
clf_k4 = lgb.LGBMClassifier(**params_onset).fit(X_tr_k4, y_tr_k4, eval_X=X_va_k4, eval_y=y_va_k4, callbacks=[lgb.early_stopping(50, verbose=False)])
p_k4 = clf_k4.predict_proba(X_va_k4)[:, 1]

def find_mcc_opt(y_t, p_t):
    best_th, best_mcc = 0.05, -1
    for th in np.linspace(0.01, 0.40, 80):
        pred = (p_t >= th).astype(int)
        mcc = float(matthews_corrcoef(y_t, pred))
        if mcc > best_mcc:
            best_mcc, best_th = mcc, th
    return float(best_th), float(best_mcc)

th_k2, mcc_k2 = find_mcc_opt(y_va_k2, p_k2)
th_k3, mcc_k3 = 0.025, 0.1411
th_k4, mcc_k4 = find_mcc_opt(y_va_k4, p_k4)

print(f"\nFrozen MCC-Optimal Thresholds per Horizon k:")
print(f"  k=2: threshold = {th_k2:.3f} (MCC = {mcc_k2:.4f})")
print(f"  k=3: threshold = {th_k3:.3f} (MCC = {mcc_k3:.4f})")
print(f"  k=4: threshold = {th_k4:.3f} (MCC = {mcc_k4:.4f})")

# Update threshold_config.yaml
cfg_path = Path("models/baselines/threshold_config.yaml")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["frozen_thresholds_per_k"] = {
    "k2": {"threshold": th_k2, "mcc": mcc_k2, "horizon": 2},
    "k3": {"threshold": th_k3, "mcc": mcc_k3, "horizon": 3},
    "k4": {"threshold": th_k4, "mcc": mcc_k4, "horizon": 4},
}
cfg["lead_time_metrics"] = df_leads.to_dict(orient="records")

with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)

print(f"\nSuccessfully updated {cfg_path}")
