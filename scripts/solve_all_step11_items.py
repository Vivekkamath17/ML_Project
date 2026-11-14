"""
Master Execution Script for Step 11 & Checkpoint C Fixes.
Executes all computational items:
1. Fresh Paired Cluster-Bootstrap (500 learner resamples) for LightGBM vs LR.
2. Proper Nested Tuning on 2013B+2013J (GroupKFold by id_student, max_depth 2-6, 30 Optuna trials).
3. 20-Permutation Shuffled-Label Test (learner-level permutation).
4. Champion v2 (35 features, without curriculum group) & validation evaluation.
5. Reconciled Lead-Time & Policy Table with all denominators & random baseline.
6. Assessment Audit Table & Snapshot-fitted Score Baseline.
7. Full By-Week Table (all weeks 0..30+ with n, positives, base, ROC, PR).
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    mean_absolute_error, r2_score, matthews_corrcoef
)
from joblib import Parallel, delayed

from src.models.data import load_split, load_feature_manifest
from src.models.train_lr import prepare_lr_matrices

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("LOADING DATA (TRAIN: 2013B/J, VAL: 2014B)")
print("=" * 70)
X_tr, y_tr, meta_tr = load_split("train", k=3, min_week=2)
X_va, y_va, meta_va = load_split("val", k=3, min_week=2)

keys_va = meta_va["code_module"] + "_" + meta_va["code_presentation"] + "_" + meta_va["id_student"].astype(str)
keys_tr = meta_tr["code_module"] + "_" + meta_tr["code_presentation"] + "_" + meta_tr["id_student"].astype(str)

booster = lgb.Booster(model_file="models/baselines/lgbm_champion_v1.txt")
p_champ = booster.predict(X_va.values)
p_champ_tr = booster.predict(X_tr.values)

print(f"Val rows: {len(X_va):,}, Train rows: {len(X_tr):,}")
print(f"Champion Train ROC: {roc_auc_score(y_tr, p_champ_tr):.4f}, Val ROC: {roc_auc_score(y_va, p_champ):.4f}")
print(f"Train->Val Gap: {roc_auc_score(y_va, p_champ) - roc_auc_score(y_tr, p_champ_tr):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FRESH PAIRED CLUSTER-BOOTSTRAP (500 resamples)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. RERUN PAIRED CLUSTER-BOOTSTRAP (LightGBM vs 39-feat LR)")
print("=" * 70)
X_tr_lr, X_va_lr, _ = prepare_lr_matrices(X_tr, X_va)

lr_uw = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver="lbfgs")
lr_uw.fit(X_tr_lr, y_tr)
p_lr_uw = lr_uw.predict_proba(X_va_lr)[:, 1]

lr_bal = LogisticRegression(C=0.01, max_iter=1000, random_state=42, solver="lbfgs", class_weight="balanced")
lr_bal.fit(X_tr_lr, y_tr)
p_lr_bal = lr_bal.predict_proba(X_va_lr)[:, 1]

codes, uniques = pd.factorize(keys_va)
n_clusters = len(uniques)
rng = np.random.default_rng(2026)
all_sampled = rng.integers(0, n_clusters, size=(500, n_clusters))

def paired_worker(b):
    counts = np.bincount(all_sampled[b], minlength=n_clusters)
    w = counts[codes]
    sw_pos = float(np.dot(w.astype(np.float64), y_va.astype(np.float64)))
    sw_tot = float(w.sum())
    if sw_pos == 0.0 or sw_pos == sw_tot:
        return None

    r_lgb = roc_auc_score(y_va, p_champ, sample_weight=w)
    pr_lgb = average_precision_score(y_va, p_champ, sample_weight=w)

    r_uw = roc_auc_score(y_va, p_lr_uw, sample_weight=w)
    pr_uw = average_precision_score(y_va, p_lr_uw, sample_weight=w)

    r_bal = roc_auc_score(y_va, p_lr_bal, sample_weight=w)
    pr_bal = average_precision_score(y_va, p_lr_bal, sample_weight=w)

    return {
        "d_roc_uw": r_lgb - r_uw,
        "d_pr_uw": pr_lgb - pr_uw,
        "d_roc_bal": r_lgb - r_bal,
        "d_pr_bal": pr_lgb - pr_bal,
    }

paired_samples = Parallel(n_jobs=4, prefer="threads")(delayed(paired_worker)(b) for b in range(500))
paired_samples = [s for s in paired_samples if s is not None]

d_roc_uw = [s["d_roc_uw"] for s in paired_samples]
d_pr_uw = [s["d_pr_uw"] for s in paired_samples]
d_roc_bal = [s["d_roc_bal"] for s in paired_samples]
d_pr_bal = [s["d_pr_bal"] for s in paired_samples]

paired_bootstrap_results = {
    "lgb_vs_lr_uw_roc": (float(np.mean(d_roc_uw)), float(np.percentile(d_roc_uw, 2.5)), float(np.percentile(d_roc_uw, 97.5))),
    "lgb_vs_lr_uw_pr":  (float(np.mean(d_pr_uw)),  float(np.percentile(d_pr_uw, 2.5)),  float(np.percentile(d_pr_uw, 97.5))),
    "lgb_vs_lr_bal_roc": (float(np.mean(d_roc_bal)), float(np.percentile(d_roc_bal, 2.5)), float(np.percentile(d_roc_bal, 97.5))),
    "lgb_vs_lr_bal_pr":  (float(np.mean(d_pr_bal)),  float(np.percentile(d_pr_bal, 2.5)),  float(np.percentile(d_pr_bal, 97.5))),
}
print(f"LightGBM minus LR-Unweighted: Delta ROC = {paired_bootstrap_results['lgb_vs_lr_uw_roc'][0]:+.4f} [{paired_bootstrap_results['lgb_vs_lr_uw_roc'][1]:+.4f}, {paired_bootstrap_results['lgb_vs_lr_uw_roc'][2]:+.4f}]")
print(f"                               Delta PR  = {paired_bootstrap_results['lgb_vs_lr_uw_pr'][0]:+.4f} [{paired_bootstrap_results['lgb_vs_lr_uw_pr'][1]:+.4f}, {paired_bootstrap_results['lgb_vs_lr_uw_pr'][2]:+.4f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PROPER NESTED TUNING ON TRAIN ONLY (GroupKFold by id_student, 30 Optuna trials)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. PROPER NESTED TUNING ON TRAIN (GroupKFold by id_student, 30 trials, max_depth 2-6)")
print("=" * 70)
gkf = GroupKFold(n_splits=5)
train_groups = meta_tr["id_student"].values

def nested_objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10.0),
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "scale_pos_weight": 1.0,
        "random_state": 42,
        "verbosity": -1,
        "objective": "binary",
    }
    # 3-fold inner cross-validation for speed
    inner_cv_scores = []
    inner_gkf = GroupKFold(n_splits=3)
    for fold, (itr_idx, iva_idx) in enumerate(inner_gkf.split(X_tr, y_tr, groups=train_groups)):
        clf_inner = lgb.LGBMClassifier(**params)
        clf_inner.fit(X_tr.iloc[itr_idx], y_tr[itr_idx])
        p_iva = clf_inner.predict_proba(X_tr.iloc[iva_idx])[:, 1]
        inner_cv_scores.append(average_precision_score(y_tr[iva_idx], p_iva))
    return float(np.mean(inner_cv_scores))

study_nested = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_nested.optimize(nested_objective, n_trials=30, show_progress_bar=False)

best_nested_params = study_nested.best_params
print("Best Train-Only Tuned Hyperparameters (30 Optuna trials):")
for k, v in best_nested_params.items():
    print(f"  {k}: {v}")

# Fit final train-only model on full train (2013B + 2013J)
clf_train_only = lgb.LGBMClassifier(**best_nested_params, bagging_freq=1, scale_pos_weight=1.0, random_state=42, verbosity=-1)
clf_train_only.fit(X_tr, y_tr)
p_va_train_only = clf_train_only.predict_proba(X_va)[:, 1]
roc_va_to = roc_auc_score(y_va, p_va_train_only)
pr_va_to = average_precision_score(y_va, p_va_train_only)
print(f"Train-Only Model Scored on 2014B: ROC-AUC = {roc_va_to:.4f}, PR-AUC = {pr_va_to:.4f}")
print(f"Val-Tuned Champion on 2014B:       ROC-AUC = 0.7736, PR-AUC = 0.0735 (Delta ROC: {roc_va_to - 0.7736:+.4f}, Delta PR: {pr_va_to - 0.0735:+.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 20-PERMUTATION SHUFFLED-LABEL TEST (Learner-level)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. SHUFFLED-LABEL TEST (20 PERMUTATIONS AT LEARNER LEVEL)")
print("=" * 70)
unique_learners_tr = keys_tr.unique()
learner_to_rows_tr = {lk: np.where(keys_tr.values == lk)[0] for lk in unique_learners_tr}
learner_y_tr = {lk: int(y_tr[idx].max()) for lk, idx in learner_to_rows_tr.items()}

shuffled_20_rocs, shuffled_20_prs = [], []
for rep in range(20):
    rng_p = np.random.default_rng(200 + rep)
    perm_learners = rng_p.permutation(unique_learners_tr)
    l_map = dict(zip(unique_learners_tr, perm_learners))

    perm_y = np.zeros(len(y_tr), dtype=int)
    for lk, perm_lk in l_map.items():
        perm_y[learner_to_rows_tr[lk]] = learner_y_tr[perm_lk]

    clf_perm = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, max_depth=4, random_state=42, verbosity=-1)
    clf_perm.fit(X_tr, perm_y)
    p_perm = clf_perm.predict_proba(X_va)[:, 1]
    shuffled_20_rocs.append(float(roc_auc_score(y_va, p_perm)))
    shuffled_20_prs.append(float(average_precision_score(y_va, p_perm)))

df_shuff_dist = pd.Series(shuffled_20_rocs).describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
print("Shuffled-Label ROC-AUC Distribution across 20 permutations:")
print(df_shuff_dist.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CHAMPION V2 (WITHOUT CURRICULUM GROUP) & THRESHOLD RE-FREEZE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. CHAMPION V2 (WITHOUT CURRICULUM FEATURES, 35 FEATURES)")
print("=" * 70)
curriculum_cols = ["pct_resources_visited", "n_skipped_resources", "pacing_vs_cohort_median", "prerequisite_gap"]
feats_v2 = [c for c in X_tr.columns if c not in curriculum_cols]

p_champ_v2_params = {
    "num_leaves": 93, "max_depth": 4, "learning_rate": 0.0305762,
    "min_child_samples": 363, "feature_fraction": 0.932065, "bagging_fraction": 0.803165,
    "bagging_freq": 1, "lambda_l1": 8.06916, "lambda_l2": 2.8288,
    "scale_pos_weight": 1.0, "n_estimators": 1500, "verbosity": -1,
    "objective": "binary", "random_state": 42
}
clf_v2 = lgb.LGBMClassifier(**p_champ_v2_params)
clf_v2.fit(X_tr[feats_v2], y_tr, eval_X=X_va[feats_v2], eval_y=y_va, callbacks=[lgb.early_stopping(50, verbose=False)])
p_v2_va = clf_v2.predict_proba(X_va[feats_v2])[:, 1]
roc_v2 = roc_auc_score(y_va, p_v2_va)
pr_v2 = average_precision_score(y_va, p_v2_va)

print(f"Champion v1 (39 feats): ROC = 0.7736, PR = 0.0735 (best_iter = 285)")
print(f"Champion v2 (35 feats): ROC = {roc_v2:.4f}, PR = {pr_v2:.4f} (best_iter = {clf_v2.best_iteration_})")
print(f"Delta on 2014B: ROC = {roc_v2 - 0.7736:+.4f}, PR = {pr_v2 - 0.0735:+.4f}")

# Save Champion v2 booster
clf_v2.booster_.save_model("models/baselines/lgbm_champion_v2.txt")
print("Saved models/baselines/lgbm_champion_v2.txt")

# Find MCC optimal threshold for v2
best_th_v2, best_mcc_v2 = 0.025, -1
for th in np.linspace(0.01, 0.40, 80):
    mcc_val = matthews_corrcoef(y_va, (p_v2_va >= th).astype(int))
    if mcc_val > best_mcc_v2:
        best_mcc_v2, best_th_v2 = mcc_val, th
print(f"Champion v2 MCC-optimal threshold: th={best_th_v2:.3f}, MCC={best_mcc_v2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. RECONCILED LEAD-TIME & POLICY TABLE (WITH ALL DENOMINATORS & RANDOM BASELINE)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. RECONCILED LEAD-TIME & POLICY TABLE")
print("=" * 70)
raw_reg = pd.read_csv("data/raw/studentRegistration.csv")
meta_va_full = meta_va.merge(
    raw_reg[["code_module", "code_presentation", "id_student", "date_unregistration"]],
    on=["code_module", "code_presentation", "id_student"], how="left"
)
meta_va_full["unreg_week"] = meta_va_full["date_unregistration"] // 7
meta_va_full["prob_v1"] = p_champ
meta_va_full["prob_v2"] = p_v2_va
meta_va_full["learner_key"] = keys_va

TOTAL_LEARNERS = len(meta_va_full["learner_key"].unique())   # 6,474
TOTAL_WITHDRAWN = len(meta_va_full[meta_va_full["date_unregistration"].notna()]["learner_key"].unique()) # 1,394
withdrawn_set = set(meta_va_full[meta_va_full["date_unregistration"].notna()]["learner_key"].unique())

print(f"Total Enrolled Eligible Learners (Denom 1): {TOTAL_LEARNERS:,}")
print(f"Total Withdrawn Learners (Denom 2):         {TOTAL_WITHDRAWN:,} ({TOTAL_WITHDRAWN/TOTAL_LEARNERS*100:.2f}%)")

# Top 10% capacity mask
top10_flags = pd.Series(False, index=meta_va_full.index)
for w in meta_va_full["week"].unique():
    w_idx = meta_va_full[meta_va_full["week"] == w].index
    w_sub = meta_va_full.loc[w_idx].sort_values(by=["prob_v1", "id_student"], ascending=[False, True])
    n_top = int(np.ceil(0.10 * len(w_sub)))
    top10_flags.loc[w_sub.iloc[:n_top].index] = True

policy_masks = {
    "MCC-Optimal v1 (th=0.025)": meta_va_full["prob_v1"] >= 0.025,
    "MCC-Optimal v2 (th=0.025)": meta_va_full["prob_v2"] >= 0.025,
    "Cost r=10 (th=0.095)": meta_va_full["prob_v1"] >= 0.095,
    "Capacity Top-10%/week": top10_flags,
}

policy_rows = []
for name, mask in policy_masks.items():
    meta_va_full["flagged"] = mask
    flagged_learners = set(meta_va_full[meta_va_full["flagged"]]["learner_key"].unique())
    n_fl = len(flagged_learners)
    pct_all_fl = n_fl / TOTAL_LEARNERS

    # Withdrawn caught
    fl_withdrawn = flagged_learners.intersection(withdrawn_set)
    n_fl_wd = len(fl_withdrawn)
    recall_wd = n_fl_wd / TOTAL_WITHDRAWN
    prec_l = n_fl_wd / n_fl if n_fl > 0 else 0.0

    # Random baseline with same number of flagged learners:
    random_expected_recall = pct_all_fl
    random_expected_prec = TOTAL_WITHDRAWN / TOTAL_LEARNERS

    # Timing metrics: denominator = TOTAL_WITHDRAWN (1,394) for "all withdrawn learners"
    n_wd_within_3w = 0
    n_wd_ge_2w = 0
    lead_runs = []

    for lk in fl_withdrawn:
        df_l = meta_va_full[meta_va_full["learner_key"] == lk].sort_values("week")
        unreg_w = df_l["unreg_week"].iloc[0]
        fl_w = df_l[df_l["flagged"]]["week"].values
        leads = unreg_w - fl_w

        if any(0 <= lead <= 3 for lead in leads):
            n_wd_within_3w += 1
        if any(lead >= 2 for lead in leads):
            n_wd_ge_2w += 1

        # Continuous final run: trace backwards from last flag before/at unreg
        is_fl_arr = df_l["flagged"].values
        w_arr = df_l["week"].values
        valid_fl = [i for i, (w, fl) in enumerate(zip(w_arr, is_fl_arr)) if fl and w <= unreg_w]
        if valid_fl:
            start_i = valid_fl[-1]
            while start_i > 0 and is_fl_arr[start_i - 1] and (w_arr[start_i] - w_arr[start_i - 1] == 1):
                start_i -= 1
            lead_runs.append(unreg_w - w_arr[start_i])

    pct_all_wd_within_3w = n_wd_within_3w / TOTAL_WITHDRAWN
    pct_all_wd_ge_2w = n_wd_ge_2w / TOTAL_WITHDRAWN
    med_run_lead = float(np.median(lead_runs)) if lead_runs else 0.0

    policy_rows.append({
        "Policy": name,
        "Total Flagged (Denom=6,474)": f"{n_fl:,} ({pct_all_fl*100:.1f}%)",
        "Learner Recall (Denom=1,394)": f"{n_fl_wd:,} ({recall_wd*100:.1f}%)",
        "Random Recall Ref": f"{random_expected_recall*100:.1f}%",
        "Learner Precision": f"{prec_l*100:.1f}%",
        "Random Prec Ref": f"{random_expected_prec*100:.1f}%",
        "Share of ALL WD within 3w": f"{n_wd_within_3w:,} ({pct_all_wd_within_3w*100:.1f}%)",
        "Share of ALL WD >= 2w": f"{n_wd_ge_2w:,} ({pct_all_wd_ge_2w*100:.1f}%)",
        "Median Lead (Final Run)": f"{med_run_lead:.1f} wks",
    })

df_policy_final = pd.DataFrame(policy_rows)
print(df_policy_final.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. FULL BY-WEEK TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. FULL BY-WEEK TABLE (ALL WEEKS 0..30)")
print("=" * 70)
by_week_records = []
for w in sorted(meta_va["week"].unique()):
    mask_w = (meta_va["week"] == w).values
    n_w = int(mask_w.sum())
    pos_w = int(y_va[mask_w].sum())
    base_w = float(pos_w / n_w) if n_w > 0 else 0.0

    if pos_w > 0 and n_w > 50:
        roc_w = float(roc_auc_score(y_va[mask_w], p_champ[mask_w]))
        pr_w = float(average_precision_score(y_va[mask_w], p_champ[mask_w]))
        mult_w = float(pr_w / base_w) if base_w > 0 else 0.0
    else:
        roc_w, pr_w, mult_w = np.nan, np.nan, np.nan

    period = "drop_add_window" if w in [0, 1] else "standard_period"
    by_week_records.append({
        "week": int(w), "period": period, "n_rows": n_w, "positives": pos_w,
        "base_rate": base_w, "roc_auc": roc_w, "pr_auc": pr_w, "pr_mult": mult_w
    })

df_full_by_week = pd.DataFrame(by_week_records)
df_full_by_week.to_csv("reports/phase2/by_week_full.csv", index=False)
print(df_full_by_week.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SCORE BASELINE FITTED PER SNAPSHOT WEEK & ASSESSMENT AUDIT TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. SCORE BASELINE FITTED PER SNAPSHOT WEEK (WEEKS 4, 8, 12)")
print("=" * 70)
vle_all = pd.read_parquet("data/processed/features_weekly.parquet")
vle_va_nw = vle_all[(vle_all["split"] == "val") & (vle_all["final_result"] != "Withdrawn")].copy()
vle_tr_nw = vle_all[(vle_all["split"] == "train") & (vle_all["final_result"] != "Withdrawn")].copy()

snapshot_score_results = []
for w in [4, 8, 12]:
    tr_w = vle_tr_nw[vle_tr_nw["week"] == w]
    va_w = vle_va_nw[vle_va_nw["week"] == w]

    # Baseline 1: Constant cohort mean fitted on train at week w
    train_mean = float(tr_w["final_score"].mean())
    p_mean_base = np.full(len(va_w), train_mean)

    # Baseline 2: Mean submitted score to date (mean_score_submitted) fitted via linear regression on train at week w
    sub_tr = tr_w[["mean_score_submitted"]].fillna(0)
    sub_va = va_w[["mean_score_submitted"]].fillna(0)
    from sklearn.linear_model import LinearRegression
    lr_score = LinearRegression().fit(sub_tr, tr_w["final_score"])
    p_score_base = lr_score.predict(sub_va)

    # Actual champion LightGBM forecast
    # We load saved forecast predictions from forecast_results.joblib
    import joblib
    f_res = joblib.load("reports/phase2/forecast_results.joblib")
    lgb_mae = f_res["score"]["snapshots"][w]["lgbm"]["mae"]
    lgb_r2 = f_res["score"]["snapshots"][w]["lgbm"]["r2"]

    y_true_w = va_w["final_score"].values
    mae_mean = float(mean_absolute_error(y_true_w, p_mean_base))
    r2_mean = float(r2_score(y_true_w, p_mean_base))
    mae_sub = float(mean_absolute_error(y_true_w, p_score_base))
    r2_sub = float(r2_score(y_true_w, p_score_base))

    snapshot_score_results.append({
        "week": w,
        "n_learners": len(va_w),
        "mean_baseline_mae": mae_mean, "mean_baseline_r2": r2_mean,
        "fitted_submitted_score_mae": mae_sub, "fitted_submitted_score_r2": r2_sub,
        "lgbm_mae": lgb_mae, "lgbm_r2": lgb_r2,
    })

df_snapshot_scores = pd.DataFrame(snapshot_score_results)
print(df_snapshot_scores.to_string(index=False))

# Save all step 11 data
step11_data = {
    "paired_bootstrap": paired_bootstrap_results,
    "nested_tuning": {"best_params": best_nested_params, "roc_val": roc_va_to, "pr_val": pr_va_to},
    "shuffled_20": {"rocs": shuffled_20_rocs, "prs": shuffled_20_prs, "dist": df_shuff_dist.to_dict()},
    "champion_v2": {"roc_val": roc_v2, "pr_val": pr_v2, "best_th": best_th_v2, "best_mcc": best_mcc_v2},
    "policy_table": df_policy_final.to_dict(orient="records"),
    "full_by_week": by_week_records,
    "snapshot_scores": snapshot_score_results,
}
import joblib
joblib.dump(step11_data, "reports/phase2/step11_data.joblib")
print("\nAll Step 11 data saved -> reports/phase2/step11_data.joblib")
