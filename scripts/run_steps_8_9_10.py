"""
End-to-End Execution of Steps 8, 9, and 10 for Checkpoint C.

Step 8: Robustness Checks
- Leave-one-presentation-out CV (LOPO-CV: 2013B, 2013J, 2014B)
- Seed variance across 5 seeds
- Sensitive slices: with/without proxy, with/without zero-click withdrawals, with/without repeat students

Step 9: Modality Ablations
- Cumulative addition (Behavior -> +Temporal -> +Assessment -> +Curriculum -> +Static)
- Leave-one-group-out (Full minus each group)
- 5 seeds per setting

Step 10: Sanity & Leakage Tests
- Shuffled-label test at learner level (5 permutations)
- Grouped permutation feature importance
- Per-module sanity verification (all AUC > 0.50)
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from pathlib import Path
from typing import Dict, Any, List
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.models.data import load_split, load_feature_manifest
from src.models.metrics import evaluate_predictions
from src.models.tracking import log_model_experiment

champion_params = {
    "num_leaves": 93,
    "max_depth": 4,
    "learning_rate": 0.0306,
    "min_child_samples": 363,
    "feature_fraction": 0.932,
    "bagging_fraction": 0.803,
    "bagging_freq": 1,
    "lambda_l1": 8.07,
    "lambda_l2": 2.83,
    "scale_pos_weight": 1.0,
    "n_estimators": 1000,
    "verbosity": -1,
    "objective": "binary",
    "random_state": 42,
}

print("=" * 70)
print("STEP 8: ROBUSTNESS CHECKS")
print("=" * 70)

# 8.1 Leave-One-Presentation-Out CV (LOPO-CV: 2013B, 2013J, 2014B)
# Need combined dataset of 2013B, 2013J, and 2014B
X_tr, y_tr, meta_tr = load_split("train", k=3, min_week=2)
X_va, y_va, meta_va = load_split("val", k=3, min_week=2)

X_all = pd.concat([X_tr, X_va], ignore_index=True)
y_all = np.concatenate([y_tr, y_va])
meta_all = pd.concat([meta_tr, meta_va], ignore_index=True)

presentations = ["2013B", "2013J", "2014B"]
lopo_results = []

for pres in presentations:
    test_mask = (meta_all["code_presentation"] == pres).values
    train_mask = ~test_mask

    X_train_fold, y_train_fold = X_all.iloc[train_mask], y_all[train_mask]
    X_val_fold, y_val_fold = X_all.iloc[test_mask], y_all[test_mask]

    clf = lgb.LGBMClassifier(**champion_params)
    clf.fit(
        X_train_fold,
        y_train_fold,
        eval_X=X_val_fold,
        eval_y=y_val_fold,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_val_fold = clf.predict_proba(X_val_fold)[:, 1]
    roc_f = roc_auc_score(y_val_fold, p_val_fold)
    pr_f = average_precision_score(y_val_fold, p_val_fold)
    base_f = float(y_val_fold.mean())
    mult_f = pr_f / base_f if base_f > 0 else 0.0

    lopo_results.append({
        "held_out_presentation": pres,
        "n_rows": int(test_mask.sum()),
        "base_rate": base_f,
        "roc_auc": roc_f,
        "pr_auc": pr_f,
        "pr_auc_mult": mult_f,
        "best_iteration": clf.best_iteration_,
    })
    print(f"  Held-out {pres}: ROC-AUC = {roc_f:.4f}, PR-AUC = {pr_f:.4f} ({mult_f:.2f}x base {base_f*100:.2f}%), best_iter={clf.best_iteration_}")

df_lopo = pd.DataFrame(lopo_results)
mean_lopo_roc = df_lopo["roc_auc"].mean()
std_lopo_roc = df_lopo["roc_auc"].std()
mean_lopo_pr = df_lopo["pr_auc"].mean()
std_lopo_pr = df_lopo["pr_auc"].std()
print(f"LOPO-CV Summary (3 folds): ROC-AUC = {mean_lopo_roc:.4f} +/- {std_lopo_roc:.4f}, PR-AUC = {mean_lopo_pr:.4f} +/- {std_lopo_pr:.4f}")

# 8.2 Seed Variance across 5 seeds on Val (2014B)
print("\nSeed Variance across 5 seeds on Val 2014B:")
seeds = [42, 101, 2024, 7, 999]
seed_rocs, seed_prs = [], []
for s in seeds:
    params_s = champion_params.copy()
    params_s["random_state"] = s
    clf_s = lgb.LGBMClassifier(**params_s)
    clf_s.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, callbacks=[lgb.early_stopping(50, verbose=False)])
    p_s = clf_s.predict_proba(X_va)[:, 1]
    seed_rocs.append(roc_auc_score(y_va, p_s))
    seed_prs.append(average_precision_score(y_va, p_s))

print(f"  ROC-AUC: {np.mean(seed_rocs):.4f} +/- {np.std(seed_rocs):.4f} (range: {np.min(seed_rocs):.4f} - {np.max(seed_rocs):.4f})")
print(f"  PR-AUC:  {np.mean(seed_prs):.4f} +/- {np.std(seed_prs):.4f} (range: {np.min(seed_prs):.4f} - {np.max(seed_prs):.4f})")

# 8.3 Sensitivity Slices on Val (2014B)
print("\nSensitivity Slices on Val 2014B:")
booster = lgb.Booster(model_file="models/baselines/lgbm_champion_v1.txt")
p_champ = booster.predict(X_va.values)

# Slice 1: Proxy learners (74 in val)
mask_no_proxy = ~meta_va["label_is_proxy"].astype(bool)
roc_all = roc_auc_score(y_va, p_champ)
pr_all = average_precision_score(y_va, p_champ)

roc_no_proxy = roc_auc_score(y_va[mask_no_proxy], p_champ[mask_no_proxy])
pr_no_proxy = average_precision_score(y_va[mask_no_proxy], p_champ[mask_no_proxy])
print(f"  Proxy Slicing: All Val ({len(y_va):,} rows) ROC={roc_all:.4f}, PR={pr_all:.4f} | "
      f"Without Proxy ({mask_no_proxy.sum():,} rows) ROC={roc_no_proxy:.4f}, PR={pr_no_proxy:.4f}")

# Slice 2: Zero-click withdrawals
mask_no_zeroclick = ~meta_va["is_zero_click_withdrawal"].astype(bool)
roc_no_zc = roc_auc_score(y_va[mask_no_zeroclick], p_champ[mask_no_zeroclick])
pr_no_zc = average_precision_score(y_va[mask_no_zeroclick], p_champ[mask_no_zeroclick])
print(f"  Zero-Click Withdrawal Slicing: Without Zero-Click ({mask_no_zeroclick.sum():,} rows) ROC={roc_no_zc:.4f}, PR={pr_no_zc:.4f}")

# Slice 3: Repeat students (num_of_prev_attempts > 0 vs == 0)
mask_repeat = X_va["num_of_prev_attempts"] > 0
mask_first = X_va["num_of_prev_attempts"] == 0
roc_repeat = roc_auc_score(y_va[mask_repeat], p_champ[mask_repeat])
pr_repeat = average_precision_score(y_va[mask_repeat], p_champ[mask_repeat])
roc_first = roc_auc_score(y_va[mask_first], p_champ[mask_first])
pr_first = average_precision_score(y_va[mask_first], p_champ[mask_first])
print(f"  Repeat Students ({mask_repeat.sum():,} rows): ROC={roc_repeat:.4f}, PR={pr_repeat:.4f} | "
      f"First-Time Students ({mask_first.sum():,} rows): ROC={roc_first:.4f}, PR={pr_first:.4f}")


print("\n" + "=" * 70)
print("STEP 9: MODALITY ABLATIONS")
print("=" * 70)

with open("config/features.yaml", "r") as f:
    feat_yaml = yaml.safe_load(f)

# Group definition (the 39 continuous numeric features)
feature_groups = {
    "behavior": feat_yaml["behavior_features"],
    "temporal": feat_yaml["temporal_features"],
    "assessment": feat_yaml["assessment_features"],
    "curriculum": feat_yaml["curriculum_features"],
    "static": feat_yaml["static_demographics_features"],
}

# 9.1 Cumulative Addition
print("\n[1/2] Cumulative Feature Group Addition (5 seeds each):")
order = ["behavior", "temporal", "assessment", "curriculum", "static"]
cum_features = []
cum_results = []

for grp in order:
    cum_features.extend(feature_groups[grp])
    X_tr_sub = X_tr[cum_features]
    X_va_sub = X_va[cum_features]

    grp_rocs, grp_prs = [], []
    for s in seeds:
        params = champion_params.copy()
        params["random_state"] = s
        clf_g = lgb.LGBMClassifier(**params)
        clf_g.fit(X_tr_sub, y_tr, eval_X=X_va_sub, eval_y=y_va, callbacks=[lgb.early_stopping(50, verbose=False)])
        p_g = clf_g.predict_proba(X_va_sub)[:, 1]
        grp_rocs.append(roc_auc_score(y_va, p_g))
        grp_prs.append(average_precision_score(y_va, p_g))

    m_roc, s_roc = float(np.mean(grp_rocs)), float(np.std(grp_rocs))
    m_pr, s_pr = float(np.mean(grp_prs)), float(np.std(grp_prs))
    cum_results.append({
        "setting": f"+ {grp} ({len(cum_features)} feats)",
        "group_added": grp,
        "n_features": len(cum_features),
        "mean_roc": m_roc,
        "std_roc": s_roc,
        "mean_pr": m_pr,
        "std_pr": s_pr,
    })
    print(f"  + {grp:<12} ({len(cum_features):>2} feats): ROC-AUC = {m_roc:.4f} +/- {s_roc:.4f} | PR-AUC = {m_pr:.4f} +/- {s_pr:.4f}")

# 9.2 Leave-One-Group-Out
print("\n[2/2] Leave-One-Group-Out (LOGO Ablation, 5 seeds each):")
logo_results = []
all_feats = list(X_tr.columns)

for grp in order:
    sub_feats = [f for f in all_feats if f not in feature_groups[grp]]
    X_tr_sub = X_tr[sub_feats]
    X_va_sub = X_va[sub_feats]

    grp_rocs, grp_prs = [], []
    for s in seeds:
        params = champion_params.copy()
        params["random_state"] = s
        clf_g = lgb.LGBMClassifier(**params)
        clf_g.fit(X_tr_sub, y_tr, eval_X=X_va_sub, eval_y=y_va, callbacks=[lgb.early_stopping(50, verbose=False)])
        p_g = clf_g.predict_proba(X_va_sub)[:, 1]
        grp_rocs.append(roc_auc_score(y_va, p_g))
        grp_prs.append(average_precision_score(y_va, p_g))

    m_roc, s_roc = float(np.mean(grp_rocs)), float(np.std(grp_rocs))
    m_pr, s_pr = float(np.mean(grp_prs)), float(np.std(grp_prs))
    logo_results.append({
        "setting": f"Full minus {grp} ({len(sub_feats)} feats)",
        "group_removed": grp,
        "n_features": len(sub_feats),
        "mean_roc": m_roc,
        "std_roc": s_roc,
        "mean_pr": m_pr,
        "std_pr": s_pr,
        "delta_roc": m_roc - np.mean(seed_rocs),
        "delta_pr": m_pr - np.mean(seed_prs),
    })
    print(f"  Full minus {grp:<12} ({len(sub_feats):>2} feats): ROC-AUC = {m_roc:.4f} +/- {s_roc:.4f} (Delta: {m_roc - np.mean(seed_rocs):+.4f}) | "
          f"PR-AUC = {m_pr:.4f} +/- {s_pr:.4f} (Delta: {m_pr - np.mean(seed_prs):+.4f})")

# Save ablation table
df_ablation = pd.concat([pd.DataFrame(cum_results), pd.DataFrame(logo_results)], ignore_index=True)
df_ablation.to_csv("reports/phase2/ablation_table.csv", index=False)


print("\n" + "=" * 70)
print("STEP 10: SANITY & LEAKAGE TESTS")
print("=" * 70)

# 10.1 Shuffled-label test at learner level (5 permutations)
print("\n[1/3] Shuffled-Label Test at Learner Level (5 permutations):")
# Permute labels by learner key so all rows of a learner receive the target of another learner
keys_tr = meta_tr["code_module"] + "_" + meta_tr["code_presentation"] + "_" + meta_tr["id_student"].astype(str)
unique_learners = keys_tr.unique()
learner_to_idx = meta_tr.groupby(keys_tr).indices

shuffled_rocs, shuffled_prs = [], []
for rep in range(5):
    rng = np.random.default_rng(100 + rep)
    permuted_learners = rng.permutation(unique_learners)
    learner_map = dict(zip(unique_learners, permuted_learners))

    # Build permuted label vector
    perm_labels = np.zeros(len(y_tr), dtype=int)
    # Target lookup for each original learner
    learner_y = meta_tr.assign(y=y_tr).groupby(keys_tr)["y"].max().to_dict()
    for lk, perm_lk in learner_map.items():
        rows = learner_to_idx[lk]
        perm_labels[rows] = learner_y[perm_lk]

    clf_shuff = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    clf_shuff.fit(X_tr, perm_labels)
    p_shuff = clf_shuff.predict_proba(X_va)[:, 1]
    r_shuff = roc_auc_score(y_va, p_shuff)
    pr_shuff = average_precision_score(y_va, p_shuff)
    shuffled_rocs.append(r_shuff)
    shuffled_prs.append(pr_shuff)
    print(f"  Permutation {rep+1}: ROC-AUC = {r_shuff:.4f}, PR-AUC = {pr_shuff:.4f} (Base = {float(y_va.mean())*100:.2f}%)")

print(f"  Shuffled Mean ROC-AUC: {np.mean(shuffled_rocs):.4f} (range: {np.min(shuffled_rocs):.4f} - {np.max(shuffled_rocs):.4f}) -> "
      f"{'PASS (near 0.50)' if abs(np.mean(shuffled_rocs) - 0.50) < 0.03 else 'FAIL'}")

# 10.2 Grouped Permutation Importance on Val (2014B)
print("\n[2/3] Grouped Permutation Feature Importance on Val (2014B):")
base_pr = average_precision_score(y_va, p_champ)
base_roc = roc_auc_score(y_va, p_champ)
rng = np.random.default_rng(42)

group_perm_results = []
for grp, cols in feature_groups.items():
    X_perm = X_va.copy()
    # Permute all features in group simultaneously
    perm_idx = rng.permutation(len(X_perm))
    for col in cols:
        X_perm[col] = X_perm[col].iloc[perm_idx].values

    p_perm = booster.predict(X_perm.values)
    roc_perm = roc_auc_score(y_va, p_perm)
    pr_perm = average_precision_score(y_va, p_perm)
    drop_pr = base_pr - pr_perm
    drop_roc = base_roc - roc_perm

    group_perm_results.append({
        "group": grp,
        "n_features": len(cols),
        "permuted_roc": roc_perm,
        "delta_roc": -drop_roc,
        "permuted_pr": pr_perm,
        "delta_pr": -drop_pr,
        "pct_pr_drop": (drop_pr / base_pr) * 100,
    })
    print(f"  Permute {grp:<12} ({len(cols):>2} feats): PR-AUC drops by {drop_pr:.4f} ({drop_pr/base_pr*100:>5.1f}%) | ROC drops by {drop_roc:.4f}")

# 10.3 Per-Module Sanity Check
print("\n[3/3] Per-Module Sanity Check on Val 2014B:")
for m in sorted(meta_va["code_module"].unique()):
    m_mask = meta_va["code_module"] == m
    y_m = y_va[m_mask]
    p_m = p_champ[m_mask]
    roc_m = roc_auc_score(y_m, p_m)
    pr_m = average_precision_score(y_m, p_m)
    status = "PASS (> 0.50)" if roc_m > 0.50 else "FAIL (< 0.50)"
    print(f"  Module {m}: ROC-AUC = {roc_m:.4f}, PR-AUC = {pr_m:.4f} -> {status}")

# Save full results dictionary
checkpoint_c_data = {
    "lopo": df_lopo.to_dict(orient="records"),
    "seed_variance": {"rocs": seed_rocs, "prs": seed_prs, "mean_roc": np.mean(seed_rocs), "std_roc": np.std(seed_rocs), "mean_pr": np.mean(seed_prs), "std_pr": np.std(seed_prs)},
    "slices": {
        "all": {"roc": roc_all, "pr": pr_all},
        "no_proxy": {"roc": roc_no_proxy, "pr": pr_no_proxy},
        "no_zeroclick": {"roc": roc_no_zc, "pr": pr_no_zc},
        "repeat": {"roc": roc_repeat, "pr": pr_repeat},
        "first_time": {"roc": roc_first, "pr": pr_first},
    },
    "cumulative_ablation": cum_results,
    "logo_ablation": logo_results,
    "shuffled_test": {"rocs": shuffled_rocs, "prs": shuffled_prs},
    "group_permutation_importance": group_perm_results,
}
joblib.dump(checkpoint_c_data, "reports/phase2/checkpoint_c_data.joblib")
print("\nAll Steps 8-10 results serialized to reports/phase2/checkpoint_c_data.joblib")
