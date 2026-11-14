"""
Steps 8, 9, and 10 for Checkpoint C — fixed version.

Fixes from v1:
  - feature_groups now reads feat_yaml["features"]["<group>"] (not feat_yaml["<group>_features"])
  - champion_params uses exact values from lgbm_champion_v1.txt (lr=0.0305762, 285 trees)
  - n_estimators set to 1500 (capped by early stopping at ~285 in practice)
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
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.data import load_split

# ── Exact champion hyperparameters from lgbm_champion_v1.txt ────────────────
CHAMPION_PARAMS: Dict[str, Any] = {
    "num_leaves": 93,
    "max_depth": 4,
    "learning_rate": 0.0305762,
    "min_child_samples": 363,
    "feature_fraction": 0.932065,
    "bagging_fraction": 0.803165,
    "bagging_freq": 1,
    "lambda_l1": 8.06916,
    "lambda_l2": 2.8288,
    "scale_pos_weight": 1.0,
    "n_estimators": 1500,          # capped by early stopping
    "verbosity": -1,
    "objective": "binary",
    "random_state": 42,
}
SEEDS = [42, 101, 2024, 7, 999]

# ── Load train / val splits ──────────────────────────────────────────────────
print("Loading train/val splits …")
X_tr, y_tr, meta_tr = load_split("train", k=3, min_week=2)
X_va, y_va, meta_va = load_split("val",   k=3, min_week=2)
booster = lgb.Booster(model_file="models/baselines/lgbm_champion_v1.txt")
p_champ = booster.predict(X_va.values)

print(f"  Train: {len(X_tr):,} rows | Val: {len(X_va):,} rows")
print(f"  Champion on val => ROC={roc_auc_score(y_va, p_champ):.4f}  PR={average_precision_score(y_va, p_champ):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: ROBUSTNESS CHECKS")
print("=" * 70)

# ── 8.1 LOPO-CV ─────────────────────────────────────────────────────────────
print("\n[1/3] Leave-One-Presentation-Out CV (3 folds):")
X_all  = pd.concat([X_tr, X_va], ignore_index=True)
y_all  = np.concatenate([y_tr, y_va])
meta_all = pd.concat([meta_tr, meta_va], ignore_index=True)

presentations = ["2013B", "2013J", "2014B"]
lopo_results: List[Dict] = []

for pres in presentations:
    test_mask  = (meta_all["code_presentation"] == pres).values
    train_mask = ~test_mask
    X_tr_f, y_tr_f = X_all.iloc[train_mask], y_all[train_mask]
    X_va_f, y_va_f = X_all.iloc[test_mask],  y_all[test_mask]

    clf = lgb.LGBMClassifier(**CHAMPION_PARAMS)
    clf.fit(X_tr_f, y_tr_f,
            eval_X=X_va_f, eval_y=y_va_f,
            callbacks=[lgb.early_stopping(50, verbose=False)])
    p_f   = clf.predict_proba(X_va_f)[:, 1]
    roc_f = roc_auc_score(y_va_f, p_f)
    pr_f  = average_precision_score(y_va_f, p_f)
    base  = float(y_va_f.mean())
    lopo_results.append(dict(held_out=pres, n_rows=int(test_mask.sum()),
                             base_rate=base, roc_auc=roc_f, pr_auc=pr_f,
                             pr_mult=pr_f/base if base>0 else 0,
                             best_iter=clf.best_iteration_))
    print(f"  Held-out {pres}: ROC={roc_f:.4f}  PR={pr_f:.4f} "
          f"({pr_f/base:.2f}× base {base*100:.2f}%)  best_iter={clf.best_iteration_}")

df_lopo = pd.DataFrame(lopo_results)
mean_roc_lopo = df_lopo["roc_auc"].mean()
std_roc_lopo  = df_lopo["roc_auc"].std()
mean_pr_lopo  = df_lopo["pr_auc"].mean()
std_pr_lopo   = df_lopo["pr_auc"].std()
print(f"LOPO Mean: ROC={mean_roc_lopo:.4f}±{std_roc_lopo:.4f}  PR={mean_pr_lopo:.4f}±{std_pr_lopo:.4f}")

# ── 8.2 Seed Variance ───────────────────────────────────────────────────────
print("\n[2/3] Seed Variance (5 seeds, train→val):")
seed_rocs, seed_prs = [], []
for s in SEEDS:
    p_s = CHAMPION_PARAMS.copy(); p_s["random_state"] = s
    clf_s = lgb.LGBMClassifier(**p_s)
    clf_s.fit(X_tr, y_tr,
              eval_X=X_va, eval_y=y_va,
              callbacks=[lgb.early_stopping(50, verbose=False)])
    prob_s = clf_s.predict_proba(X_va)[:, 1]
    seed_rocs.append(roc_auc_score(y_va, prob_s))
    seed_prs.append(average_precision_score(y_va, prob_s))
    print(f"  seed={s}: ROC={seed_rocs[-1]:.4f}  PR={seed_prs[-1]:.4f}")

print(f"  Mean ROC={np.mean(seed_rocs):.4f}±{np.std(seed_rocs):.4f} "
      f"(range {np.min(seed_rocs):.4f}–{np.max(seed_rocs):.4f})")
print(f"  Mean PR ={np.mean(seed_prs):.4f}±{np.std(seed_prs):.4f} "
      f"(range {np.min(seed_prs):.4f}–{np.max(seed_prs):.4f})")

# ── 8.3 Sensitivity Slices ──────────────────────────────────────────────────
print("\n[3/3] Sensitivity Slices on Val 2014B (champion booster):")
roc_all = roc_auc_score(y_va, p_champ)
pr_all  = average_precision_score(y_va, p_champ)
print(f"  Baseline (all): ROC={roc_all:.4f}  PR={pr_all:.4f}  n={len(y_va):,}")

# Proxy slice
mask_no_proxy = ~meta_va["label_is_proxy"].astype(bool)
roc_np = roc_auc_score(y_va[mask_no_proxy], p_champ[mask_no_proxy])
pr_np  = average_precision_score(y_va[mask_no_proxy], p_champ[mask_no_proxy])
n_proxy = (~mask_no_proxy).sum()
print(f"  Remove proxy ({n_proxy} rows): ROC={roc_np:.4f}  PR={pr_np:.4f}")

# Zero-click withdrawal slice
mask_no_zc = ~meta_va["is_zero_click_withdrawal"].astype(bool)
roc_nz = roc_auc_score(y_va[mask_no_zc], p_champ[mask_no_zc])
pr_nz  = average_precision_score(y_va[mask_no_zc], p_champ[mask_no_zc])
n_zc = (~mask_no_zc).sum()
print(f"  Remove zero-click withdrawals ({n_zc} rows): ROC={roc_nz:.4f}  PR={pr_nz:.4f}")

# Repeat vs first-time
mask_repeat = (X_va["num_of_prev_attempts"] > 0).values
mask_first  = ~mask_repeat
roc_rep = roc_auc_score(y_va[mask_repeat], p_champ[mask_repeat])
pr_rep  = average_precision_score(y_va[mask_repeat], p_champ[mask_repeat])
roc_fst = roc_auc_score(y_va[mask_first],  p_champ[mask_first])
pr_fst  = average_precision_score(y_va[mask_first],  p_champ[mask_first])
print(f"  Repeat students ({mask_repeat.sum():,} rows): ROC={roc_rep:.4f}  PR={pr_rep:.4f}")
print(f"  First-time students ({mask_first.sum():,} rows): ROC={roc_fst:.4f}  PR={pr_fst:.4f}")

slices_dict = dict(
    all=dict(roc=roc_all, pr=pr_all, n=int(len(y_va))),
    no_proxy=dict(roc=roc_np, pr=pr_np, n=int(mask_no_proxy.sum())),
    no_zeroclick=dict(roc=roc_nz, pr=pr_nz, n=int(mask_no_zc.sum())),
    repeat=dict(roc=roc_rep, pr=pr_rep, n=int(mask_repeat.sum())),
    first_time=dict(roc=roc_fst, pr=pr_fst, n=int(mask_first.sum())),
)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 9: MODALITY ABLATIONS")
print("=" * 70)

with open("config/features.yaml") as f:
    feat_yaml = yaml.safe_load(f)

# Filter each group to only include the 39 continuous numeric features present in X_tr
feature_groups: Dict[str, List[str]] = {
    grp: [c for c in cols if c in X_tr.columns]
    for grp, cols in feat_yaml["features"].items()
}
print(f"Feature groups loaded (39 continuous features): { {k: len(v) for k, v in feature_groups.items()} }")

# Cumulative addition order: behavior → temporal → assessment → curriculum → static
CUM_ORDER = ["behavior", "temporal", "assessment", "curriculum", "static"]
all_feats = list(X_tr.columns)   # 39 sanctioned features (already in correct order)

# ── 9.1 Cumulative Addition ──────────────────────────────────────────────────
print("\n[1/2] Cumulative Feature Group Addition (5 seeds each):")
cum_features: List[str] = []
cum_results: List[Dict] = []

for grp in CUM_ORDER:
    cum_features.extend(feature_groups[grp])
    X_tr_sub = X_tr[cum_features]
    X_va_sub = X_va[cum_features]

    grp_rocs, grp_prs = [], []
    for s in SEEDS:
        p_s = CHAMPION_PARAMS.copy(); p_s["random_state"] = s
        clf_g = lgb.LGBMClassifier(**p_s)
        clf_g.fit(X_tr_sub, y_tr,
                  eval_X=X_va_sub, eval_y=y_va,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        prob_g = clf_g.predict_proba(X_va_sub)[:, 1]
        grp_rocs.append(roc_auc_score(y_va, prob_g))
        grp_prs.append(average_precision_score(y_va, prob_g))

    m_roc = float(np.mean(grp_rocs)); s_roc = float(np.std(grp_rocs))
    m_pr  = float(np.mean(grp_prs));  s_pr  = float(np.std(grp_prs))
    cum_results.append(dict(setting=f"+{grp} ({len(cum_features)} feats)",
                            group_added=grp, n_features=len(cum_features),
                            mean_roc=m_roc, std_roc=s_roc,
                            mean_pr=m_pr, std_pr=s_pr))
    print(f"  +{grp:<12} ({len(cum_features):>2} feats): "
          f"ROC={m_roc:.4f}±{s_roc:.4f}  PR={m_pr:.4f}±{s_pr:.4f}")

# ── 9.2 Leave-One-Group-Out ──────────────────────────────────────────────────
print("\n[2/2] Leave-One-Group-Out Ablation (5 seeds each):")
logo_results: List[Dict] = []
full_mean_roc = np.mean(seed_rocs)
full_mean_pr  = np.mean(seed_prs)

for grp in CUM_ORDER:
    sub_feats = [f for f in all_feats if f not in feature_groups[grp]]
    X_tr_sub = X_tr[sub_feats]
    X_va_sub = X_va[sub_feats]

    grp_rocs, grp_prs = [], []
    for s in SEEDS:
        p_s = CHAMPION_PARAMS.copy(); p_s["random_state"] = s
        clf_g = lgb.LGBMClassifier(**p_s)
        clf_g.fit(X_tr_sub, y_tr,
                  eval_X=X_va_sub, eval_y=y_va,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        prob_g = clf_g.predict_proba(X_va_sub)[:, 1]
        grp_rocs.append(roc_auc_score(y_va, prob_g))
        grp_prs.append(average_precision_score(y_va, prob_g))

    m_roc = float(np.mean(grp_rocs)); s_roc = float(np.std(grp_rocs))
    m_pr  = float(np.mean(grp_prs));  s_pr  = float(np.std(grp_prs))
    d_roc = m_roc - full_mean_roc
    d_pr  = m_pr  - full_mean_pr
    logo_results.append(dict(setting=f"Full minus {grp} ({len(sub_feats)} feats)",
                             group_removed=grp, n_features=len(sub_feats),
                             mean_roc=m_roc, std_roc=s_roc,
                             mean_pr=m_pr, std_pr=s_pr,
                             delta_roc=d_roc, delta_pr=d_pr))
    print(f"  Full−{grp:<12} ({len(sub_feats):>2} feats): "
          f"ROC={m_roc:.4f}±{s_roc:.4f} (Δ{d_roc:+.4f})  "
          f"PR={m_pr:.4f}±{s_pr:.4f} (Δ{d_pr:+.4f})")

df_ablation = pd.concat([pd.DataFrame(cum_results), pd.DataFrame(logo_results)], ignore_index=True)
df_ablation.to_csv("reports/phase2/ablation_table.csv", index=False)
print("  Saved → reports/phase2/ablation_table.csv")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 10: SANITY & LEAKAGE TESTS")
print("=" * 70)

# ── 10.1 Shuffled-label test (learner-level permutation) ────────────────────
print("\n[1/3] Shuffled-Label Test (5 permutations, learner-level):")
keys_tr = (meta_tr["code_module"] + "_"
           + meta_tr["code_presentation"] + "_"
           + meta_tr["id_student"].astype(str))
unique_learners = keys_tr.unique()
# Map learner key → row indices
learner_to_rows = {lk: np.where(keys_tr.values == lk)[0] for lk in unique_learners}
# Map learner key → majority label
learner_y = {lk: int(y_tr[idx].max()) for lk, idx in learner_to_rows.items()}

shuffled_rocs: List[float] = []
shuffled_prs:  List[float] = []

for rep in range(5):
    rng = np.random.default_rng(100 + rep)
    perm_learners = rng.permutation(unique_learners)
    learner_map = dict(zip(unique_learners, perm_learners))

    perm_labels = np.zeros(len(y_tr), dtype=int)
    for lk, perm_lk in learner_map.items():
        perm_labels[learner_to_rows[lk]] = learner_y[perm_lk]

    clf_sh = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, random_state=42, verbosity=-1)
    clf_sh.fit(X_tr, perm_labels)
    prob_sh = clf_sh.predict_proba(X_va)[:, 1]
    r_sh = roc_auc_score(y_va, prob_sh)
    pr_sh = average_precision_score(y_va, prob_sh)
    shuffled_rocs.append(r_sh)
    shuffled_prs.append(pr_sh)
    print(f"  Permutation {rep+1}: ROC={r_sh:.4f}  PR={pr_sh:.4f}")

mean_sh_roc = float(np.mean(shuffled_rocs))
pass_fail = "PASS (< 0.03 from 0.50)" if abs(mean_sh_roc - 0.50) < 0.03 else "FAIL"
print(f"  Shuffled mean ROC={mean_sh_roc:.4f} "
      f"(range {np.min(shuffled_rocs):.4f}–{np.max(shuffled_rocs):.4f}) → {pass_fail}")

# ── 10.2 Grouped Permutation Importance ─────────────────────────────────────
print("\n[2/3] Grouped Permutation Feature Importance on Val 2014B:")
base_roc = roc_auc_score(y_va, p_champ)
base_pr  = average_precision_score(y_va, p_champ)
rng = np.random.default_rng(42)

group_perm: List[Dict] = []
for grp, cols in feature_groups.items():
    X_perm = X_va.copy()
    perm_idx = rng.permutation(len(X_perm))
    for col in cols:
        if col in X_perm.columns:
            X_perm[col] = X_perm[col].iloc[perm_idx].values
    prob_perm = booster.predict(X_perm.values)
    roc_perm  = roc_auc_score(y_va, prob_perm)
    pr_perm   = average_precision_score(y_va, prob_perm)
    d_roc = base_roc - roc_perm
    d_pr  = base_pr  - pr_perm
    group_perm.append(dict(group=grp, n_features=len(cols),
                           permuted_roc=roc_perm, delta_roc=d_roc,
                           permuted_pr=pr_perm, delta_pr=d_pr,
                           pct_pr_drop=d_pr/base_pr*100))
    print(f"  Permute {grp:<12} ({len(cols):>2} feats): "
          f"PR drops {d_pr:.4f} ({d_pr/base_pr*100:.1f}%)  "
          f"ROC drops {d_roc:.4f}")

# Sort by PR drop descending
group_perm.sort(key=lambda x: x["delta_pr"], reverse=True)
print("  Ranked by PR drop:", [r["group"] for r in group_perm])

# ── 10.3 Per-Module AUC Sanity Check ────────────────────────────────────────
print("\n[3/3] Per-Module Sanity Check on Val 2014B:")
module_sanity: List[Dict] = []
all_pass = True
for m in sorted(meta_va["code_module"].unique()):
    m_mask = (meta_va["code_module"] == m).values
    y_m = y_va[m_mask]; p_m = p_champ[m_mask]
    roc_m = roc_auc_score(y_m, p_m)
    pr_m  = average_precision_score(y_m, p_m)
    base_m = float(y_m.mean())
    status = "PASS" if roc_m > 0.50 else "FAIL ← AUC < 0.50"
    if roc_m <= 0.50:
        all_pass = False
    module_sanity.append(dict(module=m, n_rows=int(m_mask.sum()),
                              base_rate=base_m, roc_auc=roc_m, pr_auc=pr_m))
    print(f"  {m}: n={m_mask.sum():>6,}  base={base_m*100:.2f}%  "
          f"ROC={roc_m:.4f}  PR={pr_m:.4f}  → {status}")
print(f"  All modules pass AUC > 0.50: {'YES' if all_pass else 'NO — see FAIL rows above'}")

# ── Save checkpoint C data ───────────────────────────────────────────────────
checkpoint_c_data = {
    "lopo": df_lopo.to_dict(orient="records"),
    "seed_variance": dict(rocs=seed_rocs, prs=seed_prs,
                          mean_roc=float(np.mean(seed_rocs)), std_roc=float(np.std(seed_rocs)),
                          mean_pr=float(np.mean(seed_prs)),  std_pr=float(np.std(seed_prs))),
    "slices": slices_dict,
    "cumulative_ablation": cum_results,
    "logo_ablation": logo_results,
    "shuffled_test": dict(rocs=shuffled_rocs, prs=shuffled_prs,
                          mean_roc=mean_sh_roc, pass_fail=pass_fail),
    "group_permutation_importance": group_perm,
    "module_sanity": module_sanity,
}
joblib.dump(checkpoint_c_data, "reports/phase2/checkpoint_c_data.joblib")
print("\nAll Steps 8–10 results saved → reports/phase2/checkpoint_c_data.joblib")
