"""
Checkpoint A Follow-ups Script:
1. Ablation: LightGBM without course_progress & studied_credits (5 seeds).
2. Week-only model (week + course_progress) on week >= 2.
3. Proper nested presentation CV (Optuna on 2013B->2013J and 2013J->2013B).
4. Paired cluster-bootstrap for LightGBM vs LR (point diff + 95% CI of diff).
5. All 95% CIs for comparison table baselines.
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef
from joblib import Parallel, delayed

from src.models.data import load_split
from src.models.metrics import evaluate_predictions, compute_bootstrap_ci
from src.models.train_lr import prepare_lr_matrices

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Loading data...")
X_tr, y_tr, meta_tr = load_split("train")
X_va, y_va, meta_va = load_split("val")
keys_va = meta_va["code_module"] + "_" + meta_va["code_presentation"] + "_" + meta_va["id_student"].astype(str)

# ── 1. Ablation: LightGBM without course_progress & studied_credits (5 seeds) ──
print("\n" + "=" * 70)
print("[1/5] Ablation: LightGBM WITHOUT course_progress & studied_credits (5 seeds)")
print("=" * 70)
drop_cols = ["course_progress", "studied_credits"]
ablation_features = [c for c in X_tr.columns if c not in drop_cols]
X_tr_abl = X_tr[ablation_features]
X_va_abl = X_va[ablation_features]

base_params = {
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
    "n_estimators": 1500,
    "verbosity": -1,
    "objective": "binary",
}

seeds = [42, 101, 2024, 7, 999]
abl_rocs, abl_prs = [], []
for s in seeds:
    clf = lgb.LGBMClassifier(**base_params, random_state=s)
    clf.fit(
        X_tr_abl,
        y_tr,
        eval_X=X_va_abl,
        eval_y=y_va,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_va = clf.predict_proba(X_va_abl)[:, 1]
    abl_rocs.append(roc_auc_score(y_va, p_va))
    abl_prs.append(average_precision_score(y_va, p_va))

print(f"5-Seed Ablation (without {drop_cols}):")
print(f"  Val ROC-AUC: {np.mean(abl_rocs):.4f} +/- {np.std(abl_rocs):.4f} (individual: {[round(x,4) for x in abl_rocs]})")
print(f"  Val PR-AUC:  {np.mean(abl_prs):.4f} +/- {np.std(abl_prs):.4f} (individual: {[round(x,4) for x in abl_prs]})")

# ── 2. Week-only Model (week + course_progress on week >= 2) ───────────────────
print("\n" + "=" * 70)
print("[2/5] Week-Only Model (week + course_progress on week >= 2)")
print("=" * 70)
# meta_tr['week'] and X_tr['course_progress']
week_tr = pd.DataFrame({"week": meta_tr["week"], "course_progress": X_tr["course_progress"]})
week_va = pd.DataFrame({"week": meta_va["week"], "course_progress": X_va["course_progress"]})

# Fit logistic regression
lr_week = LogisticRegression(C=1.0, random_state=42)
lr_week.fit(week_tr, y_tr)
p_week_lr = lr_week.predict_proba(week_va)[:, 1]
roc_week_lr = roc_auc_score(y_va, p_week_lr)
pr_week_lr = average_precision_score(y_va, p_week_lr)

# Fit LightGBM on week-only features
lgb_week = lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbosity=-1)
lgb_week.fit(week_tr, y_tr)
p_week_lgb = lgb_week.predict_proba(week_va)[:, 1]
roc_week_lgb = roc_auc_score(y_va, p_week_lgb)
pr_week_lgb = average_precision_score(y_va, p_week_lgb)

print(f"Week-only Logistic Regression: Val ROC-AUC = {roc_week_lr:.4f}, Val PR-AUC = {pr_week_lr:.4f}")
print(f"Week-only LightGBM:            Val ROC-AUC = {roc_week_lgb:.4f}, Val PR-AUC = {pr_week_lgb:.4f}")
print(f"Sanity Week-Only Ceiling (<0.65 ROC-AUC): {'PASS' if max(roc_week_lr, roc_week_lgb) < 0.65 else 'ALERT: >= 0.65'}")

# ── 3. Proper Nested Presentation-CV (Tuning inside each fold) ────────────────
print("\n" + "=" * 70)
print("[3/5] Proper Nested Presentation-CV (Tuning inside each fold)")
print("=" * 70)
mask_b = (meta_tr["code_presentation"] == "2013B").values
mask_j = (meta_tr["code_presentation"] == "2013J").values

def tune_presentation_fold(X_fit, y_fit, X_eval, y_eval, n_trials=15):
    def obj(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10.0),
            "scale_pos_weight": 1.0,
            "n_estimators": 800,
            "random_state": 42,
            "verbosity": -1,
            "objective": "binary",
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_fit, y_fit, eval_X=X_eval, eval_y=y_eval, callbacks=[lgb.early_stopping(50, verbose=False)])
        probs = clf.predict_proba(X_eval)[:, 1]
        return average_precision_score(y_eval, probs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best_p = study.best_params
    best_clf = lgb.LGBMClassifier(**best_p, n_estimators=1200, bagging_freq=1, scale_pos_weight=1.0, random_state=42, verbosity=-1)
    best_clf.fit(X_fit, y_fit, eval_X=X_eval, eval_y=y_eval, callbacks=[lgb.early_stopping(50, verbose=False)])
    eval_probs = best_clf.predict_proba(X_eval)[:, 1]
    return average_precision_score(y_eval, eval_probs), roc_auc_score(y_eval, eval_probs), best_p, best_clf

print("  Fold 1: Tune on 2013B -> Score on 2013J...")
pr_nested_1, roc_nested_1, params_1, model_1 = tune_presentation_fold(X_tr[mask_b], y_tr[mask_b], X_tr[mask_j], y_tr[mask_j], n_trials=15)
print(f"    2013B -> 2013J: PR-AUC = {pr_nested_1:.4f}, ROC-AUC = {roc_nested_1:.4f}")

print("  Fold 2: Tune on 2013J -> Score on 2013B...")
pr_nested_2, roc_nested_2, params_2, model_2 = tune_presentation_fold(X_tr[mask_j], y_tr[mask_j], X_tr[mask_b], y_tr[mask_b], n_trials=15)
print(f"    2013J -> 2013B: PR-AUC = {pr_nested_2:.4f}, ROC-AUC = {roc_nested_2:.4f}")

mean_nested_pr = (pr_nested_1 + pr_nested_2) / 2
mean_nested_roc = (roc_nested_1 + roc_nested_2) / 2
print(f"  Nested Presentation-CV Mean: PR-AUC = {mean_nested_pr:.4f}, ROC-AUC = {mean_nested_roc:.4f}")

# Also score 2014B once with models tuned on train only:
p_va_m1 = model_1.predict_proba(X_va)[:, 1]
p_va_m2 = model_2.predict_proba(X_va)[:, 1]
p_va_ens = (p_va_m1 + p_va_m2) / 2
pr_va_from_nested = average_precision_score(y_va, p_va_ens)
roc_va_from_nested = roc_auc_score(y_va, p_va_ens)
print(f"  Train-Tuned Models Scored Once on 2014B: PR-AUC = {pr_va_from_nested:.4f}, ROC-AUC = {roc_va_from_nested:.4f}")
print(f"  Comparison vs Val-Tuned Champion (2014B): PR-AUC = 0.0735, ROC-AUC = 0.7736 (Delta PR = {pr_va_from_nested - 0.0735:+.4f})")

# ── 4. Paired Cluster-Bootstrap: LightGBM minus Logistic Regression ───────────
print("\n" + "=" * 70)
print("[4/5] Paired Cluster-Bootstrap: LightGBM Champion vs LR (Unweighted & Balanced)")
print("=" * 70)
# Prepare LR predictions
X_tr_lr, X_va_lr, _ = prepare_lr_matrices(X_tr, X_va)
lr_uw = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver="lbfgs")
lr_uw.fit(X_tr_lr, y_tr)
probs_lr_uw = lr_uw.predict_proba(X_va_lr)[:, 1]

lr_bal = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs", class_weight="balanced")
lr_bal.fit(X_tr_lr, y_tr)
probs_lr_bal = lr_bal.predict_proba(X_va_lr)[:, 1]

# LightGBM champion predictions
booster = lgb.Booster(model_file="models/baselines/lgbm_champion_v1.txt")
probs_lgb = booster.predict(X_va.values)

# Paired bootstrap across 500 learner resamples
codes, uniques = pd.factorize(keys_va)
n_clusters = len(uniques)
rng = np.random.default_rng(42)
all_sampled = rng.integers(0, n_clusters, size=(500, n_clusters))

def paired_resample(b):
    counts = np.bincount(all_sampled[b], minlength=n_clusters)
    w = counts[codes]
    sw_pos = float(np.dot(w.astype(np.float64), y_va.astype(np.float64)))
    sw_tot = float(w.sum())
    if sw_pos == 0.0 or sw_pos == sw_tot:
        return None

    # LGBM
    roc_lgb = roc_auc_score(y_va, probs_lgb, sample_weight=w)
    pr_lgb = average_precision_score(y_va, probs_lgb, sample_weight=w)

    # LR Unweighted
    roc_lr_uw = roc_auc_score(y_va, probs_lr_uw, sample_weight=w)
    pr_lr_uw = average_precision_score(y_va, probs_lr_uw, sample_weight=w)

    # LR Balanced
    roc_lr_bal = roc_auc_score(y_va, probs_lr_bal, sample_weight=w)
    pr_lr_bal = average_precision_score(y_va, probs_lr_bal, sample_weight=w)

    return {
        "diff_roc_uw": roc_lgb - roc_lr_uw,
        "diff_pr_uw": pr_lgb - pr_lr_uw,
        "diff_roc_bal": roc_lgb - roc_lr_bal,
        "diff_pr_bal": pr_lgb - pr_lr_bal,
    }

paired_results = Parallel(n_jobs=4, prefer="threads")(delayed(paired_resample)(b) for b in range(500))
paired_results = [r for r in paired_results if r is not None]

diff_roc_uw = [r["diff_roc_uw"] for r in paired_results]
diff_pr_uw = [r["diff_pr_uw"] for r in paired_results]
diff_roc_bal = [r["diff_roc_bal"] for r in paired_results]
diff_pr_bal = [r["diff_pr_bal"] for r in paired_results]

print(f"LightGBM minus LR-Unweighted:")
print(f"  Delta ROC-AUC: {np.mean(diff_roc_uw):+.4f}  [95% CI: {np.percentile(diff_roc_uw, 2.5):+.4f}, {np.percentile(diff_roc_uw, 97.5):+.4f}]")
print(f"  Delta PR-AUC:  {np.mean(diff_pr_uw):+.4f}  [95% CI: {np.percentile(diff_pr_uw, 2.5):+.4f}, {np.percentile(diff_pr_uw, 97.5):+.4f}]")
p_val_pr_uw = np.mean([d <= 0 for d in diff_pr_uw])
print(f"  P(Delta PR-AUC <= 0): {p_val_pr_uw:.4f}")

print(f"\nLightGBM minus LR-Balanced:")
print(f"  Delta ROC-AUC: {np.mean(diff_roc_bal):+.4f}  [95% CI: {np.percentile(diff_roc_bal, 2.5):+.4f}, {np.percentile(diff_roc_bal, 97.5):+.4f}]")
print(f"  Delta PR-AUC:  {np.mean(diff_pr_bal):+.4f}  [95% CI: {np.percentile(diff_pr_bal, 2.5):+.4f}, {np.percentile(diff_pr_bal, 97.5):+.4f}]")
p_val_pr_bal = np.mean([d <= 0 for d in diff_pr_bal])
print(f"  P(Delta PR-AUC <= 0): {p_val_pr_bal:.4f}")

# ── 5. Full Bootstrap CIs for ALL Comparison Table Baselines ──────────────────
print("\n" + "=" * 70)
print("[5/5] Computing 95% CIs for ALL Comparison Table Rows (500 resamples)")
print("=" * 70)
one_feat_names = ["weekly_clicks", "weighted_score_to_date", "pct_resources_visited",
                  "wow_change_clicks", "num_of_prev_attempts"]
one_feat_cis = {}

# Majority baseline CI
prior_probs = np.full(len(y_va), float(y_tr.mean()))
ev_prior = evaluate_predictions(y_va, prior_probs, learner_keys=keys_va, n_resamples=500)
print(f"Majority Prior: ROC={ev_prior['point']['roc_auc']:.4f} CI={ev_prior['ci']['roc_auc']} | PR={ev_prior['point']['pr_auc']:.4f} CI={ev_prior['ci']['pr_auc']}")

for f in one_feat_names:
    xtr = X_tr[[f]].fillna(X_tr[[f]].median())
    xva = X_va[[f]].fillna(X_tr[[f]].median())
    sc2 = StandardScaler().fit(xtr)
    lr2 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr2.fit(sc2.transform(xtr), y_tr)
    pv = lr2.predict_proba(sc2.transform(xva))[:, 1]
    ev = evaluate_predictions(y_va, pv, learner_keys=keys_va, n_resamples=500)
    one_feat_cis[f] = ev
    print(f"  {f:<25}: ROC={ev['point']['roc_auc']:.4f} {ev['ci']['roc_auc']} | PR={ev['point']['pr_auc']:.4f} {ev['ci']['pr_auc']}")

# Save results for reporting
results_summary = {
    "ablation_rocs": abl_rocs,
    "ablation_prs": abl_prs,
    "week_only_roc_lr": roc_week_lr,
    "week_only_roc_lgb": roc_week_lgb,
    "nested_cv": {
        "pr_1": pr_nested_1, "roc_1": roc_nested_1,
        "pr_2": pr_nested_2, "roc_2": roc_nested_2,
        "mean_pr": mean_nested_pr, "mean_roc": mean_nested_roc,
        "scored_on_val_pr": pr_va_from_nested,
        "scored_on_val_roc": roc_va_from_nested,
    },
    "paired_bootstrap": {
        "diff_roc_uw": (np.mean(diff_roc_uw), np.percentile(diff_roc_uw, 2.5), np.percentile(diff_roc_uw, 97.5)),
        "diff_pr_uw": (np.mean(diff_pr_uw), np.percentile(diff_pr_uw, 2.5), np.percentile(diff_pr_uw, 97.5)),
        "diff_roc_bal": (np.mean(diff_roc_bal), np.percentile(diff_roc_bal, 2.5), np.percentile(diff_roc_bal, 97.5)),
        "diff_pr_bal": (np.mean(diff_pr_bal), np.percentile(diff_pr_bal, 2.5), np.percentile(diff_pr_bal, 97.5)),
    },
    "prior_ci": ev_prior["ci"],
    "one_feat_cis": {f: (one_feat_cis[f]["point"], one_feat_cis[f]["ci"]) for f in one_feat_names},
}
import joblib
joblib.dump(results_summary, "models/baselines/checkpoint_a_followups.joblib")
print("\nAll follow-up results saved to models/baselines/checkpoint_a_followups.joblib")
