"""
Step 6: Horizon, Week, and Module Analysis.
- Multi-horizon evaluation (k=2, 3, 4) reusing champion hyperparameters.
- By-week metrics (reporting weeks 0-1 separately from weeks >= 2).
- By-module metrics (noting CCC unseen in train, AAA absent from val).
- Secondary target evaluation: label_inactive_k3 (disengagement onset).
- Saves CSV tables to reports/phase2/.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.models.data import load_split
from src.models.metrics import evaluate_predictions


def run_horizon_analysis(
    horizons: List[int] = [2, 3, 4],
    n_resamples: int = 500,
) -> pd.DataFrame:
    """Evaluate champion architecture across horizons k=2, 3, 4."""
    print("=" * 60)
    print("RUNNING STEP 6.1: HORIZON ANALYSIS (k=2, 3, 4)")
    print("=" * 60)

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
        "n_estimators": 1500,
        "random_state": 42,
        "verbosity": -1,
        "objective": "binary",
    }

    records = []
    for k in horizons:
        X_tr, y_tr, meta_tr = load_split("train", k=k, min_week=2)
        X_va, y_va, meta_va = load_split("val", k=k, min_week=2)
        keys_va = meta_va["code_module"] + "_" + meta_va["code_presentation"] + "_" + meta_va["id_student"].astype(str)

        clf = lgb.LGBMClassifier(**champion_params)
        clf.fit(
            X_tr,
            y_tr,
            eval_X=X_va,
            eval_y=y_va,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        probs = clf.predict_proba(X_va)[:, 1]
        ev = evaluate_predictions(y_va, probs, learner_keys=keys_va, n_resamples=n_resamples)
        pt = ev["point"]
        ci = ev["ci"]

        records.append({
            "horizon_k": k,
            "n_val_rows": len(y_va),
            "base_rate": pt["base_rate"],
            "roc_auc": pt["roc_auc"],
            "roc_auc_ci_lower": ci["roc_auc"][0],
            "roc_auc_ci_upper": ci["roc_auc"][1],
            "pr_auc": pt["pr_auc"],
            "pr_auc_mult": pt["pr_auc_mult"],
            "pr_auc_ci_lower": ci["pr_auc"][0],
            "pr_auc_ci_upper": ci["pr_auc"][1],
            "brier": pt["brier"],
            "best_iteration": clf.best_iteration_,
        })
        print(f"k={k}: ROC-AUC={pt['roc_auc']:.4f} [{ci['roc_auc'][0]:.4f},{ci['roc_auc'][1]:.4f}] | "
              f"PR-AUC={pt['pr_auc']:.4f} ({pt['pr_auc_mult']:.2f}x) [{ci['pr_auc'][0]:.4f},{ci['pr_auc'][1]:.4f}] | "
              f"Base={pt['base_rate']*100:.2f}%")

    df_horizons = pd.DataFrame(records)
    out_dir = Path("reports/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_horizons.to_csv(out_dir / "by_horizon.csv", index=False)
    return df_horizons


def run_by_week_and_module_analysis(k: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute by-week and by-module metrics using champion predictions."""
    print("\n" + "=" * 60)
    print("RUNNING STEP 6.2 & 6.3: BY-WEEK AND BY-MODULE BREAKDOWNS")
    print("=" * 60)

    # Load all weeks (including weeks 0 and 1) for the full breakdown
    X_tr, y_tr, meta_tr = load_split("train", k=k, min_week=0)
    X_va, y_va, meta_va = load_split("val", k=k, min_week=0)

    # Fit champion on train (all weeks)
    booster = lgb.Booster(model_file="models/baselines/lgbm_champion_v1.txt")
    probs = booster.predict(X_va.values)

    df_eval = meta_va.copy()
    df_eval["y_true"] = y_val = y_va
    df_eval["y_prob"] = probs

    # ── By Week Breakdown ──────────────────────────────────────────────────
    week_records = []
    weeks = sorted(df_eval["week"].unique())

    for w in weeks:
        w_sub = df_eval[df_eval["week"] == w]
        n_rows = len(w_sub)
        n_pos = int(w_sub["y_true"].sum())
        base = n_pos / n_rows if n_rows > 0 else 0.0

        if n_pos > 0 and n_pos < n_rows:
            try:
                roc = float(roc_auc_score(w_sub["y_true"], w_sub["y_prob"]))
            except Exception:
                roc = 0.5
            try:
                pr = float(average_precision_score(w_sub["y_true"], w_sub["y_prob"]))
            except Exception:
                pr = base
        else:
            roc = 0.5
            pr = base

        pr_mult = float(pr / base) if base > 0 else 0.0
        period = "drop_add_window (weeks 0-1)" if w < 2 else "standard_period (weeks >= 2)"

        week_records.append({
            "week": int(w),
            "period": period,
            "n_rows": n_rows,
            "n_positives": n_pos,
            "base_rate": base,
            "roc_auc": roc,
            "pr_auc": pr,
            "pr_auc_mult": pr_mult,
        })

    df_week = pd.DataFrame(week_records)
    out_dir = Path("reports/phase2")
    df_week.to_csv(out_dir / "by_week.csv", index=False)

    print("\nBy-Week Summary (Weeks 0-1 vs Select Weeks >= 2):")
    sample_weeks = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    print(df_week[df_week["week"].isin(sample_weeks)][["week", "period", "n_rows", "base_rate", "roc_auc", "pr_auc", "pr_auc_mult"]].to_string(index=False))

    # ── By Module Breakdown (Validation Split 2014B) ──────────────────────
    module_records = []
    modules = sorted(df_eval["code_module"].unique())

    for m in modules:
        m_sub = df_eval[(df_eval["code_module"] == m) & (df_eval["week"] >= 2)]
        n_rows = len(m_sub)
        n_pos = int(m_sub["y_true"].sum())
        base = n_pos / n_rows if n_rows > 0 else 0.0

        if n_pos > 0 and n_pos < n_rows:
            try:
                roc = float(roc_auc_score(m_sub["y_true"], m_sub["y_prob"]))
            except Exception:
                roc = 0.5
            try:
                pr = float(average_precision_score(m_sub["y_true"], m_sub["y_prob"]))
            except Exception:
                pr = base
        else:
            roc = 0.5
            pr = base

        pr_mult = float(pr / base) if base > 0 else 0.0
        note = "UNSEEN IN TRAIN" if m == "CCC" else "Present in Train"

        module_records.append({
            "code_module": m,
            "train_status": note,
            "n_rows": n_rows,
            "n_positives": n_pos,
            "base_rate": base,
            "roc_auc": roc,
            "pr_auc": pr,
            "pr_auc_mult": pr_mult,
        })

    df_module = pd.DataFrame(module_records)
    df_module.to_csv(out_dir / "by_module.csv", index=False)

    print("\nBy-Module Summary (Validation Split 2014B, week >= 2):")
    print(df_module.to_string(index=False))

    return df_week, df_module


def run_secondary_target_analysis(k: int = 3, n_resamples: int = 500) -> Dict[str, Any]:
    """Train and evaluate champion architecture on secondary target: label_inactive_k3."""
    print("\n" + "=" * 60)
    print("RUNNING STEP 6.4: SECONDARY TARGET (label_inactive_k3: Disengagement Onset)")
    print("=" * 60)

    # Eligible rows for onset: clicks_in_week_t > 0, non-censored
    X_tr, y_tr, meta_tr = load_split("train", k=k, min_week=2, target_col="label_inactive_k3")
    X_va, y_va, meta_va = load_split("val", k=k, min_week=2, target_col="label_inactive_k3")
    keys_va = meta_va["code_module"] + "_" + meta_va["code_presentation"] + "_" + meta_va["id_student"].astype(str)

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
        "n_estimators": 1500,
        "random_state": 42,
        "verbosity": -1,
        "objective": "binary",
    }

    clf = lgb.LGBMClassifier(**champion_params)
    clf.fit(
        X_tr,
        y_tr,
        eval_X=X_va,
        eval_y=y_va,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    probs = clf.predict_proba(X_va)[:, 1]
    ev = evaluate_predictions(y_va, probs, learner_keys=keys_va, n_resamples=n_resamples)
    pt = ev["point"]
    ci = ev["ci"]

    print(f"Secondary Target (label_inactive_k3) Results:")
    print(f"  Val rows: {len(y_va):,} | Base Rate: {pt['base_rate']*100:.2f}%")
    print(f"  ROC-AUC:  {pt['roc_auc']:.4f}  [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][1]:.4f}]")
    print(f"  PR-AUC:   {pt['pr_auc']:.4f}  [{ci['pr_auc'][0]:.4f}, {ci['pr_auc'][1]:.4f}] ({pt['pr_auc_mult']:.2f}x)")
    print(f"  Brier:    {pt['brier']:.4f}")

    return {
        "point": pt,
        "ci": ci,
        "best_iteration": clf.best_iteration_,
    }


def run_all_step6():
    df_hor = run_horizon_analysis(horizons=[2, 3, 4], n_resamples=500)
    df_wk, df_mod = run_by_week_and_module_analysis(k=3)
    res_inact = run_secondary_target_analysis(k=3, n_resamples=500)
    return df_hor, df_wk, df_mod, res_inact


if __name__ == "__main__":
    run_all_step6()
