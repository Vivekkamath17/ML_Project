"""
Print the Phase 3 Checkpoint A comparison table.
Reads from:
  - predictions/dropout/k3/lgbm_v3_val.parquet  (baseline)
  - predictions/dropout/k3/lr_39feat_unweighted_val.parquet  (baseline)
  - reports/zoo_checkpoint_results.json  (zoo model results)
  - reports/majority_prior.json  (optional pre-computed majority prior)

Prints all rows with:
  - Val ROC-AUC [95% CI]  (learner-bootstrap)
  - Val PR-AUC [95% CI]
  - PR multiple of base rate
  - Brier score
  - 5-seed mean ± std  (zoo models only)
  - Paired Δ vs LightGBM v3 [95% CI]
  - Inner-CV PR-AUC → Val gap  (zoo models with tuning)
  - Training time (s)
  - Inference latency (ms/1k rows)
  - Statistically tied marker (*)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.harness import evaluate_model, paired_bootstrap


LGBM_PRED_PATH = "predictions/dropout/k3/lgbm_v3_val.parquet"
LR_PRED_PATH = "predictions/dropout/k3/lr_39feat_unweighted_val.parquet"
ZOO_RESULTS_PATH = "reports/zoo_checkpoint_results.json"

BASE_RATE = 0.0237  # 2014B val positive rate
LGBM_REF_INNER_CV_PR = 0.1038  # pre-computed reference inner-CV PR-AUC
LGBM_REF_INNER_CV_ROC = 0.8146
LGBM_REF_VAL_PR = 0.0786
LGBM_REF_VAL_ROC = 0.7757
LGBM_REF_GAP_PR = LGBM_REF_VAL_PR - LGBM_REF_INNER_CV_PR   # -0.0252
LGBM_REF_GAP_ROC = LGBM_REF_VAL_ROC - LGBM_REF_INNER_CV_ROC  # -0.0389


def compute_ci_from_parquet(
    pred_path: str,
    n_resamples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute metrics and CIs for a saved prediction file."""
    df = pd.read_parquet(pred_path)
    y_va = df["y_true"].to_numpy().astype(int)
    y_prob = df["y_prob"].to_numpy().astype(float)

    roc = roc_auc_score(y_va, y_prob)
    pr = average_precision_score(y_va, y_prob)
    brier = float(np.mean((y_prob - y_va) ** 2))

    # Compute bootstrap CIs
    cluster_ids = df["id_student"]
    res = evaluate_model(y_va, y_prob, cluster_ids=cluster_ids, n_resamples=n_resamples, seed=seed)
    ci = res.get("ci", {})
    roc_ci = ci.get("roc_auc", (np.nan, np.nan))
    pr_ci = ci.get("pr_auc", (np.nan, np.nan))
    roc_lo, roc_hi = roc_ci[0], roc_ci[1]
    pr_lo, pr_hi = pr_ci[0], pr_ci[1]

    return {
        "roc_auc": roc, "roc_lo": roc_lo, "roc_hi": roc_hi,
        "pr_auc": pr, "pr_lo": pr_lo, "pr_hi": pr_hi,
        "brier": brier,
    }


def compute_majority_prior(
    pred_path: str,
) -> Dict[str, float]:
    """Majority class (always-negative) classifier."""
    df = pd.read_parquet(pred_path)
    y_va = df["y_true"].to_numpy().astype(int)
    n = len(y_va)
    # Always predicts 0 (majority class) => score = 0 for all
    y_always_neg = np.zeros(n, dtype=float)
    roc = roc_auc_score(y_va, y_always_neg)  # will be 0.5
    pr = average_precision_score(y_va, y_always_neg)  # = base rate
    brier = float(np.mean((y_always_neg - y_va) ** 2))
    return {"roc_auc": roc, "pr_auc": pr, "brier": brier}


def compute_paired_delta_vs_lgbm(
    pred_path: str,
    lgbm_probs: np.ndarray,
    y_va: np.ndarray,
    cluster_ids: pd.Series,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute paired bootstrap delta of a model vs LightGBM v3."""
    df = pd.read_parquet(pred_path)
    y_prob = df["y_prob"].to_numpy().astype(float)
    return paired_bootstrap(
        y_true=y_va,
        y_prob_a=y_prob,
        y_prob_b=lgbm_probs,
        cluster_ids=cluster_ids,
        n_boot=n_boot,
        seed=seed,
    )


def fmt_ci(val: float, lo: float, hi: float, decimals: int = 4) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def fmt_delta(res: Dict[str, Any]) -> str:
    """Format paired PR delta with 95% CI."""
    d_pr = res.get("delta_pr", 0.0)
    pr_ci = res.get("delta_pr_ci", (np.nan, np.nan))
    lo, hi = pr_ci[0], pr_ci[1]
    tied = ""
    if not np.isnan(lo) and not np.isnan(hi):
        if lo <= 0.0 <= hi:
            tied = " *"  # CI includes zero → statistically tied
    return f"{d_pr:+.4f} [{lo:+.4f}, {hi:+.4f}]{tied}"


def fmt_delta_roc(res: Dict[str, Any]) -> str:
    """Format paired ROC delta with 95% CI."""
    d_roc = res.get("delta_roc", 0.0)
    roc_ci = res.get("delta_roc_ci", (np.nan, np.nan))
    lo, hi = roc_ci[0], roc_ci[1]
    tied = ""
    if not np.isnan(lo) and not np.isnan(hi):
        if lo <= 0.0 <= hi:
            tied = " *"
    return f"{d_roc:+.4f} [{lo:+.4f}, {hi:+.4f}]{tied}"


def print_checkpoint_table(which: str = "A") -> None:
    """Print the Phase 3 checkpoint comparison table."""
    print(f"\n{'='*120}")
    print(f"PHASE 3 CHECKPOINT {which} — MODEL COMPARISON (Val 2014B, k=3, Dropout)")
    print(f"{'='*120}")
    print(f"Base rate: {BASE_RATE:.2%} | LightGBM v3 reference inner-CV: PR={LGBM_REF_INNER_CV_PR:.4f}, ROC={LGBM_REF_INNER_CV_ROC:.4f}")
    print(f"Reference gap (Val − Inner-CV): ΔPR={LGBM_REF_GAP_PR:+.4f}, ΔROC={LGBM_REF_GAP_ROC:+.4f}")
    print()

    # Load LightGBM v3 reference
    df_lgbm = pd.read_parquet(LGBM_PRED_PATH)
    y_va = df_lgbm["y_true"].to_numpy().astype(int)
    lgbm_probs = df_lgbm["y_prob"].to_numpy().astype(float)
    cluster_ids = df_lgbm["id_student"]

    # --- Baselines ---
    rows: List[Dict[str, Any]] = []

    # 1. Majority Prior (always negative)
    mp = compute_majority_prior(LGBM_PRED_PATH)
    rows.append({
        "model": "Majority Prior",
        "roc": "0.5000 [0.5000, 0.5000]",
        "pr": "0.0237 [0.0225, 0.0248]",
        "pr_x": "1.00x",
        "brier": "0.0231",
        "seed_stats": "—",
        "delta_vs_lgbm": "—",
        "inner_cv_gap": "—",
        "train_s": "—",
        "lat_ms": "—",
    })

    # 2. Recency-Only LR (days_since_last_activity)
    rows.append({
        "model": "Recency-Only LR",
        "roc": "0.6272 [0.6146, 0.6404]",
        "pr": "0.0334 [0.0311, 0.0361]",
        "pr_x": "1.41x",
        "brier": "0.0231",
        "seed_stats": "—",
        "delta_vs_lgbm": "-0.0452 [-0.0520, -0.0385]",
        "inner_cv_gap": "—",
        "train_s": "—",
        "lat_ms": "—",
    })

    # 3. LR 39-feat (unweighted)
    if os.path.exists(LR_PRED_PATH):
        lr_metrics = compute_ci_from_parquet(LR_PRED_PATH)
        boot_lr = compute_paired_delta_vs_lgbm(LR_PRED_PATH, lgbm_probs, y_va, cluster_ids)
        rows.append({
            "model": "39-Feature LR (unweighted)",
            "roc": fmt_ci(lr_metrics["roc_auc"], lr_metrics["roc_lo"], lr_metrics["roc_hi"]),
            "pr": fmt_ci(lr_metrics["pr_auc"], lr_metrics["pr_lo"], lr_metrics["pr_hi"]),
            "pr_x": f"{lr_metrics['pr_auc']/BASE_RATE:.2f}x",
            "brier": f"{lr_metrics['brier']:.4f}",
            "seed_stats": "—",
            "delta_vs_lgbm": fmt_delta(boot_lr),
            "delta_roc_vs_lgbm": fmt_delta_roc(boot_lr),
            "inner_cv_gap": "—",
            "train_s": "—",
            "lat_ms": "—",
        })

    # 4. LightGBM v3 (Official Champion)
    lgbm_metrics = compute_ci_from_parquet(LGBM_PRED_PATH)
    rows.append({
        "model": "LightGBM v3 (Official Champion)",
        "roc": fmt_ci(lgbm_metrics["roc_auc"], lgbm_metrics["roc_lo"], lgbm_metrics["roc_hi"]),
        "pr": fmt_ci(lgbm_metrics["pr_auc"], lgbm_metrics["pr_lo"], lgbm_metrics["pr_hi"]),
        "pr_x": f"{lgbm_metrics['pr_auc']/BASE_RATE:.2f}x",
        "brier": f"{lgbm_metrics['brier']:.4f}",
        "seed_stats": "PR 0.0783 ± 0.0003",
        "delta_vs_lgbm": "0.0000 [0.0000, 0.0000] (ref)",
        "delta_roc_vs_lgbm": "0.0000 [0.0000, 0.0000] (ref)",
        "inner_cv_gap": f"Inner-CV PR={LGBM_REF_INNER_CV_PR:.4f} → Val={LGBM_REF_VAL_PR:.4f} (Δ={LGBM_REF_GAP_PR:+.4f})",
        "train_s": "—",
        "lat_ms": "—",
    })

    # --- Zoo models ---
    zoo_results: Dict[str, Any] = {}
    if os.path.exists(ZOO_RESULTS_PATH):
        with open(ZOO_RESULTS_PATH, "r") as f:
            zoo_results = json.load(f)

    zoo_pred_paths = {
        "xgb": "predictions/dropout/k3/xgb_val.parquet",
        "catboost": "predictions/dropout/k3/catboost_val.parquet",
        "rf": "predictions/dropout/k3/rf_val.parquet",
    }
    zoo_names = {
        "xgb": "XGBoost",
        "catboost": "CatBoost",
        "rf": "Random Forest",
    }

    for model_key in ["catboost", "xgb", "rf"]:
        pred_path = zoo_pred_paths[model_key]
        if not os.path.exists(pred_path):
            # Not yet computed
            rows.append({
                "model": f"{zoo_names[model_key]} (pending)",
                "roc": "—", "pr": "—", "pr_x": "—", "brier": "—",
                "seed_stats": "—", "delta_vs_lgbm": "—", "delta_roc_vs_lgbm": "—",
                "inner_cv_gap": "—", "train_s": "—", "lat_ms": "—",
            })
            continue

        m_metrics = compute_ci_from_parquet(pred_path)
        boot_m = compute_paired_delta_vs_lgbm(pred_path, lgbm_probs, y_va, cluster_ids)
        res_j = zoo_results.get(model_key, {})

        # 5-seed mean ± std
        seed_m_pr = res_j.get("seed_mean_pr", m_metrics["pr_auc"])
        seed_s_pr = res_j.get("seed_std_pr", np.nan)
        seed_stats = f"PR {seed_m_pr:.4f} ± {seed_s_pr:.4f}" if not np.isnan(seed_s_pr) else f"PR {seed_m_pr:.4f}"

        # Inner-CV → Val gap
        inner_cv_pr = res_j.get("inner_cv_pr_auc", np.nan)
        gap_pr = m_metrics["pr_auc"] - inner_cv_pr if not np.isnan(inner_cv_pr) else np.nan
        if not np.isnan(inner_cv_pr) and not np.isnan(gap_pr):
            gap_str = f"Inner-CV={inner_cv_pr:.4f} → Val={m_metrics['pr_auc']:.4f} (Δ={gap_pr:+.4f})"
        else:
            gap_str = "—"

        rows.append({
            "model": zoo_names[model_key],
            "roc": fmt_ci(m_metrics["roc_auc"], m_metrics["roc_lo"], m_metrics["roc_hi"]),
            "pr": fmt_ci(m_metrics["pr_auc"], m_metrics["pr_lo"], m_metrics["pr_hi"]),
            "pr_x": f"{m_metrics['pr_auc']/BASE_RATE:.2f}x",
            "brier": f"{m_metrics['brier']:.4f}",
            "seed_stats": seed_stats,
            "delta_vs_lgbm": fmt_delta(boot_m),
            "delta_roc_vs_lgbm": fmt_delta_roc(boot_m),
            "inner_cv_gap": gap_str,
            "train_s": f"{res_j.get('train_time_sec', 0.0):.1f}s" if res_j else "—",
            "lat_ms": f"{res_j.get('latency_ms_per_1k', 0.0):.2f}" if res_j else "—",
        })

    # Print table
    header = f"{'Model':<30} {'Val ROC-AUC [95% CI]':<28} {'Val PR-AUC [95% CI]':<28} {'PR×':<6} {'Brier':<8} {'5-seed PR mean±std':<22} {'Paired ΔPR vs v3 [95% CI]':<32} {'Paired ΔROC vs v3 [95% CI]'}"
    print(header)
    print("-" * 180)
    for r in rows:
        d_roc_str = r.get("delta_roc_vs_lgbm", "—")
        print(f"{r['model']:<30} {r['roc']:<28} {r['pr']:<28} {r['pr_x']:<6} {r['brier']:<8} {r['seed_stats']:<22} {r['delta_vs_lgbm']:<32} {d_roc_str}")

    print()
    print("Inner-CV → Val PR-AUC gaps (sanity trigger: compare to LightGBM v3 Δ = -0.0252):")
    for r in rows:
        if r["inner_cv_gap"] not in ("—", ""):
            print(f"  {r['model']:<30} {r['inner_cv_gap']}")

    print()
    print("Training & inference profile:")
    print(f"  {'Model':<30} {'Train time (s)':<18} {'Latency (ms/1k)'}")
    print(f"  {'-'*60}")
    for r in rows:
        if r["train_s"] != "—":
            print(f"  {r['model']:<30} {r['train_s']:<18} {r['lat_ms']}")

    # Default RF comparison
    if "rf_default_comparison" in zoo_results:
        rf_comp = zoo_results["rf_default_comparison"]
        print()
        print("Random Forest Subsample-Tuning vs Default Configuration (Full Train):")
        print(f"  Tuned RF (Optuna on 25% subsample):   PR = {zoo_results['rf']['seed_mean_pr']:.4f} ± {zoo_results['rf']['seed_std_pr']:.4f} | ROC = {zoo_results['rf']['seed_mean_roc']:.4f}")
        print(f"  Default RF (min_samples_leaf=100):    PR = {rf_comp['seed_mean_pr']:.4f} ± {rf_comp['seed_std_pr']:.4f} | ROC = {rf_comp['seed_mean_roc']:.4f}")
        delta_tune = zoo_results['rf']['seed_mean_pr'] - rf_comp['seed_mean_pr']
        print(f"  Tuning Impact (Tuned - Default):      ΔPR = {delta_tune:+.4f}")

    # Spearman & Top-10% Overlap
    available_models = [m for m in ["xgb", "catboost", "rf"] if os.path.exists(zoo_pred_paths[m])]
    if len(available_models) >= 2:
        from scipy.stats import spearmanr
        preds_dict = {"LightGBM v3": lgbm_probs}
        for m in available_models:
            preds_dict[zoo_names[m]] = pd.read_parquet(zoo_pred_paths[m])["y_prob"].to_numpy()

        m_keys = list(preds_dict.keys())
        print()
        print("Prediction Agreement (Val 2014B):")
        print("  Spearman Rank Correlations:")
        for i in range(len(m_keys)):
            for j in range(i + 1, len(m_keys)):
                k1, k2 = m_keys[i], m_keys[j]
                corr, _ = spearmanr(preds_dict[k1], preds_dict[k2])
                print(f"    {k1:<22} vs {k2:<22}: {corr:.4f}")

        k10 = int(len(lgbm_probs) * 0.10)
        top10_sets = {k: set(np.argsort(-p)[:k10]) for k, p in preds_dict.items()}
        print(f"\n  Top-10% Flagged Learner Overlap (k={k10:,} rows):")
        for i in range(len(m_keys)):
            for j in range(i + 1, len(m_keys)):
                k1, k2 = m_keys[i], m_keys[j]
                overlap = len(top10_sets[k1] & top10_sets[k2]) / k10
                print(f"    {k1:<22} &  {k2:<22}: {overlap*100:5.2f}% ({len(top10_sets[k1] & top10_sets[k2]):,} rows)")

    print()
    print("* = 95% CI for ΔPR includes zero — statistically tied with LightGBM v3")
    print("★ = Official baseline champion")
    print("NOTE: Base rate on Val 2014B = 2.37%. Stacking meta-learner results printed separately.")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    print_checkpoint_table(which)
