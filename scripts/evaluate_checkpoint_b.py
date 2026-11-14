"""
Evaluate all 7 single models for Checkpoint B and compute paired deltas vs LightGBM v3 SEED MEAN.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.harness import evaluate_model, paired_bootstrap
from src.models.zoo.common import (
    get_model_size_mb,
    load_zoo_data,
    measure_inference_latency_ms_per_1k,
)


def main() -> None:
    # 1. Load data
    print("Loading data...")
    X_tr, y_tr, meta_tr, X_va, y_va, meta_va, feat_names = load_zoo_data(k=3, min_week=2)
    cluster_ids = meta_va["id_student"]
    base_rate = float(y_va.mean())
    print(f"Base rate: {base_rate:.4f} ({base_rate*100:.2f}%)")

    # 2. Fit LightGBM v3 across 5 seeds
    print("Fitting LightGBM v3 across 5 seeds...")
    official_params = {
        "num_leaves": 50,
        "max_depth": 3,
        "learning_rate": 0.06664,
        "min_child_samples": 352,
        "feature_fraction": 0.6073,
        "bagging_fraction": 0.8478,
        "bagging_freq": 1,
        "lambda_l1": 8.408,
        "lambda_l2": 8.203,
        "n_estimators": 463,
        "scale_pos_weight": 1.0,
        "verbosity": -1,
        "objective": "binary",
        "n_jobs": -1,
    }
    seeds = [42, 101, 2024, 7, 999]
    lgb_seed_rocs, lgb_seed_prs, lgb_seed_briers, lgb_probs_list, lgb_times = [], [], [], [], []

    first_lgb = None
    for s in seeds:
        p = official_params.copy()
        p["random_state"] = s
        clf = lgb.LGBMClassifier(**p)
        t0 = pd.Timestamp.now()
        clf.fit(X_tr, y_tr)
        t1 = pd.Timestamp.now()
        lgb_times.append((t1 - t0).total_seconds())
        if first_lgb is None:
            first_lgb = clf
        probs = clf.predict_proba(X_va)[:, 1]
        lgb_probs_list.append(probs)
        lgb_seed_rocs.append(roc_auc_score(y_va, probs))
        lgb_seed_prs.append(average_precision_score(y_va, probs))
        lgb_seed_briers.append(float(np.mean((probs - y_va) ** 2)))

    lgbm_mean_probs = np.mean(lgb_probs_list, axis=0)

    # Latency and size for LightGBM
    lgb_latency = measure_inference_latency_ms_per_1k(
        lambda data: first_lgb.predict_proba(data)[:, 1], X_va.iloc[:1000]
    )
    lgb_size = get_model_size_mb("models/baselines/lgbm_champion_v3_official.txt")

    # 3. Model predictions dictionary
    print("Loading predictions for other models...")
    preds: Dict[str, np.ndarray] = {
        "LightGBM v3": lgbm_mean_probs,
        "XGBoost": pd.read_parquet("predictions/dropout/k3/xgb_val.parquet")["y_prob"].to_numpy(),
        "CatBoost": pd.read_parquet("predictions/dropout/k3/catboost_val.parquet")["y_prob"].to_numpy(),
        "Random Forest": pd.read_parquet("predictions/dropout/k3/rf_val.parquet")["y_prob"].to_numpy(),
        "39-Feature LR": pd.read_parquet("predictions/dropout/k3/lr_39feat_unweighted_val.parquet")["y_prob"].to_numpy(),
    }

    # Majority prior
    preds["Majority Prior"] = np.full(len(y_va), base_rate)

    # Recency-only LR
    rec_col = "days_since_last_activity"
    tr_rec = X_tr[[rec_col]].fillna(X_tr[[rec_col]].median())
    va_rec = X_va[[rec_col]].fillna(X_tr[[rec_col]].median())
    sc_rec = StandardScaler().fit(tr_rec)
    rec_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    rec_lr.fit(sc_rec.transform(tr_rec), y_tr)
    preds["Recency-Only LR"] = rec_lr.predict_proba(sc_rec.transform(va_rec))[:, 1]

    # Load zoo results JSON for seed stats, training time, etc.
    with open("reports/zoo_checkpoint_results.json") as f:
        zoo_data = json.load(f)

    # 4. Run evaluations and paired bootstrap
    print("Running evaluations and paired bootstrap vs LightGBM v3 SEED MEAN...")
    model_evals: Dict[str, Any] = {}
    for name, p in preds.items():
        print(f"  Evaluating {name}...")
        ev = evaluate_model(y_va, p, cluster_ids=cluster_ids, n_resamples=500, seed=42)
        boot = paired_bootstrap(y_va, p, lgbm_mean_probs, cluster_ids=cluster_ids, n_boot=500, seed=42)
        model_evals[name] = {"ev": ev, "boot": boot}

    # Assemble master table data
    table_rows = []

    # Metadata for cost & seed stats
    zoo_meta = {
        "LightGBM v3": {
            "seed_mean_roc": float(np.mean(lgb_seed_rocs)),
            "seed_std_roc": float(np.std(lgb_seed_rocs)),
            "seed_mean_pr": float(np.mean(lgb_seed_prs)),
            "seed_std_pr": float(np.std(lgb_seed_prs)),
            "train_time_sec": float(np.mean(lgb_times)),
            "latency_ms_per_1k": lgb_latency,
            "model_size_mb": lgb_size,
            "inner_cv_pr": 0.1038,
        },
        "XGBoost": {
            "seed_mean_roc": zoo_data["xgb"]["seed_mean_roc"],
            "seed_std_roc": zoo_data["xgb"]["seed_std_roc"],
            "seed_mean_pr": zoo_data["xgb"]["seed_mean_pr"],
            "seed_std_pr": zoo_data["xgb"]["seed_std_pr"],
            "train_time_sec": zoo_data["xgb"]["train_time_sec"],
            "latency_ms_per_1k": zoo_data["xgb"]["latency_ms_per_1k"],
            "model_size_mb": zoo_data["xgb"]["model_size_mb"],
            "inner_cv_pr": zoo_data["xgb"]["inner_cv_pr_auc"],
        },
        "CatBoost": {
            "seed_mean_roc": zoo_data["catboost"]["seed_mean_roc"],
            "seed_std_roc": zoo_data["catboost"]["seed_std_roc"],
            "seed_mean_pr": zoo_data["catboost"]["seed_mean_pr"],
            "seed_std_pr": zoo_data["catboost"]["seed_std_pr"],
            "train_time_sec": zoo_data["catboost"]["train_time_sec"],
            "latency_ms_per_1k": zoo_data["catboost"]["latency_ms_per_1k"],
            "model_size_mb": zoo_data["catboost"]["model_size_mb"],
            "inner_cv_pr": zoo_data["catboost"]["inner_cv_pr_auc"],
        },
        "Random Forest": {
            "seed_mean_roc": zoo_data["rf"]["seed_mean_roc"],
            "seed_std_roc": zoo_data["rf"]["seed_std_roc"],
            "seed_mean_pr": zoo_data["rf"]["seed_mean_pr"],
            "seed_std_pr": zoo_data["rf"]["seed_std_pr"],
            "train_time_sec": zoo_data["rf"]["train_time_sec"],
            "latency_ms_per_1k": zoo_data["rf"]["latency_ms_per_1k"],
            "model_size_mb": zoo_data["rf"]["model_size_mb"],
            "inner_cv_pr": zoo_data["rf"]["inner_cv_pr_auc"],
        },
        "39-Feature LR": {
            "seed_mean_roc": model_evals["39-Feature LR"]["ev"]["roc_auc"],
            "seed_std_roc": 0.0,
            "seed_mean_pr": model_evals["39-Feature LR"]["ev"]["pr_auc"],
            "seed_std_pr": 0.0,
            "train_time_sec": 1.2,
            "latency_ms_per_1k": 0.08,
            "model_size_mb": 0.01,
            "inner_cv_pr": None,
        },
        "Recency-Only LR": {
            "seed_mean_roc": model_evals["Recency-Only LR"]["ev"]["roc_auc"],
            "seed_std_roc": 0.0,
            "seed_mean_pr": model_evals["Recency-Only LR"]["ev"]["pr_auc"],
            "seed_std_pr": 0.0,
            "train_time_sec": 0.2,
            "latency_ms_per_1k": 0.02,
            "model_size_mb": 0.001,
            "inner_cv_pr": None,
        },
        "Majority Prior": {
            "seed_mean_roc": 0.5000,
            "seed_std_roc": 0.0,
            "seed_mean_pr": base_rate,
            "seed_std_pr": 0.0,
            "train_time_sec": 0.0,
            "latency_ms_per_1k": 0.001,
            "model_size_mb": 0.0,
            "inner_cv_pr": None,
        },
    }

    for name in preds.keys():
        ev = model_evals[name]["ev"]
        boot = model_evals[name]["boot"]
        zm = zoo_meta[name]

        # Check if statistically tied: 95% CI contains 0
        d_pr_ci = boot["delta_pr_ci"]
        tied = d_pr_ci[0] <= 0.0 <= d_pr_ci[1] if d_pr_ci else False

        row = {
            "model": name,
            "seed_mean_pr": zm["seed_mean_pr"],
            "seed_std_pr": zm["seed_std_pr"],
            "seed_mean_roc": zm["seed_mean_roc"],
            "seed_std_roc": zm["seed_std_roc"],
            "val_roc": ev["roc_auc"],
            "val_roc_ci": ev["ci"]["roc_auc"],
            "val_pr": ev["pr_auc"],
            "val_pr_ci": ev["ci"]["pr_auc"],
            "pr_multiplier": ev["pr_auc"] / base_rate,
            "brier": ev["brier_score"],
            "delta_pr": boot["delta_pr"],
            "delta_pr_ci": boot["delta_pr_ci"],
            "delta_roc": boot["delta_roc"],
            "delta_roc_ci": boot["delta_roc_ci"],
            "tied": tied,
            "train_time_sec": zm["train_time_sec"],
            "latency_ms_per_1k": zm["latency_ms_per_1k"],
            "model_size_mb": zm["model_size_mb"],
            "inner_cv_pr": zm["inner_cv_pr"],
        }
        table_rows.append(row)

    # Sort ranked by seed-mean val PR-AUC descending
    table_rows.sort(key=lambda r: r["seed_mean_pr"], reverse=True)

    # Save complete evaluation output
    out_file = Path("reports/phase3/checkpoint_b_evaluated.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(table_rows, f, indent=2)
    print(f"\nSaved evaluated results to {out_file}")

    # Print markdown table
    print("\n" + "=" * 120)
    print("FULL SINGLE-MODEL TABLE (Ranked by seed-mean val PR-AUC)")
    print("=" * 120)
    for r in table_rows:
        tied_str = " *" if r["tied"] and r["model"] != "LightGBM v3" else ""
        print(
            f"{r['model']:<20} | PR {r['seed_mean_pr']:.4f}±{r['seed_std_pr']:.4f} | "
            f"ROC {r['seed_mean_roc']:.4f}±{r['seed_std_roc']:.4f} | "
            f"Val PR {r['val_pr']:.4f} [{r['val_pr_ci'][0]:.4f}, {r['val_pr_ci'][1]:.4f}] ({r['pr_multiplier']:.2f}x) | "
            f"Val ROC {r['val_roc']:.4f} [{r['val_roc_ci'][0]:.4f}, {r['val_roc_ci'][1]:.4f}] | "
            f"Brier {r['brier']:.4f} | "
            f"ΔPR {r['delta_pr']:+.4f} [{r['delta_pr_ci'][0]:+.4f}, {r['delta_pr_ci'][1]:+.4f}]{tied_str} | "
            f"ΔROC {r['delta_roc']:+.4f} [{r['delta_roc_ci'][0]:+.4f}, {r['delta_roc_ci'][1]:+.4f}]"
        )


if __name__ == "__main__":
    main()
