"""
Unified CLI driver for Phase 3 Model Zoo (OULAD early dropout prediction).
Supports tuning (Optuna 30 trials with 5-fold inner GroupKFold),
multi-seed training (5 seeds on full Train), and evaluation on Val (2014B).

Usage:
  python -m scripts.run_zoo --model xgb --stage all
  python -m scripts.run_zoo --model catboost --stage all
  python -m scripts.run_zoo --model rf --stage all
  python -m scripts.run_zoo --model et --stage all
  python -m scripts.run_zoo --model mlp --stage all
  python -m scripts.run_zoo --model ensemble
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.harness import evaluate_model, paired_bootstrap
from src.models.zoo.common import (
    get_imputed_pipeline,
    load_zoo_data,
)
from src.models.zoo.xgb import fit_and_evaluate_xgb, tune_xgb
from src.models.zoo.catboost_model import fit_and_evaluate_catboost, tune_catboost
from src.models.zoo.bagging import fit_and_evaluate_bagging, fit_default_rf_comparison, tune_bagging
from src.models.zoo.ensemble import build_rank_average_ensemble, build_stacking_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Model Zoo Driver")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgb", "catboost", "rf", "ensemble", "all"],
        help="Model family to run",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["tune", "fit", "all"],
        help="Stage to execute: tune (Optuna), fit (multi-seed train & val eval), or all",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/zoo.yaml",
        help="Path to zoo configuration YAML",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override number of Optuna trials (default from config: 30)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override seeds list (default from config: [42, 101, 2024, 7, 999])",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_feature_hash(features: List[str]) -> str:
    return hashlib.sha256("::".join(sorted(features)).encode("utf-8")).hexdigest()[:12]


def save_serializable(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python native types."""
    if isinstance(obj, dict):
        return {k: save_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [save_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [save_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    n_trials = args.trials or cfg["experiment"]["n_trials"]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    moderate_pos_weight = float(cfg["experiment"]["moderate_pos_weight"])
    subsample_ratio = float(cfg["experiment"]["rf_et_tune_subsample_ratio"])

    print("=" * 80)
    print("PHASE 3 MODEL ZOO BENCHMARK")
    print(f"Target: {cfg['experiment']['target']} (k={cfg['experiment']['horizon_k']})")
    print(f"Model: {args.model.upper()} | Stage: {args.stage} | Optuna Trials: {n_trials} | Seeds: {seeds}")
    print(f"Shared Moderate Class Weight: {moderate_pos_weight}")
    print("=" * 80)

    # 1. Load data cleanly (Train + Val only, 38 v3 features)
    X_tr, y_tr, meta_tr, X_va, y_va, meta_va, feature_names = load_zoo_data(
        k=cfg["experiment"]["horizon_k"],
        min_week=cfg["experiment"]["min_week"],
        drop_never_active=cfg["experiment"]["drop_never_active"],
    )

    feat_hash = compute_feature_hash(feature_names)
    print(f"Sanctioned v3 Feature Count: {len(feature_names)} | Hash: {feat_hash}")
    assert len(feature_names) == 38, f"Expected 38 features, got {len(feature_names)}"

    # Imputed pipeline for bagging and MLP
    preprocessor = get_imputed_pipeline(X_tr)
    X_tr_imp = preprocessor.transform(X_tr)
    X_va_imp = preprocessor.transform(X_va)
    imp_feature_names = preprocessor.get_feature_names()
    print(f"Imputed features (tree=38, bagging/MLP=43): {X_tr_imp.shape[1]}")

    results_dir = Path("reports")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_file = results_dir / "zoo_checkpoint_results.json"
    existing_results: Dict[str, Any] = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = {}

    zoo_pred_paths = {
        "xgb": "predictions/dropout/k3/xgb_val.parquet",
        "catboost": "predictions/dropout/k3/catboost_val.parquet",
        "rf": "predictions/dropout/k3/rf_val.parquet",
    }

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------
    if args.model in ["xgb", "all"]:
        print("\n" + "#" * 60)
        print("RUNNING XGBOOST")
        print("#" * 60)

        study_path = "models/zoo/xgb_optuna_study.joblib"
        best_params_file = Path("models/zoo/xgb_best_params.json")

        if args.stage in ["tune", "all"] and not os.path.exists(study_path):
            print(f"\n[Step 1/2] Tuning XGBoost over {n_trials} trials (5-fold inner CV)...")
            best_params, chosen_n_estimators, best_val, study = tune_xgb(
                X_tr=X_tr,
                y_tr=y_tr,
                meta_tr=meta_tr,
                n_trials=n_trials,
                seed=seeds[0],
                moderate_pos_weight=moderate_pos_weight,
                study_save_path=study_path,
            )
            print(f"Best Trial #{study.best_trial.number}: Inner-CV PR-AUC = {best_val:.4f}")
            print(f"Optimal Hyperparameters: {best_params}")
            print(f"Median Inner Iterations (Chosen n_estimators): {chosen_n_estimators}")
            with open(best_params_file, "w") as f:
                json.dump({"best_params": best_params, "n_estimators": chosen_n_estimators}, f, indent=2)
        else:
            if best_params_file.exists():
                with open(best_params_file, "r") as f:
                    saved_info = json.load(f)
                    chosen_n_estimators = saved_info["n_estimators"]
                    best_params = saved_info["best_params"]
                print(f"Loaded XGBoost params: n_estimators={chosen_n_estimators}, inner_cv_pr={best_params.get('inner_cv_pr_auc', 'N/A')}")
            else:
                raise FileNotFoundError(f"XGBoost best params not found at {best_params_file}. Run --stage tune first.")

        if args.stage in ["fit", "all"]:
            print(f"\n[Step 2/2] Fitting XGBoost across {len(seeds)} seeds on full Train...")
            xgb_res = fit_and_evaluate_xgb(
                X_tr=X_tr, y_tr=y_tr, meta_tr=meta_tr,
                X_va=X_va, y_va=y_va, meta_va=meta_va,
                best_params=best_params,
                chosen_n_estimators=chosen_n_estimators,
                seeds=seeds,
                moderate_pos_weight=moderate_pos_weight,
            )
            existing_results["xgb"] = save_serializable({
                "model_name": xgb_res["model_name"],
                "best_params": xgb_res["best_params"],
                "chosen_n_estimators": xgb_res["chosen_n_estimators"],
                "inner_cv_pr_auc": xgb_res["inner_cv_pr_auc"],
                "seed_mean_roc": xgb_res["seed_mean_roc"],
                "seed_std_roc": xgb_res["seed_std_roc"],
                "seed_mean_pr": xgb_res["seed_mean_pr"],
                "seed_std_pr": xgb_res["seed_std_pr"],
                "seed_mean_brier": xgb_res["seed_mean_brier"],
                "train_time_sec": xgb_res["train_time_sec"],
                "latency_ms_per_1k": xgb_res["latency_ms_per_1k"],
                "model_size_mb": xgb_res["model_size_mb"],
                "paired_vs_lgbm": xgb_res["paired_vs_lgbm"],
                "top_features": xgb_res["top_features"],
            })
            with open(summary_file, "w") as f:
                json.dump(existing_results, f, indent=2)
            print(f"XGBoost: seed_mean_pr={xgb_res['seed_mean_pr']:.4f} ± {xgb_res['seed_std_pr']:.4f}, seed_mean_roc={xgb_res['seed_mean_roc']:.4f}")

    # ---------------------------------------------------------
    # CatBoost
    # ---------------------------------------------------------
    if args.model in ["catboost", "all"]:
        print("\n" + "#" * 60)
        print("RUNNING CATBOOST")
        print("#" * 60)

        study_path = "models/zoo/catboost_optuna_study.joblib"
        best_params_file = Path("models/zoo/catboost_best_params.json")

        if args.stage in ["tune", "all"] and not os.path.exists(study_path):
            print(f"\n[Step 1/2] Tuning CatBoost over {n_trials} trials (5-fold inner CV, GPU)...")
            best_params, chosen_n_estimators, best_val, study = tune_catboost(
                X_tr=X_tr, y_tr=y_tr, meta_tr=meta_tr,
                n_trials=n_trials, seed=seeds[0],
                moderate_pos_weight=moderate_pos_weight,
                study_save_path=study_path, use_gpu=True,
            )
            print(f"Best Trial #{study.best_trial.number}: Inner-CV PR-AUC = {best_val:.4f}")
            print(f"Optimal Hyperparameters: {best_params}")
            print(f"Median Inner Iterations: {chosen_n_estimators}")
            with open(best_params_file, "w") as f:
                json.dump({"best_params": best_params, "n_estimators": chosen_n_estimators}, f, indent=2)
        else:
            if best_params_file.exists():
                with open(best_params_file, "r") as f:
                    saved_info = json.load(f)
                    chosen_n_estimators = saved_info["n_estimators"]
                    best_params = saved_info["best_params"]
                print(f"Loaded CatBoost params: n_estimators={chosen_n_estimators}")
            else:
                raise FileNotFoundError(f"CatBoost best params not found at {best_params_file}. Run --stage tune first.")

        if args.stage in ["fit", "all"]:
            print(f"\n[Step 2/2] Fitting CatBoost across {len(seeds)} seeds on full Train...")
            cb_res = fit_and_evaluate_catboost(
                X_tr=X_tr, y_tr=y_tr, meta_tr=meta_tr,
                X_va=X_va, y_va=y_va, meta_va=meta_va,
                best_params=best_params,
                chosen_n_estimators=chosen_n_estimators,
                seeds=seeds,
                moderate_pos_weight=moderate_pos_weight,
                use_gpu=True,
            )
            existing_results["catboost"] = save_serializable({
                "model_name": cb_res["model_name"],
                "best_params": cb_res["best_params"],
                "chosen_n_estimators": cb_res["chosen_n_estimators"],
                "inner_cv_pr_auc": cb_res["inner_cv_pr_auc"],
                "seed_mean_roc": cb_res["seed_mean_roc"],
                "seed_std_roc": cb_res["seed_std_roc"],
                "seed_mean_pr": cb_res["seed_mean_pr"],
                "seed_std_pr": cb_res["seed_std_pr"],
                "seed_mean_brier": cb_res["seed_mean_brier"],
                "train_time_sec": cb_res["train_time_sec"],
                "latency_ms_per_1k": cb_res["latency_ms_per_1k"],
                "model_size_mb": cb_res["model_size_mb"],
                "paired_vs_lgbm": cb_res["paired_vs_lgbm"],
                "top_features": cb_res["top_features"],
            })
            with open(summary_file, "w") as f:
                json.dump(existing_results, f, indent=2)
            print(f"CatBoost: seed_mean_pr={cb_res['seed_mean_pr']:.4f} ± {cb_res['seed_std_pr']:.4f}, seed_mean_roc={cb_res['seed_mean_roc']:.4f}")

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------
    if args.model in ["rf", "all"]:
        print("\n" + "#" * 60)
        print("RUNNING RANDOM FOREST")
        print("#" * 60)

        study_path = "models/zoo/rf_optuna_study.joblib"
        best_params_file = Path("models/zoo/rf_best_params.json")

        if args.stage in ["tune", "all"] and not os.path.exists(study_path):
            print(f"\n[Step 1/2] Tuning RF over {n_trials} trials (25% subsample, 5-fold inner CV)...")
            best_params, best_val, study = tune_bagging(
                model_type="rf",
                X_tr_imp=X_tr_imp, y_tr=y_tr, meta_tr=meta_tr,
                n_trials=n_trials, seed=seeds[0],
                subsample_ratio=subsample_ratio,
                study_save_path=study_path,
            )
            print(f"Best Trial #{study.best_trial.number}: Inner-CV PR-AUC = {best_val:.4f}")
            print(f"Optimal Hyperparameters: {best_params}")
            with open(best_params_file, "w") as f:
                json.dump(best_params, f, indent=2)
        else:
            if best_params_file.exists():
                with open(best_params_file, "r") as f:
                    best_params = json.load(f)
                print(f"Loaded RF params: {best_params}")
            else:
                raise FileNotFoundError(f"RF best params not found at {best_params_file}. Run --stage tune first.")

        if args.stage in ["fit", "all"]:
            print(f"\n[Step 2/2] Fitting RF across {len(seeds)} seeds on full Train (43 imputed features)...")
            rf_res = fit_and_evaluate_bagging(
                model_type="rf",
                X_tr_imp=X_tr_imp, y_tr=y_tr, meta_tr=meta_tr,
                X_va_imp=X_va_imp, y_va=y_va, meta_va=meta_va,
                feature_names=imp_feature_names,
                best_params=best_params, seeds=seeds,
            )
            existing_results["rf"] = save_serializable({
                "model_name": rf_res["model_name"],
                "best_params": rf_res["best_params"],
                "inner_cv_pr_auc": rf_res["inner_cv_pr_auc"],
                "seed_mean_roc": rf_res["seed_mean_roc"],
                "seed_std_roc": rf_res["seed_std_roc"],
                "seed_mean_pr": rf_res["seed_mean_pr"],
                "seed_std_pr": rf_res["seed_std_pr"],
                "seed_mean_brier": rf_res["seed_mean_brier"],
                "train_time_sec": rf_res["train_time_sec"],
                "latency_ms_per_1k": rf_res["latency_ms_per_1k"],
                "model_size_mb": rf_res["model_size_mb"],
                "paired_vs_lgbm": rf_res["paired_vs_lgbm"],
                "top_features": rf_res["top_features"],
            })
            with open(summary_file, "w") as f:
                json.dump(existing_results, f, indent=2)
            print(f"RF: seed_mean_pr={rf_res['seed_mean_pr']:.4f} ± {rf_res['seed_std_pr']:.4f}, seed_mean_roc={rf_res['seed_mean_roc']:.4f}")

            # Also fit default configuration on full Train to test tuning impact
            print("\n[Step 2b] Fitting default RF configuration (min_samples_leaf=100) to evaluate tuning impact...")
            default_rf_res = fit_default_rf_comparison(
                X_tr_imp=X_tr_imp, y_tr=y_tr, meta_tr=meta_tr,
                X_va_imp=X_va_imp, y_va=y_va, meta_va=meta_va,
                feature_names=imp_feature_names,
                seeds=seeds,
            )
            existing_results["rf_default_comparison"] = default_rf_res
            with open(summary_file, "w") as f:
                json.dump(existing_results, f, indent=2)
            print(f"Default RF: seed_mean_pr={default_rf_res['seed_mean_pr']:.4f} ± {default_rf_res['seed_std_pr']:.4f}, seed_mean_roc={default_rf_res['seed_mean_roc']:.4f}")
            print(f"Subsample Tuning ΔPR (Tuned - Default): {rf_res['seed_mean_pr'] - default_rf_res['seed_mean_pr']:+.4f}")


    # ---------------------------------------------------------
    # Ensembles
    # ---------------------------------------------------------
    if args.model in ["ensemble", "all"]:
        print("\n" + "#" * 60)
        print("RUNNING ENSEMBLES")
        print("#" * 60)

        print("\n[1/2] Building Rank-Average Ensemble...")
        try:
            rank_res = build_rank_average_ensemble(
                pred_paths=zoo_pred_paths,
                y_va=y_va,
                meta_va=meta_va,
            )
            print(f"Rank-avg: ROC={rank_res['roc_auc']:.4f}, PR={rank_res['pr_auc']:.4f}, Brier={rank_res['brier_score']:.4f}")
            existing_results["rank_avg_ensemble"] = save_serializable({
                "model_name": rank_res["model_name"],
                "models_used": rank_res["models_used"],
                "roc_auc": rank_res["roc_auc"],
                "pr_auc": rank_res["pr_auc"],
                "brier_score": rank_res["brier_score"],
                "paired_vs_lgbm": rank_res["paired_vs_lgbm"],
            })
        except Exception as e:
            print(f"[ensemble rank_avg] skipped — {e}")

        print("\n[2/2] Building Stacking LR Ensemble...")
        try:
            stacking_res = build_stacking_ensemble(
                pred_paths=zoo_pred_paths,
                y_va=y_va,
                meta_va=meta_va,
            )
            print(f"Stacking: ROC={stacking_res['roc_auc']:.4f}, PR={stacking_res['pr_auc']:.4f}, Brier={stacking_res['brier_score']:.4f}")
            print(f"NOTE: {stacking_res['note']}")
            existing_results["stacking_lr"] = save_serializable({
                "model_name": stacking_res["model_name"],
                "models_used": stacking_res["models_used"],
                "roc_auc": stacking_res["roc_auc"],
                "pr_auc": stacking_res["pr_auc"],
                "brier_score": stacking_res["brier_score"],
                "paired_vs_lgbm": stacking_res["paired_vs_lgbm"],
                "note": stacking_res["note"],
            })
        except Exception as e:
            print(f"[ensemble stacking] skipped — {e}")

        with open(summary_file, "w") as f:
            json.dump(existing_results, f, indent=2)

    print("\nRun completed successfully.")


if __name__ == "__main__":
    main()
