"""
Step 4: LightGBM Main Baseline & Optuna Tuning.
- Native NaN handling, unscaled features.
- Imbalance exploration: unweighted, scale_pos_weight moderate (7.0), and full (neg/pos).
- Optuna search maximizing Validation PR-AUC (average_precision).
- Early stopping patience >= 50.
- Model champion serialized to models/baselines/lgbm_champion_v1.txt.
- Logged to MLflow with full metrics, feature gain importances, and bootstrap CIs.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from src.models.data import load_split
from src.models.metrics import evaluate_predictions
from src.models.tracking import log_model_experiment

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_and_tune_lgbm(
    k: int = 3,
    min_week: int = 2,
    drop_never_active: bool = True,
    n_trials: int = 50,
    n_resamples: int = 500,
) -> Dict[str, Any]:
    """Train LightGBM baseline, explore imbalance weights, run Optuna search."""
    print("=" * 60)
    print("RUNNING STEP 4: LIGHTGBM (MAIN BASELINE) & OPTUNA TUNING")
    print("=" * 60)

    X_train, y_train, meta_train = load_split("train", k=k, min_week=min_week, drop_never_active=drop_never_active)
    X_val, y_val, meta_val = load_split("val", k=k, min_week=min_week, drop_never_active=drop_never_active)
    learner_keys_val = meta_val["code_module"] + "_" + meta_val["code_presentation"] + "_" + meta_val["id_student"].astype(str)

    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    print(f"Calculated train negative/positive ratio: {pos_weight:.2f}")

    # 1. Compare imbalance settings with baseline default parameters
    print("\n[1/3] Comparing Imbalance Weight Settings on Default Parameters...")
    imbalance_configs = {
        "unweighted": 1.0,
        "moderate_weight_7": 7.0,
        "full_pos_weight": pos_weight,
    }
    imbalance_results = {}

    for name, spw in imbalance_configs.items():
        clf = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            scale_pos_weight=spw,
            random_state=42,
            verbosity=-1,
            objective="binary",
        )
        clf.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_probs = clf.predict_proba(X_val)[:, 1]
        ev = evaluate_predictions(y_val, val_probs, learner_keys=learner_keys_val, n_resamples=100)
        imbalance_results[name] = {
            "scale_pos_weight": spw,
            "best_iteration": clf.best_iteration_,
            "point": ev["point"],
            "ci": ev["ci"],
        }
        print(f"  {name:<18} (spw={spw:.1f}) -> Val PR-AUC: {ev['point']['pr_auc']:.4f} | Brier: {ev['point']['brier']:.4f} | ROC-AUC: {ev['point']['roc_auc']:.4f}")

    # 2. Optuna Hyperparameter Search (maximizing Validation PR-AUC)
    print(f"\n[2/3] Running Optuna Search ({n_trials} trials, early_stopping=50)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, 3.0, 5.0]),
            "n_estimators": 1000,
            "random_state": 42,
            "verbosity": -1,
            "objective": "binary",
        }

        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_probs = clf.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_probs)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value
    print(f"Optuna search complete. Best Val PR-AUC: {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")

    # 3. Fit Champion Model
    print(f"\n[3/3] Training Champion LightGBM with Best Hyperparameters & Bootstrap CI ({n_resamples} resamples)...")
    champion_params = {
        "n_estimators": 1500,
        "random_state": 42,
        "verbosity": -1,
        "objective": "binary",
        "bagging_freq": 1,
        **best_params,
    }

    champion = lgb.LGBMClassifier(**champion_params)
    champion.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    champ_probs = champion.predict_proba(X_val)[:, 1]
    champ_eval = evaluate_predictions(y_val, champ_probs, learner_keys=learner_keys_val, n_resamples=n_resamples)

    # Feature Importance by Gain
    booster = champion.booster_
    gains = booster.feature_importance(importance_type="gain")
    feat_names = list(X_train.columns)
    gain_df = pd.DataFrame({"feature": feat_names, "gain": gains}).sort_values("gain", ascending=False)
    total_gain = gain_df["gain"].sum()
    gain_df["pct_gain"] = (gain_df["gain"] / total_gain * 100).round(2)

    # Save champion model artifact
    models_dir = Path("models/baselines")
    models_dir.mkdir(parents=True, exist_ok=True)
    champ_model_path = models_dir / "lgbm_champion_v1.txt"
    booster.save_model(str(champ_model_path))
    print(f"Champion model saved to {champ_model_path}")

    # Log to MLflow
    log_model_experiment(
        experiment_name="dropout",
        run_name="lgbm_champion_v1",
        params={
            "model": "lightgbm",
            "best_iteration": champion.best_iteration_,
            "k": k,
            "min_week": min_week,
            **champion_params,
        },
        metrics={
            "roc_auc": champ_eval["point"]["roc_auc"],
            "pr_auc": champ_eval["point"]["pr_auc"],
            "pr_auc_mult": champ_eval["point"]["pr_auc_mult"],
            "brier": champ_eval["point"]["brier"],
            "mcc": champ_eval["point"]["mcc"],
            "precision_at_10pct": champ_eval["point"]["precision_at_10pct"],
            "recall_at_10pct": champ_eval["point"]["recall_at_10pct"],
        },
        feature_list=feat_names,
        artifacts={"champion_model": str(champ_model_path)},
    )

    return {
        "imbalance_results": imbalance_results,
        "best_params": best_params,
        "champion_eval": champ_eval,
        "gain_df": gain_df,
        "champion_model": champion,
    }


if __name__ == "__main__":
    res = train_and_tune_lgbm(n_trials=30, n_resamples=500)
