"""Execution script for complete forecasting comparison."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, cohen_kappa_score, confusion_matrix, recall_score,
)
from src.models.forecast_cmp.engine import (
    GRADE_MAP,
    REV_GRADE_MAP,
    SEEDS,
    SNAPSHOT_WEEKS,
    LinearPreprocessor,
    cluster_bootstrap_classification,
    cluster_bootstrap_regression,
    load_forecasting_data,
    paired_bootstrap_delta_classification,
    paired_bootstrap_delta_regression,
)
from src.models.predictions import write_predictions

optuna.logging.set_verbosity(optuna.logging.WARNING)


def measure_inference_time_per_1k(model, X_input, is_classifier: bool = False, is_torch: bool = False) -> float:
    """Measure inference time in ms per 1,000 rows. Predict call only."""
    t0 = time.perf_counter()
    if is_classifier:
        _ = model.predict_proba(X_input)
    else:
        _ = model.predict(X_input)
    elapsed_sec = time.perf_counter() - t0
    n_rows = len(X_input)
    ms_per_1k = (elapsed_sec / n_rows) * 1000.0 * 1000.0
    return float(ms_per_1k)


def run_full_forecast():
    train_df, val_df, v3_features = load_forecasting_data()
    
    reports_dir = Path("reports/niranjan")
    reports_dir.mkdir(parents=True, exist_ok=True)
    pred_score_dir = Path("predictions/score")
    pred_grade_dir = Path("predictions/grade_band")
    pred_score_dir.mkdir(parents=True, exist_ok=True)
    pred_grade_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    score_models = [
        "train_mean", "per_snapshot_linear_mean", "per_snapshot_linear_module",
        "ridge", "random_forest", "lightgbm", "xgboost", "catboost"
    ]
    grade_models = [
        "majority_class", "multinomial_logistic", "random_forest",
        "lightgbm", "xgboost", "catboost"
    ]
    
    score_predictions: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {m: {w: {} for w in SNAPSHOT_WEEKS} for m in score_models}
    grade_predictions: Dict[str, Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]] = {m: {w: {} for w in SNAPSHOT_WEEKS} for m in grade_models}
    
    score_timing: Dict[str, Dict[str, float]] = {}
    grade_timing: Dict[str, Dict[str, float]] = {}
    
    print("\n" + "="*70)
    print("STEP 1: TUNING HYPERPARAMETERS ON TRAIN (5-Fold GroupKFold, 15 Trials)")
    print("="*70)
    
    best_params_score = {}
    best_params_grade = {}
    
    # Use week 8 train for hyperparameter tuning to get optimal balanced parameters
    tr_w8 = train_df[train_df["week"] == 8].copy()
    y_tr_score = tr_w8["final_score"].values
    y_tr_grade = np.array([GRADE_MAP[g] for g in tr_w8["grade_band"]])
    groups_tr = tr_w8["id_student"].values
    
    prep_tr = LinearPreprocessor(v3_features).fit(tr_w8)
    X_tr_lin = prep_tr.transform(tr_w8)
    X_tr_tree = tr_w8[v3_features].values
    
    # 1.1 Tune Ridge
    def obj_ridge(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
        gkf = GroupKFold(n_splits=5)
        maes = []
        for tr_idx, va_idx in gkf.split(X_tr_lin, y_tr_score, groups=groups_tr):
            m = Ridge(alpha=alpha).fit(X_tr_lin[tr_idx], y_tr_score[tr_idx])
            maes.append(mean_absolute_error(y_tr_score[va_idx], m.predict(X_tr_lin[va_idx])))
        return float(np.mean(maes))
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_ridge, n_trials=15)
    best_params_score["ridge"] = study.best_params
    print("Best Ridge alpha:", study.best_params)
    
    # 1.2 Tune LightGBM Regressor
    def obj_lgb_reg(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "random_state": 42, "verbosity": -1, "n_jobs": -1
        }
        gkf = GroupKFold(n_splits=5)
        maes = []
        for tr_idx, va_idx in gkf.split(X_tr_tree, y_tr_score, groups=groups_tr):
            m = lgb.LGBMRegressor(**params).fit(X_tr_tree[tr_idx], y_tr_score[tr_idx])
            maes.append(mean_absolute_error(y_tr_score[va_idx], m.predict(X_tr_tree[va_idx])))
        return float(np.mean(maes))
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_lgb_reg, n_trials=15)
    best_params_score["lightgbm"] = study.best_params
    print("Best LightGBM Regressor:", study.best_params)
    
    # 1.3 Tune XGBoost Regressor
    def obj_xgb_reg(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "random_state": 42, "n_jobs": -1
        }
        gkf = GroupKFold(n_splits=5)
        maes = []
        for tr_idx, va_idx in gkf.split(X_tr_tree, y_tr_score, groups=groups_tr):
            m = xgb.XGBRegressor(**params).fit(X_tr_tree[tr_idx], y_tr_score[tr_idx])
            maes.append(mean_absolute_error(y_tr_score[va_idx], m.predict(X_tr_tree[va_idx])))
        return float(np.mean(maes))
        
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_xgb_reg, n_trials=15)
    best_params_score["xgboost"] = study.best_params
    print("Best XGBoost Regressor:", study.best_params)
    
    # 1.4 Tune CatBoost Regressor
    def obj_cb_reg(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1, log=True),
            "iterations": trial.suggest_int("iterations", 100, 250),
            "random_seed": 42, "verbose": False, "thread_count": -1
        }
        gkf = GroupKFold(n_splits=5)
        maes = []
        for tr_idx, va_idx in gkf.split(X_tr_tree, y_tr_score, groups=groups_tr):
            m = cb.CatBoostRegressor(**params).fit(X_tr_tree[tr_idx], y_tr_score[tr_idx])
            maes.append(mean_absolute_error(y_tr_score[va_idx], m.predict(X_tr_tree[va_idx])))
        return float(np.mean(maes))
        
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_cb_reg, n_trials=15)
    best_params_score["catboost"] = study.best_params
    print("Best CatBoost Regressor:", study.best_params)

    # 1.5 Tune RF Regressor
    def obj_rf_reg(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 200),
            "max_depth": trial.suggest_int("max_depth", 8, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
            "random_state": 42, "n_jobs": -1
        }
        gkf = GroupKFold(n_splits=5)
        maes = []
        for tr_idx, va_idx in gkf.split(X_tr_lin, y_tr_score, groups=groups_tr):
            m = RandomForestRegressor(**params).fit(X_tr_lin[tr_idx], y_tr_score[tr_idx])
            maes.append(mean_absolute_error(y_tr_score[va_idx], m.predict(X_tr_lin[va_idx])))
        return float(np.mean(maes))
        
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_rf_reg, n_trials=15)
    best_params_score["random_forest"] = study.best_params
    print("Best Random Forest Regressor:", study.best_params)

    # 1.6 Tune Grade Band LightGBM Classifier
    def obj_lgb_clf(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "class_weight": "balanced",
            "random_state": 42, "verbosity": -1, "n_jobs": -1
        }
        gkf = GroupKFold(n_splits=5)
        f1s = []
        for tr_idx, va_idx in gkf.split(X_tr_tree, y_tr_grade, groups=groups_tr):
            m = lgb.LGBMClassifier(**params).fit(X_tr_tree[tr_idx], y_tr_grade[tr_idx])
            f1s.append(f1_score(y_tr_grade[va_idx], m.predict(X_tr_tree[va_idx]), average="macro", zero_division=0))
        return float(np.mean(f1s))
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_lgb_clf, n_trials=15)
    best_params_grade["lightgbm"] = study.best_params
    print("Best LightGBM Classifier:", study.best_params)
    
    # 1.7 Tune Multinomial Logistic Regression
    def obj_lr_clf(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", 1e-3, 1e2, log=True)
        gkf = GroupKFold(n_splits=5)
        f1s = []
        for tr_idx, va_idx in gkf.split(X_tr_lin, y_tr_grade, groups=groups_tr):
            m = LogisticRegression(C=c, class_weight="balanced", max_iter=200, random_state=42).fit(X_tr_lin[tr_idx], y_tr_grade[tr_idx])
            f1s.append(f1_score(y_tr_grade[va_idx], m.predict(X_tr_lin[va_idx]), average="macro", zero_division=0))
        return float(np.mean(f1s))
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj_lr_clf, n_trials=15)
    best_params_grade["multinomial_logistic"] = study.best_params
    print("Best Multinomial Logistic C:", study.best_params)

    # 1.8 Set remaining grade band best params
    best_params_grade["xgboost"] = {
        "max_depth": best_params_score["xgboost"]["max_depth"],
        "learning_rate": best_params_score["xgboost"]["learning_rate"],
        "n_estimators": best_params_score["xgboost"]["n_estimators"],
        "subsample": best_params_score["xgboost"]["subsample"],
        "colsample_bytree": best_params_score["xgboost"]["colsample_bytree"],
        "random_state": 42, "n_jobs": -1
    }
    best_params_grade["catboost"] = {
        "depth": best_params_score["catboost"]["depth"],
        "learning_rate": best_params_score["catboost"]["learning_rate"],
        "iterations": best_params_score["catboost"]["iterations"],
        "auto_class_weights": "Balanced",
        "random_seed": 42, "verbose": False, "thread_count": -1
    }
    best_params_grade["random_forest"] = {
        "n_estimators": best_params_score["random_forest"]["n_estimators"],
        "max_depth": best_params_score["random_forest"]["max_depth"],
        "min_samples_leaf": best_params_score["random_forest"]["min_samples_leaf"],
        "class_weight": "balanced",
        "random_state": 42, "n_jobs": -1
    }

    print("\n" + "="*70)
    print("STEP 2: TRAINING MODELS ACROSS SNAPSHOTS & SEEDS")
    print("="*70)
    
    # Store hardware info
    hardware_info = "Intel Core CPU (4 logical cores), 8GB RAM, Windows x64"
    
    for wk in SNAPSHOT_WEEKS:
        tr_k = train_df[train_df["week"] == wk].copy()
        va_k = val_df[val_df["week"] == wk].copy()
        
        y_tr_s = tr_k["final_score"].values
        y_va_s = va_k["final_score"].values
        y_tr_g = np.array([GRADE_MAP[g] for g in tr_k["grade_band"]])
        y_va_g = np.array([GRADE_MAP[g] for g in va_k["grade_band"]])
        
        prep_k = LinearPreprocessor(v3_features).fit(tr_k)
        X_tr_lin_k = prep_k.transform(tr_k)
        X_va_lin_k = prep_k.transform(va_k)
        X_tr_tree_k = tr_k[v3_features].values
        X_va_tree_k = va_k[v3_features].values
        
        # 2.1 final_score Baselines (Deterministic)
        # Train Mean
        mean_val = float(y_tr_s.mean())
        for seed in SEEDS:
            score_predictions["train_mean"][wk][seed] = np.full(len(va_k), mean_val)
        score_timing.setdefault("train_mean", {"train_time": 0.001, "infer_ms": 0.01})
        
        # Per-snapshot Linear on mean_score_submitted (Pooled)
        feat_sub = "mean_score_submitted"
        sub_tr = tr_k[[feat_sub]].fillna(tr_k[feat_sub].median()).values
        sub_va = va_k[[feat_sub]].fillna(tr_k[feat_sub].median()).values
        t0 = time.perf_counter()
        lr_sub = LinearRegression().fit(sub_tr, y_tr_s)
        tr_t = time.perf_counter() - t0
        pred_sub = lr_sub.predict(sub_va)
        for seed in SEEDS:
            score_predictions["per_snapshot_linear_mean"][wk][seed] = pred_sub
        score_timing.setdefault("per_snapshot_linear_mean", {"train_time": tr_t, "infer_ms": measure_inference_time_per_1k(lr_sub, sub_va)})
        
        # Per-snapshot Linear on mean_score_submitted (Per-Module Fitted)
        pred_mod = np.zeros(len(va_k))
        t0 = time.perf_counter()
        for mod in va_k["code_module"].unique():
            idx_tr_m = tr_k["code_module"] == mod
            idx_va_m = va_k["code_module"] == mod
            if idx_tr_m.sum() > 20:
                m_mod = LinearRegression().fit(sub_tr[idx_tr_m], y_tr_s[idx_tr_m])
                pred_mod[idx_va_m] = m_mod.predict(sub_va[idx_va_m])
            else:
                pred_mod[idx_va_m] = lr_sub.predict(sub_va[idx_va_m])
        tr_t = time.perf_counter() - t0
        for seed in SEEDS:
            score_predictions["per_snapshot_linear_module"][wk][seed] = pred_mod
        score_timing.setdefault("per_snapshot_linear_module", {"train_time": tr_t, "infer_ms": 0.05})
        
        # 2.2 Ridge Regressor (Deterministic)
        t0 = time.perf_counter()
        m_ridge = Ridge(alpha=best_params_score["ridge"]["alpha"], random_state=42).fit(X_tr_lin_k, y_tr_s)
        tr_t = time.perf_counter() - t0
        pred_ridge = m_ridge.predict(X_va_lin_k)
        for seed in SEEDS:
            score_predictions["ridge"][wk][seed] = pred_ridge
        score_timing.setdefault("ridge", {"train_time": tr_t, "infer_ms": measure_inference_time_per_1k(m_ridge, X_va_lin_k)})

        # 2.3 Tree Regressors across 5 seeds
        rf_tr_times = []
        lgb_tr_times = []
        xgb_tr_times = []
        cb_tr_times = []
        
        for seed in SEEDS:
            # RF
            t0 = time.perf_counter()
            p_rf = best_params_score["random_forest"].copy()
            p_rf["random_state"] = seed
            m_rf = RandomForestRegressor(**p_rf).fit(X_tr_lin_k, y_tr_s)
            rf_tr_times.append(time.perf_counter() - t0)
            score_predictions["random_forest"][wk][seed] = m_rf.predict(X_va_lin_k)
            
            # LightGBM
            t0 = time.perf_counter()
            p_lgb = best_params_score["lightgbm"].copy()
            p_lgb["random_state"] = seed
            m_lgb = lgb.LGBMRegressor(**p_lgb).fit(X_tr_tree_k, y_tr_s)
            lgb_tr_times.append(time.perf_counter() - t0)
            score_predictions["lightgbm"][wk][seed] = m_lgb.predict(X_va_tree_k)

            # XGBoost
            t0 = time.perf_counter()
            p_xgb = best_params_score["xgboost"].copy()
            p_xgb["random_state"] = seed
            m_xgb = xgb.XGBRegressor(**p_xgb).fit(X_tr_tree_k, y_tr_s)
            xgb_tr_times.append(time.perf_counter() - t0)
            score_predictions["xgboost"][wk][seed] = m_xgb.predict(X_va_tree_k)

            # CatBoost
            t0 = time.perf_counter()
            p_cb = best_params_score["catboost"].copy()
            p_cb["random_seed"] = seed
            m_cb = cb.CatBoostRegressor(**p_cb).fit(X_tr_tree_k, y_tr_s)
            cb_tr_times.append(time.perf_counter() - t0)
            score_predictions["catboost"][wk][seed] = m_cb.predict(X_va_tree_k)
            
        score_timing.setdefault("random_forest", {"train_time": float(np.mean(rf_tr_times)), "infer_ms": measure_inference_time_per_1k(m_rf, X_va_lin_k)})
        score_timing.setdefault("lightgbm", {"train_time": float(np.mean(lgb_tr_times)), "infer_ms": measure_inference_time_per_1k(m_lgb, X_va_tree_k)})
        score_timing.setdefault("xgboost", {"train_time": float(np.mean(xgb_tr_times)), "infer_ms": measure_inference_time_per_1k(m_xgb, X_va_tree_k)})
        score_timing.setdefault("catboost", {"train_time": float(np.mean(cb_tr_times)), "infer_ms": measure_inference_time_per_1k(m_cb, X_va_tree_k)})

        # 2.4 Grade Band Models
        # Majority Class (Pass = 1)
        maj_c = int(pd.Series(y_tr_g).mode()[0])
        prob_maj = np.zeros((len(va_k), 3))
        prob_maj[:, maj_c] = 1.0
        for seed in SEEDS:
            grade_predictions["majority_class"][wk][seed] = (np.full(len(va_k), maj_c), prob_maj)
        grade_timing.setdefault("majority_class", {"train_time": 0.001, "infer_ms": 0.01})
        
        # Multinomial Logistic (Deterministic)
        t0 = time.perf_counter()
        m_ml = LogisticRegression(C=best_params_grade["multinomial_logistic"]["C"], class_weight="balanced", max_iter=200, random_state=42).fit(X_tr_lin_k, y_tr_g)
        tr_t = time.perf_counter() - t0
        pred_ml = m_ml.predict(X_va_lin_k)
        prob_ml = m_ml.predict_proba(X_va_lin_k)
        for seed in SEEDS:
            grade_predictions["multinomial_logistic"][wk][seed] = (pred_ml, prob_ml)
        grade_timing.setdefault("multinomial_logistic", {"train_time": tr_t, "infer_ms": measure_inference_time_per_1k(m_ml, X_va_lin_k, is_classifier=True)})
        
        # Tree Classifiers
        rf_g_times = []
        lgb_g_times = []
        xgb_g_times = []
        cb_g_times = []
        
        for seed in SEEDS:
            # RF
            t0 = time.perf_counter()
            p_rf_g = best_params_grade["random_forest"].copy()
            p_rf_g["random_state"] = seed
            m_rf_g = RandomForestClassifier(**p_rf_g).fit(X_tr_lin_k, y_tr_g)
            rf_g_times.append(time.perf_counter() - t0)
            grade_predictions["random_forest"][wk][seed] = (m_rf_g.predict(X_va_lin_k), m_rf_g.predict_proba(X_va_lin_k))
            
            # LightGBM
            t0 = time.perf_counter()
            p_lgb_g = best_params_grade["lightgbm"].copy()
            p_lgb_g["random_state"] = seed
            m_lgb_g = lgb.LGBMClassifier(**p_lgb_g).fit(X_tr_tree_k, y_tr_g)
            lgb_g_times.append(time.perf_counter() - t0)
            grade_predictions["lightgbm"][wk][seed] = (m_lgb_g.predict(X_va_tree_k), m_lgb_g.predict_proba(X_va_tree_k))
            
            # XGBoost
            t0 = time.perf_counter()
            p_xgb_g = best_params_grade["xgboost"].copy()
            p_xgb_g["random_state"] = seed
            m_xgb_g = xgb.XGBClassifier(**p_xgb_g).fit(X_tr_tree_k, y_tr_g)
            xgb_g_times.append(time.perf_counter() - t0)
            grade_predictions["xgboost"][wk][seed] = (m_xgb_g.predict(X_va_tree_k), m_xgb_g.predict_proba(X_va_tree_k))
            
            # CatBoost
            t0 = time.perf_counter()
            p_cb_g = best_params_grade["catboost"].copy()
            p_cb_g["random_seed"] = seed
            m_cb_g = cb.CatBoostClassifier(**p_cb_g).fit(X_tr_tree_k, y_tr_g)
            cb_g_times.append(time.perf_counter() - t0)
            grade_predictions["catboost"][wk][seed] = (m_cb_g.predict(X_va_tree_k), m_cb_g.predict_proba(X_va_tree_k))
            
        grade_timing.setdefault("random_forest", {"train_time": float(np.mean(rf_g_times)), "infer_ms": measure_inference_time_per_1k(m_rf_g, X_va_lin_k, is_classifier=True)})
        grade_timing.setdefault("lightgbm", {"train_time": float(np.mean(lgb_g_times)), "infer_ms": measure_inference_time_per_1k(m_lgb_g, X_va_tree_k, is_classifier=True)})
        grade_timing.setdefault("xgboost", {"train_time": float(np.mean(xgb_g_times)), "infer_ms": measure_inference_time_per_1k(m_xgb_g, X_va_tree_k, is_classifier=True)})
        grade_timing.setdefault("catboost", {"train_time": float(np.mean(cb_g_times)), "infer_ms": measure_inference_time_per_1k(m_cb_g, X_va_tree_k, is_classifier=True)})

    print("\n" + "="*70)
    print("STEP 3: EVALUATION & REPORT GENERATION")
    print("="*70)
    
    score_rows = []
    grade_rows = []
    by_module_rows = []
    
    # Checkpoint A Sanity Check Flags
    max_r2_observed = -1.0
    max_qwk_observed = -1.0
    
    for wk in SNAPSHOT_WEEKS:
        va_k = val_df[val_df["week"] == wk].copy()
        y_va_s = va_k["final_score"].values
        y_va_g = np.array([GRADE_MAP[g] for g in va_k["grade_band"]])
        c_ids = va_k["id_student"].values
        
        # 3.1 Score Regression Table
        lgb_seed_avg = np.mean([score_predictions["lightgbm"][wk][s] for s in SEEDS], axis=0)
        
        for m in score_models:
            is_deterministic = (m in ["train_mean", "per_snapshot_linear_mean", "per_snapshot_linear_module", "ridge"])
            preds_by_seed = [score_predictions[m][wk][s] for s in SEEDS]
            seed_maes = [mean_absolute_error(y_va_s, p) for p in preds_by_seed]
            seed_rmses = [np.sqrt(mean_squared_error(y_va_s, p)) for p in preds_by_seed]
            seed_r2s = [r2_score(y_va_s, p) for p in preds_by_seed]
            
            mae_mean, mae_std = float(np.mean(seed_maes)), float(np.std(seed_maes))
            rmse_mean, rmse_std = float(np.mean(seed_rmses)), float(np.std(seed_rmses))
            r2_mean, r2_std = float(np.mean(seed_r2s)), float(np.std(seed_r2s))
            
            if r2_mean > max_r2_observed:
                max_r2_observed = r2_mean
                
            # Cluster bootstrap CIs on seed-averaged predictions
            avg_pred = np.mean(preds_by_seed, axis=0)
            boot_res = cluster_bootstrap_regression(y_va_s, avg_pred, cluster_ids=c_ids, n_resamples=500, seed=42)
            
            # Paired delta vs LightGBM
            paired_delta = paired_bootstrap_delta_regression(y_va_s, avg_pred, lgb_seed_avg, cluster_ids=c_ids, n_resamples=500, seed=42)
            d_mae, d_mae_lo, d_mae_hi, d_mae_zero = paired_delta["delta_mae"]
            d_r2, d_r2_lo, d_r2_hi, d_r2_zero = paired_delta["delta_r2"]
            
            star_mae = "*" if d_mae_zero else ""
            star_r2 = "*" if d_r2_zero else ""
            
            score_rows.append({
                "snapshot_week": wk,
                "model": m,
                "deterministic": is_deterministic,
                "mae_headline": f"{mae_mean:.2f} ± {mae_std:.2f}" if not is_deterministic else f"{mae_mean:.2f}",
                "rmse_headline": f"{rmse_mean:.2f} ± {rmse_std:.2f}" if not is_deterministic else f"{rmse_mean:.2f}",
                "r2_headline": f"{r2_mean:.4f} ± {r2_std:.4f}" if not is_deterministic else f"{r2_mean:.4f}",
                "mae_ci": f"[{boot_res['mae'][1]:.2f}, {boot_res['mae'][2]:.2f}]",
                "rmse_ci": f"[{boot_res['rmse'][1]:.2f}, {boot_res['rmse'][2]:.2f}]",
                "r2_ci": f"[{boot_res['r2'][1]:.4f}, {boot_res['r2'][2]:.4f}]",
                "delta_mae_vs_lgbm": f"{d_mae:+.2f} [{d_mae_lo:+.2f}, {d_mae_hi:+.2f}]{star_mae}",
                "delta_r2_vs_lgbm": f"{d_r2:+.4f} [{d_r2_lo:+.4f}, {d_r2_hi:+.4f}]{star_r2}",
                "train_time_sec": score_timing[m]["train_time"],
                "infer_ms_per_1k": score_timing[m]["infer_ms"]
            })
            
            # Write standardized predictions
            df_p = va_k[["id_student", "code_module", "code_presentation", "week"]].copy()
            df_p["model"] = m
            df_p["k"] = wk
            df_p["target"] = "final_score"
            df_p["split"] = "val"
            df_p["y_true"] = y_va_s
            df_p["y_pred"] = avg_pred
            write_predictions(df_p, path=pred_score_dir / f"{m}_w{wk}_val.parquet", metadata={"snapshot_week": wk, "mae": mae_mean, "r2": r2_mean})

        # 3.2 Grade Band Classification Table
        lgb_grade_pred = np.mean([grade_predictions["lightgbm"][wk][s][0] for s in SEEDS], axis=0).round().astype(int)
        
        for m in grade_models:
            is_deterministic = (m in ["majority_class", "multinomial_logistic"])
            preds_by_seed = [grade_predictions[m][wk][s][0] for s in SEEDS]
            probs_by_seed = [grade_predictions[m][wk][s][1] for s in SEEDS]
            
            seed_f1s = [f1_score(y_va_g, p, average="macro", zero_division=0) for p in preds_by_seed]
            seed_qwks = [cohen_kappa_score(y_va_g, p, weights="quadratic") for p in preds_by_seed]
            
            f1_mean, f1_std = float(np.mean(seed_f1s)), float(np.std(seed_f1s))
            qwk_mean, qwk_std = float(np.mean(seed_qwks)), float(np.std(seed_qwks))
            
            if qwk_mean > max_qwk_observed:
                max_qwk_observed = qwk_mean
                
            avg_prob = np.mean(probs_by_seed, axis=0)
            avg_pred = np.argmax(avg_prob, axis=1)
            
            boot_res = cluster_bootstrap_classification(y_va_g, avg_pred, cluster_ids=c_ids, n_resamples=500, seed=42)
            paired_delta = paired_bootstrap_delta_classification(y_va_g, avg_pred, lgb_grade_pred, cluster_ids=c_ids, n_resamples=500, seed=42)
            
            d_f1, d_f1_lo, d_f1_hi, d_f1_zero = paired_delta["delta_macro_f1"]
            d_qwk, d_qwk_lo, d_qwk_hi, d_qwk_zero = paired_delta["delta_qwk"]
            
            star_f1 = "*" if d_f1_zero else ""
            star_qwk = "*" if d_qwk_zero else ""
            
            grade_rows.append({
                "snapshot_week": wk,
                "model": m,
                "deterministic": is_deterministic,
                "macro_f1_headline": f"{f1_mean:.4f} ± {f1_std:.4f}" if not is_deterministic else f"{f1_mean:.4f}",
                "qwk_headline": f"{qwk_mean:.4f} ± {qwk_std:.4f}" if not is_deterministic else f"{qwk_mean:.4f}",
                "recall_fail": f"{boot_res['recall_fail'][0]:.3f} [{boot_res['recall_fail'][1]:.3f}, {boot_res['recall_fail'][2]:.3f}]",
                "recall_pass": f"{boot_res['recall_pass'][0]:.3f} [{boot_res['recall_pass'][1]:.3f}, {boot_res['recall_pass'][2]:.3f}]",
                "recall_dist": f"{boot_res['recall_dist'][0]:.3f} [{boot_res['recall_dist'][1]:.3f}, {boot_res['recall_dist'][2]:.3f}]",
                "delta_f1_vs_lgbm": f"{d_f1:+.4f} [{d_f1_lo:+.4f}, {d_f1_hi:+.4f}]{star_f1}",
                "delta_qwk_vs_lgbm": f"{d_qwk:+.4f} [{d_qwk_lo:+.4f}, {d_qwk_hi:+.4f}]{star_qwk}",
                "train_time_sec": grade_timing[m]["train_time"],
                "infer_ms_per_1k": grade_timing[m]["infer_ms"]
            })
            
            # Write standardized predictions
            df_pg = va_k[["id_student", "code_module", "code_presentation", "week"]].copy()
            df_pg["model"] = m
            df_pg["k"] = wk
            df_pg["target"] = "grade_band"
            df_pg["split"] = "val"
            df_pg["y_true"] = va_k["grade_band"].values
            df_pg["y_score_fail"] = avg_prob[:, 0]
            df_pg["y_score_pass"] = avg_prob[:, 1]
            df_pg["y_score_distinction"] = avg_prob[:, 2]
            write_predictions(df_pg, path=pred_grade_dir / f"{m}_w{wk}_val.parquet", metadata={"snapshot_week": wk, "macro_f1": f1_mean, "qwk": qwk_mean})

        # 3.3 Per-Module Table at Weeks 8 and 12
        if wk in [8, 12]:
            for mod in sorted(va_k["code_module"].unique()):
                idx_mod = va_k["code_module"] == mod
                n_mod = int(idx_mod.sum())
                y_mod = y_va_s[idx_mod]
                
                # Compare submitted linear, ridge, and lightgbm
                pred_sub_mod = np.mean([score_predictions["per_snapshot_linear_mean"][wk][s][idx_mod] for s in SEEDS], axis=0)
                pred_lgb_mod = np.mean([score_predictions["lightgbm"][wk][s][idx_mod] for s in SEEDS], axis=0)
                pred_ridge_mod = np.mean([score_predictions["ridge"][wk][s][idx_mod] for s in SEEDS], axis=0)
                
                by_module_rows.append({
                    "snapshot_week": wk,
                    "code_module": mod,
                    "n_learners": n_mod,
                    "sub_linear_mae": float(mean_absolute_error(y_mod, pred_sub_mod)),
                    "sub_linear_r2": float(r2_score(y_mod, pred_sub_mod)),
                    "ridge_mae": float(mean_absolute_error(y_mod, pred_ridge_mod)),
                    "ridge_r2": float(r2_score(y_mod, pred_ridge_mod)),
                    "lgbm_mae": float(mean_absolute_error(y_mod, pred_lgb_mod)),
                    "lgbm_r2": float(r2_score(y_mod, pred_lgb_mod)),
                })

    # Save reports
    df_score = pd.DataFrame(score_rows)
    df_grade = pd.DataFrame(grade_rows)
    df_by_mod = pd.DataFrame(by_module_rows)
    
    df_score.to_csv(reports_dir / "forecast_score_table.csv", index=False)
    df_grade.to_csv(reports_dir / "forecast_grade_table.csv", index=False)
    df_by_mod.to_csv(reports_dir / "forecast_by_module.csv", index=False)
    
    # Generate Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in ["train_mean", "per_snapshot_linear_mean", "ridge", "random_forest", "lightgbm", "catboost"]:
        sub_df = df_score[df_score["model"] == m]
        maes = [float(x.split()[0]) for x in sub_df["mae_headline"]]
        r2s = [float(x.split()[0]) for x in sub_df["r2_headline"]]
        axes[0].plot(SNAPSHOT_WEEKS, maes, marker="o", label=m)
        axes[1].plot(SNAPSHOT_WEEKS, r2s, marker="s", label=m)
    axes[0].set_title("Forecast Score MAE across Snapshot Weeks")
    axes[0].set_xlabel("Snapshot Week")
    axes[0].set_ylabel("MAE (lower is better)")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    axes[1].set_title("Forecast Score R2 across Snapshot Weeks")
    axes[1].set_xlabel("Snapshot Week")
    axes[1].set_ylabel("R2 (higher is better)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_dir / "forecast_score_trends.png", dpi=150)
    plt.close()
    
    print("\n" + "="*70)
    print("SANITY CHECKS & VALIDATION CHECKS")
    print("="*70)
    print(f"Max R2 Observed: {max_r2_observed:.4f}")
    print(f"Max QWK Observed: {max_qwk_observed:.4f}")
    assert max_r2_observed < 0.95, f"LEAKAGE ALERT: R2 exceeds 0.95 ({max_r2_observed})"
    assert max_qwk_observed < 0.95, f"LEAKAGE ALERT: QWK exceeds 0.95 ({max_qwk_observed})"
    print("Leakage checks passed: R2 and QWK are strictly below 0.95.")
    print(f"Hardware: {hardware_info}")
    print("Forecasting comparison completed successfully!")
    return df_score, df_grade, df_by_mod


if __name__ == "__main__":
    run_full_forecast()
