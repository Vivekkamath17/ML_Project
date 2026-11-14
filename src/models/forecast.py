"""
Step 7: Performance Forecasting Baselines.
Population: Non-withdrawn learners only (forecasts are conditional on remaining enrolled).

Tasks:
1. final_score regression:
   - Baselines: Train Mean, weighted_score_to_date alone.
   - LightGBM Regressor.
   - Metrics: MAE, RMSE, R2.
   - Evaluated on all rows weighted by 1/(rows per learner), and snapshots at weeks 4, 8, 12.
   - By-week trajectory curves.

2. grade_band (Fail, Pass, Distinction) 3-class classification:
   - Baseline: Majority class (Pass).
   - LightGBM Multiclass with class weighting.
   - Metrics: macro-F1, per-class recall, quadratic weighted kappa, confusion matrix.
   - Evaluated on all rows weighted by 1/(rows per learner), and snapshots at weeks 4, 8, 12.
   - By-week trajectory curves.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
)

from src.models.data import load_split


def compute_learner_weights(meta_df: pd.DataFrame) -> np.ndarray:
    """Compute row weight w_i = 1 / (rows per learner) so every learner has equal mass."""
    learner_keys = meta_df["code_module"] + "_" + meta_df["code_presentation"] + "_" + meta_df["id_student"].astype(str)
    counts = learner_keys.map(learner_keys.value_counts())
    weights = (1.0 / counts).to_numpy()
    return weights


def run_score_forecasting() -> Dict[str, Any]:
    """Train and evaluate final_score regression models."""
    print("=" * 60)
    print("RUNNING STEP 7.1: FINAL_SCORE REGRESSION")
    print("=" * 60)

    # Load non-withdrawn rows only
    X_tr, y_tr, meta_tr = load_split("train", non_withdrawn_only=True, target_col="final_score")
    X_va, y_va, meta_va = load_split("val", non_withdrawn_only=True, target_col="final_score")

    w_tr = compute_learner_weights(meta_tr)
    w_va = compute_learner_weights(meta_va)

    # 1. Baseline A: Train Mean
    train_mean = float(np.average(y_tr, weights=w_tr))
    pred_mean_va = np.full(len(y_va), train_mean)

    # 2. Baseline B: weighted_score_to_date alone (Linear Regression)
    feat_b = "weighted_score_to_date"
    # Impute missing with 0 for assessment to date
    x_tr_b = X_tr[[feat_b]].fillna(0.0).to_numpy()
    x_va_b = X_va[[feat_b]].fillna(0.0).to_numpy()
    lr_b = LinearRegression().fit(x_tr_b, y_tr, sample_weight=w_tr)
    pred_b_va = lr_b.predict(x_va_b)

    # 3. LightGBM Regressor (39 features)
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=5,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1,
        objective="regression",
    )
    lgb_reg.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_X=X_va,
        eval_y=y_va,
        eval_sample_weight=[w_va],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_lgb_va = lgb_reg.predict(X_va)

    def eval_reg(y_true, y_pred, sample_weight=None) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))
        r2 = float(r2_score(y_true, y_pred, sample_weight=sample_weight))
        return {"mae": mae, "rmse": rmse, "r2": r2}

    # Summary table across all rows (weighted)
    all_mean = eval_reg(y_va, pred_mean_va, sample_weight=w_va)
    all_b = eval_reg(y_va, pred_b_va, sample_weight=w_va)
    all_lgb = eval_reg(y_va, pred_lgb_va, sample_weight=w_va)

    print("\n--- Final Score Regression Results (All Val Rows, Weighted by 1/rows-per-learner) ---")
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print(f"{'Baseline: Train Mean':<30} {all_mean['mae']:>8.2f} {all_mean['rmse']:>8.2f} {all_mean['r2']:>8.4f}")
    print(f"{'Baseline: weighted_score_to_date':<30} {all_b['mae']:>8.2f} {all_b['rmse']:>8.2f} {all_b['r2']:>8.4f}")
    print(f"{'LightGBM Regressor (39 feat)':<30} {all_lgb['mae']:>8.2f} {all_lgb['rmse']:>8.2f} {all_lgb['r2']:>8.4f}")

    # Snapshots at weeks 4, 8, 12 (One row per learner)
    snapshots = {}
    for wk in [4, 8, 12]:
        mask_wk = meta_va["week"] == wk
        if mask_wk.sum() > 0:
            y_wk = y_va[mask_wk]
            p_mean_wk = pred_mean_va[mask_wk]
            p_b_wk = pred_b_va[mask_wk]
            p_lgb_wk = pred_lgb_va[mask_wk]
            snapshots[wk] = {
                "n_learners": int(mask_wk.sum()),
                "mean_baseline": eval_reg(y_wk, p_mean_wk),
                "score_baseline": eval_reg(y_wk, p_b_wk),
                "lgbm": eval_reg(y_wk, p_lgb_wk),
            }

    print("\n--- One-Row-Per-Learner Snapshots at Weeks 4, 8, 12 ---")
    for wk, s in snapshots.items():
        print(f"Week {wk:2d} (n={s['n_learners']:,}):")
        print(f"  Train Mean:        MAE={s['mean_baseline']['mae']:.2f}, RMSE={s['mean_baseline']['rmse']:.2f}, R2={s['mean_baseline']['r2']:.4f}")
        print(f"  weighted_score:    MAE={s['score_baseline']['mae']:.2f}, RMSE={s['score_baseline']['rmse']:.2f}, R2={s['score_baseline']['r2']:.4f}")
        print(f"  LightGBM:          MAE={s['lgbm']['mae']:.2f}, RMSE={s['lgbm']['rmse']:.2f}, R2={s['lgbm']['r2']:.4f}")

    # By-week trajectory
    week_traj = []
    for w in sorted(meta_va["week"].unique()):
        mask = meta_va["week"] == w
        if mask.sum() > 50:
            ev_w = eval_reg(y_va[mask], pred_lgb_va[mask])
            ev_b_w = eval_reg(y_va[mask], pred_b_va[mask])
            week_traj.append({
                "week": int(w),
                "n": int(mask.sum()),
                "lgb_mae": ev_w["mae"],
                "lgb_r2": ev_w["r2"],
                "score_baseline_mae": ev_b_w["mae"],
                "score_baseline_r2": ev_b_w["r2"],
            })

    return {
        "all_rows_summary": {"mean": all_mean, "weighted_score": all_b, "lgbm": all_lgb},
        "snapshots": snapshots,
        "weekly_trajectory": pd.DataFrame(week_traj),
        "model": lgb_reg,
    }


def run_grade_band_forecasting() -> Dict[str, Any]:
    """Train and evaluate grade_band (Fail, Pass, Distinction) 3-class classification."""
    print("\n" + "=" * 60)
    print("RUNNING STEP 7.2: GRADE_BAND CLASSIFICATION (FAIL / PASS / DISTINCTION)")
    print("=" * 60)

    # Ordinal mapping: Fail=0, Pass=1, Distinction=2 (critical for quadratic weighted kappa)
    band_map = {"Fail": 0, "Pass": 1, "Distinction": 2}
    rev_band_map = {0: "Fail", 1: "Pass", 2: "Distinction"}

    X_tr, y_tr_raw, meta_tr = load_split("train", non_withdrawn_only=True, target_col="grade_band")
    X_va, y_va_raw, meta_va = load_split("val", non_withdrawn_only=True, target_col="grade_band")

    y_tr = np.array([band_map[b] for b in y_tr_raw])
    y_va = np.array([band_map[b] for b in y_va_raw])

    w_tr = compute_learner_weights(meta_tr)
    w_va = compute_learner_weights(meta_va)

    # Baseline: Majority Class (Pass = 1)
    majority_class = int(pd.Series(y_tr).mode()[0])
    pred_majority_va = np.full(len(y_va), majority_class)

    # LightGBM Multiclass Classifier
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=5,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        objective="multiclass",
        num_class=3,
    )
    lgb_clf.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_X=X_va,
        eval_y=y_va,
        eval_sample_weight=[w_va],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_lgb_va = lgb_clf.predict(X_va)
    prob_lgb_va = lgb_clf.predict_proba(X_va)

    def eval_clf(y_true, y_pred, sample_weight=None) -> Dict[str, Any]:
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sample_weight))
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0, sample_weight=sample_weight)
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic", sample_weight=sample_weight))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], sample_weight=sample_weight)
        return {
            "macro_f1": macro_f1,
            "recall_fail": float(recalls[0]),
            "recall_pass": float(recalls[1]),
            "recall_distinction": float(recalls[2]),
            "quadratic_kappa": qwk,
            "confusion_matrix": cm,
        }

    all_maj = eval_clf(y_va, pred_majority_va, sample_weight=w_va)
    all_lgb = eval_clf(y_va, pred_lgb_va, sample_weight=w_va)

    print("\n--- Grade Band Classification Results (All Val Rows, Weighted by 1/rows-per-learner) ---")
    print(f"{'Model':<30} {'Macro-F1':>10} {'Rec(Fail)':>10} {'Rec(Pass)':>10} {'Rec(Dist)':>10} {'QWK':>8}")
    print(f"{'Majority (Pass)':<30} {all_maj['macro_f1']:>10.4f} {all_maj['recall_fail']:>10.4f} {all_maj['recall_pass']:>10.4f} {all_maj['recall_distinction']:>10.4f} {all_maj['quadratic_kappa']:>8.4f}")
    print(f"{'LightGBM Multiclass (Balanced)':<30} {all_lgb['macro_f1']:>10.4f} {all_lgb['recall_fail']:>10.4f} {all_lgb['recall_pass']:>10.4f} {all_lgb['recall_distinction']:>10.4f} {all_lgb['quadratic_kappa']:>8.4f}")

    print("\nConfusion Matrix (LightGBM, Rows=True, Cols=Pred: [0=Fail, 1=Pass, 2=Distinction]):")
    print(all_lgb["confusion_matrix"].astype(int))

    # Snapshots at weeks 4, 8, 12
    snapshots = {}
    for wk in [4, 8, 12]:
        mask_wk = meta_va["week"] == wk
        if mask_wk.sum() > 0:
            y_wk = y_va[mask_wk]
            p_maj_wk = pred_majority_va[mask_wk]
            p_lgb_wk = pred_lgb_va[mask_wk]
            snapshots[wk] = {
                "n_learners": int(mask_wk.sum()),
                "majority": eval_clf(y_wk, p_maj_wk),
                "lgbm": eval_clf(y_wk, p_lgb_wk),
            }

    print("\n--- One-Row-Per-Learner Snapshots at Weeks 4, 8, 12 ---")
    for wk, s in snapshots.items():
        print(f"Week {wk:2d} (n={s['n_learners']:,}):")
        print(f"  Majority: Macro-F1={s['majority']['macro_f1']:.4f}, QWK={s['majority']['quadratic_kappa']:.4f}")
        print(f"  LightGBM: Macro-F1={s['lgbm']['macro_f1']:.4f}, QWK={s['lgbm']['quadratic_kappa']:.4f}, "
              f"Recalls=[Fail={s['lgbm']['recall_fail']:.2f}, Pass={s['lgbm']['recall_pass']:.2f}, Dist={s['lgbm']['recall_distinction']:.2f}]")

    # By-week trajectory
    week_traj = []
    for w in sorted(meta_va["week"].unique()):
        mask = meta_va["week"] == w
        if mask.sum() > 50:
            ev_w = eval_clf(y_va[mask], pred_lgb_va[mask])
            week_traj.append({
                "week": int(w),
                "n": int(mask.sum()),
                "macro_f1": ev_w["macro_f1"],
                "quadratic_kappa": ev_w["quadratic_kappa"],
                "recall_fail": ev_w["recall_fail"],
                "recall_pass": ev_w["recall_pass"],
                "recall_distinction": ev_w["recall_distinction"],
            })

    return {
        "all_rows_summary": {"majority": all_maj, "lgbm": all_lgb},
        "snapshots": snapshots,
        "weekly_trajectory": pd.DataFrame(week_traj),
        "model": lgb_clf,
    }


def run_all_step7():
    score_res = run_score_forecasting()
    grade_res = run_grade_band_forecasting()
    import joblib
    out_path = Path("reports/phase2/forecast_results.joblib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"score": score_res, "grade": grade_res}, out_path)
    print(f"\nAll forecasting results saved to {out_path}")
    return score_res, grade_res


if __name__ == "__main__":
    run_all_step7()
