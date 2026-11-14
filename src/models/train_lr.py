"""
Step 3: Full 39-Feature Logistic Regression.
- Imputes NaNs with TRAIN medians.
- Scales ONLY the 39 original features with the saved train scaler (models/preprocessing/train_scaler.joblib).
- Appends missing-indicator columns unscaled for assessment features with missing values.
- Tunes C on Validation PR-AUC.
- Compares unweighted vs class_weight='balanced'.
- Logs to MLflow and reports top coefficients with domain sanity check.
"""
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.data import load_split
from src.models.metrics import evaluate_predictions
from src.models.tracking import log_model_experiment


def prepare_lr_matrices(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    scaler_path: str = "models/preprocessing/train_scaler.joblib",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    1. Impute NaNs with TRAIN medians.
    2. Add missing-indicator columns for columns that had NaNs in train.
    3. Scale ONLY the original 39 columns using the saved train scaler.
    4. Concatenate scaled original features + unscaled missing indicators.
    """
    scaler_bundle = joblib.load(scaler_path)
    scaler = scaler_bundle["scaler"]
    feature_names = scaler_bundle["feature_names"]

    # Align column ordering
    X_tr = X_train[feature_names].copy()
    X_va = X_val[feature_names].copy()

    # Find columns with missing values in train
    cols_with_na = [col for col in feature_names if X_tr[col].isna().any()]
    # print(f"Identified {len(cols_with_na)} columns with missing values in TRAIN: {cols_with_na}")

    # Create missing indicators (unscaled boolean 0/1)
    indicator_names = [f"missing_{c}" for c in cols_with_na]
    tr_indicators = X_tr[cols_with_na].isna().astype(float).values
    va_indicators = X_va[cols_with_na].isna().astype(float).values

    # Impute missing values with TRAIN medians
    train_medians = X_tr.median()
    X_tr_imputed = X_tr.fillna(train_medians).values
    X_va_imputed = X_va.fillna(train_medians).values

    # Scale ONLY the 39 original features
    X_tr_scaled = scaler.transform(X_tr_imputed)
    X_va_scaled = scaler.transform(X_va_imputed)

    # Append unscaled indicators
    X_tr_final = np.hstack([X_tr_scaled, tr_indicators])
    X_va_final = np.hstack([X_va_scaled, va_indicators])

    all_names = list(feature_names) + indicator_names
    return X_tr_final, X_va_final, all_names


def train_and_tune_logistic_regression(
    k: int = 3,
    min_week: int = 2,
    drop_never_active: bool = True,
    c_grid: Optional[List[float]] = None,
    n_resamples: int = 500,
) -> Dict[str, Any]:
    """Tune C for Logistic Regression and compare unweighted vs balanced."""
    print("=" * 60)
    print("RUNNING STEP 3: LOGISTIC REGRESSION ON ALL 39 FEATURES")
    print("=" * 60)

    if c_grid is None:
        c_grid = [0.001, 0.01, 0.1, 1.0, 10.0]

    X_train, y_train, meta_train = load_split("train", k=k, min_week=min_week, drop_never_active=drop_never_active)
    X_val, y_val, meta_val = load_split("val", k=k, min_week=min_week, drop_never_active=drop_never_active)
    learner_keys_val = meta_val["code_module"] + "_" + meta_val["code_presentation"] + "_" + meta_val["id_student"].astype(str)

    X_tr_mat, X_va_mat, feature_names = prepare_lr_matrices(X_train, X_val)

    best_unweighted = {"val_pr_auc": -1.0, "c": None, "model": None, "probs": None}
    best_balanced = {"val_pr_auc": -1.0, "c": None, "model": None, "probs": None}

    # Grid search C
    for c in c_grid:
        # 1. Unweighted
        clf_unw = LogisticRegression(C=c, max_iter=1000, random_state=42, solver="lbfgs")
        clf_unw.fit(X_tr_mat, y_train)
        probs_unw = clf_unw.predict_proba(X_va_mat)[:, 1]
        ev_unw = evaluate_predictions(y_val, probs_unw, learner_keys=learner_keys_val, n_resamples=0)
        pr_unw = ev_unw["point"]["pr_auc"]

        if pr_unw > best_unweighted["val_pr_auc"]:
            best_unweighted = {"val_pr_auc": pr_unw, "c": c, "model": clf_unw, "probs": probs_unw}

        # 2. Balanced
        clf_bal = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs")
        clf_bal.fit(X_tr_mat, y_train)
        probs_bal = clf_bal.predict_proba(X_va_mat)[:, 1]
        ev_bal = evaluate_predictions(y_val, probs_bal, learner_keys=learner_keys_val, n_resamples=0)
        pr_bal = ev_bal["point"]["pr_auc"]

        if pr_bal > best_balanced["val_pr_auc"]:
            best_balanced = {"val_pr_auc": pr_bal, "c": c, "model": clf_bal, "probs": probs_bal}

        print(f"C={c:<6}: Unweighted PR-AUC={pr_unw:.4f} | Balanced PR-AUC={pr_bal:.4f}")

    # Full bootstrap evaluation on champions
    print(f"\nEvaluating Best Unweighted LR (C={best_unweighted['c']}) with {n_resamples} bootstrap resamples...")
    eval_unw = evaluate_predictions(y_val, best_unweighted["probs"], learner_keys=learner_keys_val, n_resamples=n_resamples)

    print(f"Evaluating Best Balanced LR (C={best_balanced['c']}) with {n_resamples} bootstrap resamples...")
    eval_bal = evaluate_predictions(y_val, best_balanced["probs"], learner_keys=learner_keys_val, n_resamples=n_resamples)

    # Top coefficients domain check
    coefs = best_unweighted["model"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs}).sort_values("coefficient", ascending=False)

    print("\n--- Top 8 Positive Coefficients (Increase Dropout Risk) ---")
    for r in coef_df.head(8).itertuples():
        print(f"  {r.feature:<30}: {r.coefficient:+.4f}")

    print("\n--- Top 8 Negative Coefficients (Decrease Dropout Risk) ---")
    for r in coef_df.tail(8).itertuples():
        print(f"  {r.feature:<30}: {r.coefficient:+.4f}")

    # Log best models to MLflow
    log_model_experiment(
        experiment_name="dropout",
        run_name="lr_all_features_unweighted",
        params={"model": "logistic_regression", "C": best_unweighted["c"], "class_weight": "none", "k": k, "min_week": min_week},
        metrics={
            "roc_auc": eval_unw["point"]["roc_auc"],
            "pr_auc": eval_unw["point"]["pr_auc"],
            "pr_auc_mult": eval_unw["point"]["pr_auc_mult"],
            "brier": eval_unw["point"]["brier"],
            "mcc": eval_unw["point"]["mcc"],
        },
        feature_list=feature_names,
    )

    log_model_experiment(
        experiment_name="dropout",
        run_name="lr_all_features_balanced",
        params={"model": "logistic_regression", "C": best_balanced["c"], "class_weight": "balanced", "k": k, "min_week": min_week},
        metrics={
            "roc_auc": eval_bal["point"]["roc_auc"],
            "pr_auc": eval_bal["point"]["pr_auc"],
            "pr_auc_mult": eval_bal["point"]["pr_auc_mult"],
            "brier": eval_bal["point"]["brier"],
            "mcc": eval_bal["point"]["mcc"],
        },
        feature_list=feature_names,
    )

    return {
        "best_unweighted": {"c": best_unweighted["c"], "point": eval_unw["point"], "ci": eval_unw["ci"], "model": best_unweighted["model"]},
        "best_balanced": {"c": best_balanced["c"], "point": eval_bal["point"], "ci": eval_bal["ci"], "model": best_balanced["model"]},
        "coef_df": coef_df,
    }


if __name__ == "__main__":
    train_and_tune_logistic_regression(n_resamples=100)
