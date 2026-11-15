"""Run calibration pipeline across all Phase 2 dropout validation predictions."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.models.forecast_cmp.calibration import (
    TemperatureScaling,
    compute_ece,
    learner_bootstrap_brier_delta,
)
from src.models.predictions import read_predictions, write_predictions


def run_all_calibrations() -> pd.DataFrame:
    pred_dir = Path("predictions/dropout/k3")
    calib_pred_dir = Path("predictions/calibration")
    calib_model_dir = Path("models/calibration")
    reports_dir = Path("reports/niranjan")
    fig_dir = reports_dir / "figures"

    calib_pred_dir.mkdir(parents=True, exist_ok=True)
    calib_model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "lgbm_v3": pred_dir / "lgbm_v3_val.parquet",
        "xgb": pred_dir / "xgb_val.parquet",
        "catboost": pred_dir / "catboost_val.parquet",
        "rf": pred_dir / "rf_val.parquet",
        "lr_39feat_unweighted": pred_dir / "lr_39feat_unweighted_val.parquet",
    }

    results = []

    for model_name, path in model_files.items():
        if not path.exists():
            print(f"Skipping {model_name}, file not found: {path}")
            continue

        df_val, meta = read_predictions(path)
        y_true = df_val["y_true"].to_numpy().astype(int)
        y_prob_raw = df_val["y_prob"].to_numpy().astype(float)
        student_ids = df_val["id_student"].to_numpy()

        brier_before = float(brier_score_loss(y_true, y_prob_raw))
        ece_before = compute_ece(y_true, y_prob_raw)
        c_large_before = float(np.mean(y_prob_raw) - np.mean(y_true))
        roc_before = float(roc_auc_score(y_true, y_prob_raw))
        pr_before = float(average_precision_score(y_true, y_prob_raw))

        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(df_val, y_true, groups=student_ids))

        methods = ["platt", "isotonic", "temperature"]

        for method in methods:
            oof_calib = np.zeros_like(y_prob_raw)

            # 5-fold CV strictly inside val
            for train_idx, val_idx in splits:
                y_tr, p_tr = y_true[train_idx], y_prob_raw[train_idx]
                p_va = y_prob_raw[val_idx]

                if method == "platt":
                    eps = 1e-7
                    p_tr_clip = np.clip(p_tr, eps, 1.0 - eps)
                    logits_tr = np.log(p_tr_clip / (1.0 - p_tr_clip)).reshape(-1, 1)
                    p_va_clip = np.clip(p_va, eps, 1.0 - eps)
                    logits_va = np.log(p_va_clip / (1.0 - p_va_clip)).reshape(-1, 1)

                    cal = LogisticRegression(solver="lbfgs")
                    cal.fit(logits_tr, y_tr)
                    oof_calib[val_idx] = cal.predict_proba(logits_va)[:, 1]

                elif method == "isotonic":
                    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                    cal.fit(p_tr, y_tr)
                    oof_calib[val_idx] = cal.predict(p_va)

                elif method == "temperature":
                    cal = TemperatureScaling()
                    cal.fit(p_tr, y_tr)
                    oof_calib[val_idx] = cal.predict(p_va)

            # Fit all-val calibrator for the later honest test run
            if method == "platt":
                eps = 1e-7
                p_clip = np.clip(y_prob_raw, eps, 1.0 - eps)
                logits = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
                full_cal = LogisticRegression(solver="lbfgs").fit(logits, y_true)
            elif method == "isotonic":
                full_cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(y_prob_raw, y_true)
            elif method == "temperature":
                full_cal = TemperatureScaling().fit(y_prob_raw, y_true)

            joblib.dump(full_cal, calib_model_dir / f"{model_name}_{method}_calibrator.joblib")

            # Metrics
            brier_after = float(brier_score_loss(y_true, oof_calib))
            ece_after = compute_ece(y_true, oof_calib)
            c_large_after = float(np.mean(oof_calib) - np.mean(y_true))
            roc_after = float(roc_auc_score(y_true, oof_calib))
            pr_after = float(average_precision_score(y_true, oof_calib))

            delta_brier, ci_l, ci_h = learner_bootstrap_brier_delta(
                y_true, y_prob_raw, oof_calib, cluster_ids=student_ids, n_resamples=500, seed=42
            )

            # Save OOF calibrated predictions
            df_out = df_val.copy()
            df_out["y_prob"] = oof_calib
            df_out["model"] = f"{model_name}_{method}"
            write_predictions(
                df_out,
                path=calib_pred_dir / f"{model_name}_{method}_val.parquet",
                metadata={"calibrator": method, "brier_oof": brier_after, "ece_oof": ece_after},
            )

            results.append({
                "model": model_name,
                "method": method,
                "brier_before": brier_before,
                "brier_after": brier_after,
                "brier_delta": delta_brier,
                "brier_delta_ci_low": ci_l,
                "brier_delta_ci_high": ci_h,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "citl_before": c_large_before,
                "citl_after": c_large_after,
                "roc_before": roc_before,
                "roc_after": roc_after,
                "pr_before": pr_before,
                "pr_after": pr_after,
            })

            # Plot reliability diagram
            fig, ax = plt.subplots(figsize=(6, 5))
            for p_arr, label, col in [(y_prob_raw, "Before", "grey"), (oof_calib, f"After ({method})", "blue")]:
                quantiles = np.linspace(0, 1, 11)
                edges = np.quantile(p_arr, quantiles)
                edges[-1] += 1e-7
                x_pts, y_pts = [], []
                for i in range(10):
                    m = (p_arr >= edges[i]) & (p_arr < edges[i + 1]) if i < 9 else (p_arr >= edges[i]) & (p_arr <= edges[i + 1])
                    if np.sum(m) > 0:
                        x_pts.append(np.mean(p_arr[m]))
                        y_pts.append(np.mean(y_true[m]))
                ax.plot(x_pts, y_pts, marker="o", label=label, color=col)
            ax.plot([0, 0.4], [0, 0.4], "k--", label="Perfect Calibration")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Observed Fraction Positive")
            ax.set_title(f"Reliability Diagram: {model_name} ({method})")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(fig_dir / f"reliability_{model_name}_{method}.png", dpi=150)
            plt.close()

    df_res = pd.DataFrame(results)
    df_res.to_csv(reports_dir / "calibration_summary.csv", index=False)
    print("Calibration pipeline completed successfully.")
    return df_res


if __name__ == "__main__":
    run_all_calibrations()
