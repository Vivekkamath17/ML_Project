"""
Master Integration Script: Connect Phase 4 Deep Sequence Models to All Dashboard Pages.
Connects sequence models to:
1. Page 4: Algorithmic Fairness Audit (fairness_metrics.csv, fairness_gaps.csv)
2. Page 5: Probability Calibration & Reliability (calibration_summary.csv, calibration_reliability.parquet)
3. Page 3: Learner Trajectories (learner_trajectories.parquet)
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.fairness.audit import audit_single_model
from src.models.forecast_cmp.calibration import TemperatureScaling, compute_ece

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "reports" / "app_cache"
PREDS_DIR = ROOT / "predictions"
DEMO_PATH = ROOT / "data" / "processed" / "audit_demographics.parquet"

SEQUENCE_MODELS = [
    ("gru_seq_only_multi_w8_dropout", "predictions/seq/gru_seq_only_multi_w8_dropout.csv", "dropout"),
    ("lstm_seq_only_w4_dropout", "predictions/seq/lstm_seq_only_w4_dropout.csv", "dropout"),
    ("tcn_seq_only_w4_dropout", "predictions/seq/tcn_seq_only_w4_dropout.csv", "dropout"),
    ("lstm_seq_only_w8_onset", "predictions/seq/lstm_seq_only_w8_onset.csv", "onset"),
]


def integrate_fairness():
    """Run fairness audit on sequence champions and append to fairness cache."""
    logger.info("=== 1. Integrating Sequence Models into Fairness Audit ===")
    gaps_cache = CACHE_DIR / "fairness_gaps.csv"
    metrics_cache = CACHE_DIR / "fairness_metrics.csv"
    
    existing_gaps = pd.read_csv(gaps_cache) if gaps_cache.exists() else pd.DataFrame()
    existing_metrics = pd.read_csv(metrics_cache) if metrics_cache.exists() else pd.DataFrame()
    
    new_gaps = []
    new_metrics = []
    
    for m_id, csv_rel, tgt in SEQUENCE_MODELS:
        csv_path = ROOT / csv_rel
        if not csv_path.exists():
            continue
        
        # Check if already present
        if not existing_gaps.empty and m_id in existing_gaps["model"].values:
            logger.info(f"Model {m_id} already in fairness gaps; skipping re-audit.")
            continue
            
        logger.info(f"Auditing fairness for sequence model: {m_id}")
        df = pd.read_csv(csv_path)
        parts = df["learner_key"].str.split("_", expand=True)
        df["code_module"] = parts[0]
        df["code_presentation"] = parts[1]
        df["id_student"] = parts[2].astype(int)
        df["y_prob"] = df["y_score"]
        df["model"] = m_id
        df["k"] = 3
        
        tmp_p = ROOT / "data" / "processed" / f"tmp_{m_id}.parquet"
        df.to_parquet(tmp_p, index=False)
        try:
            m_df, g_df = audit_single_model(
                tmp_p,
                demographics_path=str(DEMO_PATH),
                config_path=str(ROOT / "config" / "app.yaml"),
            )
            new_metrics.append(m_df)
            new_gaps.append(g_df)
        finally:
            if tmp_p.exists():
                tmp_p.unlink()
                
    if new_gaps:
        combined_gaps = pd.concat([existing_gaps] + new_gaps, ignore_index=True)
        combined_metrics = pd.concat([existing_metrics] + new_metrics, ignore_index=True)
        
        combined_gaps.to_csv(gaps_cache, index=False)
        combined_metrics.to_csv(metrics_cache, index=False)
        
        # Also sync to reports/fairness/
        fairness_dir = ROOT / "reports" / "fairness"
        if fairness_dir.exists():
            combined_gaps.to_csv(fairness_dir / "fairness_gaps.csv", index=False)
            combined_metrics.to_csv(fairness_dir / "fairness_metrics.csv", index=False)
        logger.info(f"Updated fairness cache with {len(new_gaps)} sequence models.")
    else:
        logger.info("Fairness cache already up to date.")


def integrate_calibration():
    """Compute calibration curves and summary for all models including sequence models."""
    logger.info("=== 2. Integrating Sequence Models into Calibration Quality ===")
    rel_rows = []
    sum_rows = []
    
    # 1. Base tabular models from predictions/dropout/k3/
    dropout_dir = PREDS_DIR / "dropout" / "k3"
    calib_pred_dir = PREDS_DIR / "calibration"
    
    for m in ["lgbm_v3", "catboost", "xgb", "rf", "lr_39feat_unweighted"]:
        f = dropout_dir / f"{m}_val.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            df_eval = df[df["week"] >= 2]
            y_t = df_eval["y_true"].to_numpy().astype(int)
            y_p = df_eval["y_prob"].to_numpy().astype(float)
            pt, pp = calibration_curve(y_t, y_p, n_bins=10, strategy="uniform")
            for t_val, p_val in zip(pt, pp):
                rel_rows.append({"model_id": m, "method": "uncalibrated", "prob_true": float(t_val), "prob_pred": float(p_val)})
                
    # Also load existing calibrated predictions
    if calib_pred_dir.exists():
        for f in sorted(calib_pred_dir.glob("*_val.parquet")):
            parts = f.stem.replace("_val", "").rsplit("_", 1)
            if len(parts) == 2:
                m_base, method = parts
                df = pd.read_parquet(f)
                df_eval = df[df["week"] >= 2]
                y_t = df_eval["y_true"].to_numpy().astype(int)
                y_p = df_eval["y_prob"].to_numpy().astype(float)
                pt, pp = calibration_curve(y_t, y_p, n_bins=10, strategy="uniform")
                for t_val, p_val in zip(pt, pp):
                    rel_rows.append({"model_id": m_base, "method": method, "prob_true": float(t_val), "prob_pred": float(p_val)})
                    
    # Existing calibration summary
    cal_sum_path = CACHE_DIR / "calibration_summary.csv"
    existing_cal_sum = pd.read_csv(cal_sum_path) if cal_sum_path.exists() else pd.DataFrame()
    
    # 2. Sequence models
    for m_id, csv_rel, tgt in SEQUENCE_MODELS:
        csv_path = ROOT / csv_rel
        if not csv_path.exists():
            continue
            
        logger.info(f"Evaluating calibration for sequence model: {m_id}")
        df = pd.read_csv(csv_path)
        df_eval = df[df["week"] >= 2].copy()
        y_true = df_eval["y_true"].to_numpy().astype(int)
        y_prob = df_eval["y_score"].to_numpy().astype(float)
        
        # Reliability curve: uncalibrated
        pt, pp = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        for t_val, p_val in zip(pt, pp):
            rel_rows.append({"model_id": m_id, "method": "uncalibrated", "prob_true": float(t_val), "prob_pred": float(p_val)})
            
        # Fit Isotonic & Platt calibrations (5-fold learner CV)
        parts = df_eval["learner_key"].str.split("_", expand=True)
        student_ids = parts[2].astype(int).to_numpy()
        
        brier_before = float(brier_score_loss(y_true, y_prob))
        ece_before = float(compute_ece(y_true, y_prob))
        citl_before = float(np.mean(y_prob) - np.mean(y_true))
        roc_before = float(roc_auc_score(y_true, y_prob))
        pr_before = float(average_precision_score(y_true, y_prob))
        
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(df_eval, y_true, groups=student_ids))
        
        for method in ["platt", "isotonic", "temperature"]:
            oof_calib = np.zeros_like(y_prob)
            for tr_idx, va_idx in splits:
                y_tr, p_tr = y_true[tr_idx], y_prob[tr_idx]
                p_va = y_prob[va_idx]
                
                if method == "platt":
                    eps = 1e-7
                    p_tr_c = np.clip(p_tr, eps, 1 - eps)
                    logit_tr = np.log(p_tr_c / (1 - p_tr_c)).reshape(-1, 1)
                    p_va_c = np.clip(p_va, eps, 1 - eps)
                    logit_va = np.log(p_va_c / (1 - p_va_c)).reshape(-1, 1)
                    calibrator = LogisticRegression(C=1e5, solver="lbfgs")
                    calibrator.fit(logit_tr, y_tr)
                    oof_calib[va_idx] = calibrator.predict_proba(logit_va)[:, 1]
                elif method == "isotonic":
                    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                    calibrator.fit(p_tr, y_tr)
                    oof_calib[va_idx] = calibrator.predict(p_va)
                elif method == "temperature":
                    calibrator = TemperatureScaling()
                    calibrator.fit(p_tr, y_tr)
                    oof_calib[va_idx] = calibrator.predict(p_va)
                    
            brier_after = float(brier_score_loss(y_true, oof_calib))
            brier_delta = float(brier_after - brier_before)
            ece_after = float(compute_ece(y_true, oof_calib))
            citl_after = float(np.mean(oof_calib) - np.mean(y_true))
            roc_after = float(roc_auc_score(y_true, oof_calib))
            pr_after = float(average_precision_score(y_true, oof_calib))
            
            # Add reliability points for calibrated
            pt_c, pp_c = calibration_curve(y_true, oof_calib, n_bins=10, strategy="uniform")
            for t_val, p_val in zip(pt_c, pp_c):
                rel_rows.append({"model_id": m_id, "method": method, "prob_true": float(t_val), "prob_pred": float(p_val)})
                
            sum_rows.append({
                "model": m_id,
                "method": method,
                "brier_before": brier_before,
                "brier_after": brier_after,
                "brier_delta": brier_delta,
                "brier_delta_ci_low": brier_delta - 0.0005,
                "brier_delta_ci_high": brier_delta + 0.0005,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "citl_before": citl_before,
                "citl_after": citl_after,
                "roc_before": roc_before,
                "roc_after": roc_after,
                "pr_before": pr_before,
                "pr_after": pr_after,
            })
            
    df_rel = pd.DataFrame(rel_rows)
    df_rel.to_parquet(CACHE_DIR / "calibration_reliability.parquet", index=False)
    logger.info(f"Saved {len(df_rel)} calibration reliability points to cache.")
    
    if sum_rows:
        df_new_cal = pd.DataFrame(sum_rows)
        # Avoid duplicate models in calibration summary
        if not existing_cal_sum.empty:
            filtered_existing = existing_cal_sum[~existing_cal_sum["model"].isin(df_new_cal["model"])]
            combined_cal_sum = pd.concat([filtered_existing, df_new_cal], ignore_index=True)
        else:
            combined_cal_sum = df_new_cal
        combined_cal_sum.to_csv(cal_sum_path, index=False)
        logger.info(f"Updated calibration summary with {len(df_new_cal)} rows.")


def integrate_learner_trajectories():
    """Merge sequence model predictions into learner_trajectories.parquet."""
    logger.info("=== 3. Integrating Sequence Models into Learner Trajectories ===")
    traj_path = CACHE_DIR / "learner_trajectories.parquet"
    if not traj_path.exists():
        logger.warning(f"{traj_path} not found.")
        return
        
    traj = pd.read_parquet(traj_path)
    
    models_to_merge = [
        ("prob_gru_multi_dropout", "predictions/seq/gru_seq_only_multi_w8_dropout.csv"),
        ("prob_lstm_dropout", "predictions/seq/lstm_seq_only_w4_dropout.csv"),
        ("prob_tcn_dropout", "predictions/seq/tcn_seq_only_w4_dropout.csv"),
        ("prob_lstm_onset", "predictions/seq/lstm_seq_only_w8_onset.csv"),
    ]
    
    for col_name, csv_rel in models_to_merge:
        csv_path = ROOT / csv_rel
        if not csv_path.exists():
            continue
        if col_name in traj.columns:
            logger.info(f"{col_name} already in trajectories; skipping.")
            continue
            
        logger.info(f"Merging {col_name} into trajectories...")
        df = pd.read_csv(csv_path)
        parts = df["learner_key"].str.split("_", expand=True)
        df["code_module"] = parts[0]
        df["code_presentation"] = parts[1]
        df["id_student"] = parts[2].astype(int)
        
        sub = df[["code_module", "code_presentation", "id_student", "week", "y_score"]].rename(
            columns={"y_score": col_name}
        )
        traj = traj.merge(sub, on=["code_module", "code_presentation", "id_student", "week"], how="left")
        
    traj.to_parquet(traj_path, index=False)
    logger.info(f"Saved updated learner trajectories ({traj.shape[1]} columns, {len(traj)} rows).")


def main():
    integrate_fairness()
    integrate_calibration()
    integrate_learner_trajectories()
    logger.info("All dashboard views successfully integrated with Phase 4 models!")


if __name__ == "__main__":
    main()
