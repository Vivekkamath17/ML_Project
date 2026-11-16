"""
Run script: Generate genuine SHAP Top-5 explanations for validation presentation 2014B.
Outputs saved to: reports/shap/<model>_k3_val_top5.parquet
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src.models.zoo.common import load_zoo_data
from src.models.data import load_split
from src.explainability.shap_generator import (
    compute_lgbm_shap,
    compute_xgb_shap,
    compute_catboost_shap,
    compute_rf_shap,
    compute_lr_shap,
)


def run_all_shap_generation():
    out_dir = REPO_ROOT / "reports" / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING SHAP FEATURE ATTRIBUTION GENERATION (VALIDATION 2014B)")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading validation split (2014B, k=3, min_week=2)...")
    t0 = time.time()
    X_tr_38, y_tr, meta_tr, X_va_38, y_va, meta_va, v3_features = load_zoo_data(k=3, min_week=2)
    print(f"Loaded {len(X_va_38):,} validation rows across {meta_va['id_student'].nunique():,} learners in {time.time()-t0:.2f}s.")

    # 2. LightGBM v3 (Champion Baseline)
    print("\n[2/6] Computing LightGBM v3 TreeSHAP...")
    t0 = time.time()
    lgbm_df = compute_lgbm_shap(X_va_38, meta_val=meta_va)
    lgbm_path = out_dir / "lgbm_v3_k3_val_top5.parquet"
    lgbm_df.to_parquet(lgbm_path, index=False, compression="zstd")
    print(f"-> Saved: {lgbm_path.name} ({len(lgbm_df):,} rows, {lgbm_path.stat().st_size / 1e6:.2f} MB) in {time.time()-t0:.2f}s")

    # 3. XGBoost
    print("\n[3/6] Computing XGBoost TreeSHAP...")
    t0 = time.time()
    xgb_df = compute_xgb_shap(X_va_38, meta_val=meta_va)
    xgb_path = out_dir / "xgb_k3_val_top5.parquet"
    xgb_df.to_parquet(xgb_path, index=False, compression="zstd")
    print(f"-> Saved: {xgb_path.name} ({len(xgb_df):,} rows, {xgb_path.stat().st_size / 1e6:.2f} MB) in {time.time()-t0:.2f}s")

    # 4. CatBoost
    print("\n[4/6] Computing CatBoost TreeSHAP...")
    t0 = time.time()
    cb_df = compute_catboost_shap(X_va_38, meta_val=meta_va)
    cb_path = out_dir / "catboost_k3_val_top5.parquet"
    cb_df.to_parquet(cb_path, index=False, compression="zstd")
    print(f"-> Saved: {cb_path.name} ({len(cb_df):,} rows, {cb_path.stat().st_size / 1e6:.2f} MB) in {time.time()-t0:.2f}s")

    # 5. Random Forest
    print("\n[5/6] Computing Random Forest TreeSHAP (Tree-Path Attribution)...")
    t0 = time.time()
    rf_df = compute_rf_shap(X_va_38, meta_val=meta_va)
    rf_path = out_dir / "rf_k3_val_top5.parquet"
    rf_df.to_parquet(rf_path, index=False, compression="zstd")
    print(f"-> Saved: {rf_path.name} ({len(rf_df):,} rows, {rf_path.stat().st_size / 1e6:.2f} MB) in {time.time()-t0:.2f}s")

    # 6. Logistic Regression
    print("\n[6/6] Computing Logistic Regression Linear SHAP...")
    t0 = time.time()
    X_tr_39, _, _ = load_split("train", k=3, min_week=2, drop_never_active=True)
    X_va_39, _, _ = load_split("val", k=3, min_week=2, drop_never_active=True)
    lr_df = compute_lr_shap(X_va_39, X_tr_39=X_tr_39, meta_val=meta_va)
    lr_path = out_dir / "lr_39feat_unweighted_k3_val_top5.parquet"
    lr_df.to_parquet(lr_path, index=False, compression="zstd")
    print(f"-> Saved: {lr_path.name} ({len(lr_df):,} rows, {lr_path.stat().st_size / 1e6:.2f} MB) in {time.time()-t0:.2f}s")

    print("\n" + "=" * 70)
    print("SHAP GENERATION COMPLETE! ALL 5 TARGET FILES GENERATED.")
    print("=" * 70)
    for p in sorted(out_dir.glob("*_k3_val_top5.parquet")):
        df_sample = pd.read_parquet(p)
        print(f"• {p.name:<38} | Rows: {len(df_sample):,} | Learners: {df_sample['id_student'].nunique():,} | Size: {p.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    run_all_shap_generation()
