"""
CLI Entry Point for Phase 4 Sequence Experiments.
Usage:
  python -m src.seq.run_experiment --model gru --window 8 --target dropout --variant seq_only --trials 30 --seeds 5
  python -m src.seq.run_experiment --model gru --window 8 --target dropout --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch

from src.seq.builder import build_sequences
from src.seq.config import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    SEEDS,
    TUNING_DIR,
)
from src.seq.dataset import StudentSequenceDataset, create_dataloader
from src.seq.models import build_model
from src.seq.evaluate import evaluate_seq_model, log_to_mlflow
from src.seq.predict_io import write_finalscore_predictions, write_seq_predictions
from src.seq.train import fit_model, set_seed
from src.seq.tune import tune_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sequence Model Experiment")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm", "tcn", "transformer"])
    parser.add_argument("--window", type=int, default=8, choices=[4, 8])
    parser.add_argument("--target", type=str, default="dropout", choices=["dropout", "onset"])
    parser.add_argument(
        "--variant",
        type=str,
        default="seq_only",
        choices=["seq_only", "hybrid", "seq_only_multi", "hybrid_multi"],
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test (2 trials, 1 seed, 2 epochs)")
    return parser.parse_args()


def run_experiment(
    model_name: str = "gru",
    window: int = 8,
    target: str = "dropout",
    variant: str = "seq_only",
    n_trials: int = 30,
    n_seeds: int = 5,
    max_epochs: int = 12,
    smoke: bool = False,
) -> Dict[str, Any]:
    print("=" * 70, flush=True)
    print(f"RUNNING SEQUENCE EXPERIMENT: {model_name.upper()} (w={window}, target={target}, variant={variant})", flush=True)
    print(f"Smoke mode: {smoke} | Trials: {n_trials if not smoke else 2} | Seeds: {n_seeds if not smoke else 1}", flush=True)
    print("=" * 70, flush=True)

    # 1. Hyperparameter Tuning (30 trials)
    best_params, _ = tune_model(
        model_name=model_name,
        window=window,
        target=target,
        variant=variant,
        n_trials=n_trials if not smoke else 2,
        smoke=smoke,
    )

    best_params["use_static"] = "hybrid" in variant
    best_params["multi_task"] = "multi" in variant
    batch_size = max(int(best_params.get("batch_size", 256)), 256)

    # 2. Load Full Datasets
    print("Loading full train and val datasets for final refit...", flush=True)
    seq_tr = build_sequences("train", window=window, target=target)
    seq_va = build_sequences("val", window=window, target=target)

    N_tr = len(seq_tr["X"])
    N_va = len(seq_va["X"])

    if smoke:
        sub_tr = np.random.default_rng(42).choice(N_tr, size=max(int(N_tr * 0.05), 500), replace=False)
        sub_va = np.random.default_rng(42).choice(N_va, size=min(1000, N_va), replace=False)
        tr_ds = StudentSequenceDataset(seq_tr, indices=sub_tr)
        va_ds = StudentSequenceDataset(seq_va, indices=sub_va)
        eval_meta = seq_va["meta"].iloc[sub_va].copy()
    else:
        tr_ds = StudentSequenceDataset(seq_tr)
        va_ds = StudentSequenceDataset(seq_va)
        eval_meta = seq_va["meta"].copy()

    tr_loader = create_dataloader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = create_dataloader(va_ds, batch_size=batch_size, shuffle=False)

    # 3. Multi-Seed Training (5 fixed seeds)
    active_seeds = SEEDS[:1] if smoke else SEEDS[:n_seeds]
    seed_probs = []
    seed_regs = []
    seed_roc_list = []
    seed_pr_list = []
    seed_train_times = []
    seed_latencies = []
    param_count = 0

    fit_epochs = 2 if smoke else max_epochs

    for s_idx, seed in enumerate(active_seeds):
        print(f"\n--- Fitting Seed {seed} ({s_idx+1}/{len(active_seeds)}) ---", flush=True)
        set_seed(seed)

        model = build_model(
            model_name=model_name,
            in_features=38,
            **best_params,
        )
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        fit_res = fit_model(
            model=model,
            train_loader=tr_loader,
            val_loader=va_loader,
            epochs=fit_epochs,
            lr=best_params.get("lr", 1e-3),
            weight_decay=best_params.get("weight_decay", 1e-4),
            target=target,
            lambda_reg=best_params.get("lambda_reg", 0.3),
            patience=2 if smoke else 3,
            verbose=True,
        )

        y_prob = fit_res["val_y_prob"]
        seed_probs.append(y_prob)
        if fit_res["val_reg_preds"] is not None:
            seed_regs.append(fit_res["val_reg_preds"])

        seed_train_times.append(fit_res["train_time_sec"])
        seed_latencies.append(fit_res["latency_ms_per_1k"])

        ev = evaluate_seq_model(
            y_true=fit_res["val_y_true"],
            val_probs=y_prob,
            cluster_ids=eval_meta["id_student"].to_numpy(),
            target=target,
            n_resamples=50 if smoke else 500,
            seed=seed,
        )
        m = ev["metrics"]
        seed_roc_list.append(m["roc_auc"])
        seed_pr_list.append(m["pr_auc"])

        print(
            f"Seed {seed}: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
            f"Train={fit_res['train_time_sec']:.1f}s, Latency={fit_res['latency_ms_per_1k']:.2f}ms/1k",
            flush=True,
        )

        run_name = f"sequence/{model_name}/{variant}_{target}_w{window}/seed{seed}"
        log_to_mlflow(
            run_name=run_name,
            params={**best_params, "window": window, "seed": seed, "k": 3},
            metrics=m,
            tags={"phase": "4", "family": "sequence", "model": model_name, "variant": variant},
            experiment_name=target,
        )

    # 4. 5-Seed Headline Mean Predictions
    headline_probs = np.mean(seed_probs, axis=0)

    headline_eval = evaluate_seq_model(
        y_true=fit_res["val_y_true"],
        val_probs=headline_probs,
        cluster_ids=eval_meta["id_student"].to_numpy(),
        target=target,
        n_resamples=100 if smoke else 500,
        seed=42,
    )

    metrics = headline_eval["metrics"]
    paired = headline_eval["paired_vs_lgbm"]

    mean_pr = float(np.mean(seed_pr_list))
    std_pr = float(np.std(seed_pr_list))
    mean_roc = float(np.mean(seed_roc_list))
    std_roc = float(np.std(seed_roc_list))
    mean_train_time = float(np.mean(seed_train_times))
    mean_latency = float(np.mean(seed_latencies))

    print("\n" + "=" * 70, flush=True)
    print(f"HEADLINE RESULTS FOR {model_name.upper()} (w={window}, target={target}, {variant}):", flush=True)
    print(f"  5-Seed PR-AUC:  {mean_pr:.4f} ± {std_pr:.4f} | Headline PR-AUC: {metrics['pr_auc']:.4f}", flush=True)
    print(f"  5-Seed ROC-AUC: {mean_roc:.4f} ± {std_roc:.4f} | Headline ROC-AUC: {metrics['roc_auc']:.4f}", flush=True)
    print(f"  Brier Score:    {metrics['brier_score']:.4f}", flush=True)
    print(f"  Train Time:     {mean_train_time:.2f}s/seed | Latency: {mean_latency:.2f} ms/1k | Params: {param_count:,}", flush=True)

    if paired is not None:
        delta_pr = paired["delta_pr"]
        pr_ci = paired["delta_pr_ci"]
        delta_roc = paired["delta_roc"]
        roc_ci = paired["delta_roc_ci"]
        tied = pr_ci[0] <= 0.0 <= pr_ci[1]
        print(f"  Paired ΔPR vs LGBM v3:  {delta_pr:+.4f} [{pr_ci[0]:+.4f}, {pr_ci[1]:+.4f}] (Tied: {tied})", flush=True)
        print(f"  Paired ΔROC vs LGBM v3: {delta_roc:+.4f} [{roc_ci[0]:+.4f}, {roc_ci[1]:+.4f}]", flush=True)
    print("=" * 70, flush=True)

    # 5. Output Prediction Files (Schema-Validated)
    pred_file = write_seq_predictions(
        meta_df=eval_meta,
        y_true=fit_res["val_y_true"],
        y_score=headline_probs,
        model_name=f"{model_name}_{variant}",
        target=target,
        window=window,
    )

    if seed_regs:
        headline_regs = np.mean(seed_regs, axis=0)
        write_finalscore_predictions(
            meta_df=eval_meta,
            y_true=eval_meta["final_score"].to_numpy() if "final_score" in eval_meta else np.zeros(len(eval_meta)),
            y_score=headline_regs,
            model_name=f"{model_name}_{variant}",
            window=window,
        )

    # 6. Append Row to by_model_seq.csv
    by_model_csv = REPORTS_DIR / "by_model_seq.csv"
    row = {
        "model": model_name,
        "variant": variant,
        "window": window,
        "target": target,
        "seed_mean_pr": mean_pr,
        "seed_std_pr": std_pr,
        "seed_mean_roc": mean_roc,
        "seed_std_roc": std_roc,
        "val_pr": metrics["pr_auc"],
        "val_roc": metrics["roc_auc"],
        "brier": metrics["brier_score"],
        "params_count": param_count,
        "train_time_sec": mean_train_time,
        "latency_ms_per_1k": mean_latency,
        "delta_pr_vs_lgbm": paired["delta_pr"] if paired else None,
        "delta_pr_ci_low": paired["delta_pr_ci"][0] if paired else None,
        "delta_pr_ci_high": paired["delta_pr_ci"][1] if paired else None,
        "delta_roc_vs_lgbm": paired["delta_roc"] if paired else None,
        "delta_roc_ci_low": paired["delta_roc_ci"][0] if paired else None,
        "delta_roc_ci_high": paired["delta_roc_ci"][1] if paired else None,
        "tied_with_lgbm": (paired["delta_pr_ci"][0] <= 0.0 <= paired["delta_pr_ci"][1]) if paired else None,
    }

    df_row = pd.DataFrame([row])
    if by_model_csv.is_file():
        df_existing = pd.read_csv(by_model_csv)
        mask = (
            (df_existing["model"] == model_name)
            & (df_existing["variant"] == variant)
            & (df_existing["window"] == window)
            & (df_existing["target"] == target)
        )
        if mask.any():
            df_existing = df_existing[~mask]
        df_combined = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df_combined = df_row

    df_combined.to_csv(by_model_csv, index=False)
    print(f"Updated {by_model_csv} with results for {model_name}_{variant}_w{window}_{target}", flush=True)

    return row


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        model_name=args.model,
        window=args.window,
        target=args.target,
        variant=args.variant,
        n_trials=args.trials,
        n_seeds=args.seeds,
        max_epochs=args.max_epochs,
        smoke=args.smoke,
    )
