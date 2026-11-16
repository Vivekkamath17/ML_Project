"""
Optuna Tuning module for Phase 4 Sequence Models.
Follows Vivek's protocol (reports/phase3/phase3_summary.md:66-80):
- 30 trials, TPE sampler with deterministic seed.
- Learner-grouped CV by id_student strictly on Train (zero student overlap).
- Objective: Mean Inner-CV PR-AUC across folds.
- MedianPruner for cutting unpromising trials early.
- Persists Optuna study (SQLite) and best params to reports/phase4/tuning/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import optuna
import torch

from src.models.harness import learner_group_cv
from src.seq.builder import build_sequences
from src.seq.config import (
    CV_FOLDS,
    OPTUNA_TRIALS,
    TUNING_DIR,
)
from src.seq.dataset import StudentSequenceDataset, create_dataloader
from src.seq.models import build_model
from src.seq.train import fit_model, set_seed


def suggest_hyperparameters(trial: optuna.Trial, model_name: str, variant: str) -> Dict[str, Any]:
    """Suggest hyperparameter configuration based on model class and variant."""
    name = model_name.lower()
    params: Dict[str, Any] = {}

    params["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [256, 512])
    params["dropout"] = trial.suggest_float("dropout", 0.05, 0.35)

    if name in {"gru", "lstm"}:
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64])
        params["d_model"] = params["hidden_size"]
        params["num_layers"] = trial.suggest_categorical("num_layers", [1, 2])
        params["pooling"] = trial.suggest_categorical("pooling", ["last", "mean_last"])
    elif name in {"tcn", "cnn"}:
        params["channels"] = trial.suggest_categorical("channels", [32, 64])
        params["kernel_size"] = trial.suggest_categorical("kernel_size", [2, 3])
        params["num_blocks"] = trial.suggest_categorical("num_blocks", [2, 3])
        params["pooling"] = trial.suggest_categorical("pooling", ["last", "mean", "mean_last"])
    elif name in {"transformer", "trans"}:
        params["d_model"] = trial.suggest_categorical("d_model", [32, 64])
        params["nhead"] = trial.suggest_categorical("nhead", [2, 4])
        params["num_layers"] = trial.suggest_categorical("num_layers", [1, 2])
        params["pooling"] = "mean"

    use_static = "hybrid" in variant
    multi_task = "multi" in variant

    params["use_static"] = use_static
    params["multi_task"] = multi_task

    if multi_task:
        params["lambda_reg"] = trial.suggest_categorical("lambda_reg", [0.1, 0.3, 1.0])

    return params


def tune_model(
    model_name: str,
    window: int = 8,
    target: str = "dropout",
    variant: str = "seq_only",
    n_trials: int = OPTUNA_TRIALS,
    cv_folds: int = 2,
    max_epochs: int = 8,
    subsample_fraction: float = 0.15,
    smoke: bool = False,
    seed: int = 42,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Run Optuna hyperparameter tuning using learner-grouped CV strictly on train split.
    """
    study_name = f"{model_name}_w{window}_{target}_{variant}"
    study_path = TUNING_DIR / f"{study_name}.db"
    storage_url = f"sqlite:///{study_path}"
    json_path = TUNING_DIR / f"{study_name}.json"

    # In non-smoke mode, check if we already have a fully completed study
    if not smoke and json_path.is_file() and study_path.is_file():
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed_trials) >= n_trials:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Loaded existing full tuning ({len(completed_trials)} trials) from {json_path}", flush=True)
                return data["best_params"], study
        except Exception:
            pass

    # If previous study had fewer trials (e.g. from smoke test), delete it to start clean 30 trials
    if not smoke and study_path.is_file():
        try:
            old_study = optuna.load_study(study_name=study_name, storage=storage_url)
            if len(old_study.trials) < n_trials:
                optuna.delete_study(study_name=study_name, storage=storage_url)
                print(f"Resetting incomplete study {study_name} to run full {n_trials} trials.", flush=True)
        except Exception:
            pass

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    print(f"Loading train sequences for tuning: window={window}, target={target}...", flush=True)
    seq_tr = build_sequences(split="train", window=window, target=target)
    N_tr = len(seq_tr["X"])

    meta_tr = seq_tr["meta"]
    student_ids = meta_tr["id_student"].to_numpy()

    rng = np.random.default_rng(seed)
    unique_students = np.unique(student_ids)

    if smoke:
        n_trials = 2
        max_epochs = 2
        sub_n = max(int(len(unique_students) * 0.05), 100)
    else:
        sub_n = int(len(unique_students) * subsample_fraction)

    sampled_students = set(rng.choice(unique_students, size=sub_n, replace=False))
    sub_mask = np.isin(student_ids, list(sampled_students))
    sub_indices = np.where(sub_mask)[0]
    sub_groups = student_ids[sub_indices]

    folds = list(learner_group_cv(np.zeros((len(sub_indices), 1)), np.zeros(len(sub_indices)), groups=sub_groups, n_splits=cv_folds))
    active_dataset_dict = {k: v[sub_indices] if isinstance(v, np.ndarray) else v for k, v in seq_tr.items()}

    print(f"Tuning setup: {len(sub_indices):,} rows across {sub_n} students ({cv_folds}-fold CV, {n_trials} trials)", flush=True)

    def objective(trial: optuna.Trial) -> float:
        hparams = suggest_hyperparameters(trial, model_name=model_name, variant=variant)
        fold_scores = []

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            set_seed(seed + fold_idx * 100)
            tr_ds = StudentSequenceDataset(active_dataset_dict, indices=tr_idx)
            va_ds = StudentSequenceDataset(active_dataset_dict, indices=va_idx)

            tr_loader = create_dataloader(tr_ds, batch_size=hparams["batch_size"], shuffle=True)
            va_loader = create_dataloader(va_ds, batch_size=hparams["batch_size"], shuffle=False)

            model = build_model(
                model_name=model_name,
                in_features=38,
                **hparams,
            )

            res = fit_model(
                model=model,
                train_loader=tr_loader,
                val_loader=va_loader,
                epochs=max_epochs,
                lr=hparams["lr"],
                weight_decay=hparams["weight_decay"],
                target=target,
                lambda_reg=hparams.get("lambda_reg", 0.3),
                patience=2 if not smoke else 1,
            )
            fold_scores.append(res["best_pr_auc"])

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    print(f"Optimizing {n_trials} trials for {study_name}...", flush=True)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    print(f"Best Trial: Inner-CV PR-AUC = {best_value:.4f}", flush=True)
    print(f"Best Hyperparameters: {best_params}", flush=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"best_inner_cv_pr": best_value, "best_params": best_params}, f, indent=2)

    return best_params, study
