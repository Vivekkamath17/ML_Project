"""
PyTorch Tabular MLP wrapper, Optuna tuning, and multi-seed training for Phase 3 Model Zoo.
Implements strict determinism (seeds, DataLoader generators, deterministic algorithms),
imputed and standardized inputs with missing indicators, and 5-seed evaluation on full Train.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.harness import evaluate_model, learner_group_cv, log_run, paired_bootstrap
from src.models.predictions import write_predictions
from src.models.zoo.common import (
    compute_median_inner_iterations,
    get_model_size_mb,
    measure_inference_latency_ms_per_1k,
)


def set_seed(seed: int) -> torch.Generator:
    """Ensure strict determinism across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class TabularMLP(nn.Module):
    """Feedforward neural network for tabular data with BatchNorm and Dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_units

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_mlp_probs(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    all_probs = []
    n = len(X)
    for i in range(0, n, batch_size):
        X_b = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
        logits = model(X_b)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


def tune_mlp(
    X_tr_imp: np.ndarray,
    y_tr: np.ndarray,
    meta_tr: pd.DataFrame,
    n_trials: int = 30,
    seed: int = 42,
    moderate_pos_weight: float = 5.0,
    study_save_path: str = "models/zoo/mlp_optuna_study.joblib",
) -> Tuple[Dict[str, Any], int, float, optuna.Study]:
    """
    Tune Tabular MLP with Optuna over 30 trials using 5-fold inner GroupKFold.
    Early stopping on inner folds selects optimal epoch count.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_tr_imp.shape[1]
    groups = meta_tr["id_student"].to_numpy()
    folds = list(learner_group_cv(pd.DataFrame(X_tr_imp), y_tr, groups=groups, n_splits=5))

    trial_fold_epochs: Dict[int, List[int]] = {}

    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_units = trial.suggest_int("hidden_units", 32, 256, step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        use_cw = trial.suggest_categorical("use_class_weight", [False, True])

        pos_w = torch.tensor([moderate_pos_weight], device=device) if use_cw else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        fold_prs = []
        best_epochs = []

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            gen = set_seed(seed + fold_idx)
            model = TabularMLP(input_dim, hidden_units, n_layers, dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            ds_tr = TensorDataset(
                torch.tensor(X_tr_imp[tr_idx], dtype=torch.float32),
                torch.tensor(y_tr[tr_idx], dtype=torch.float32),
            )
            loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=gen)

            X_f_va, y_f_val = X_tr_imp[val_idx], y_tr[val_idx]

            best_val_pr = -1.0
            best_ep = 1
            patience = 8
            patience_counter = 0

            for ep in range(1, 51):
                train_epoch(model, loader_tr, optimizer, criterion, device)
                val_preds = predict_mlp_probs(model, X_f_va, device)
                val_pr = average_precision_score(y_f_val, val_preds)

                if val_pr > best_val_pr:
                    best_val_pr = val_pr
                    best_ep = ep
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            fold_prs.append(best_val_pr)
            best_epochs.append(best_ep)

        trial_fold_epochs[trial.number] = best_epochs
        return float(np.mean(fold_prs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_trial_num = study.best_trial.number
    best_fold_ep = trial_fold_epochs.get(best_trial_num, [15, 15, 15, 15, 15])
    chosen_epochs = compute_median_inner_iterations(best_fold_ep)

    Path(study_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, study_save_path)

    best_params = study.best_params.copy()
    best_params["epochs"] = chosen_epochs
    best_params["inner_cv_pr_auc"] = study.best_value
    best_params["device"] = str(device)

    return best_params, chosen_epochs, study.best_value, study


def fit_and_evaluate_mlp(
    X_tr_imp: np.ndarray,
    y_tr: np.ndarray,
    meta_tr: pd.DataFrame,
    X_va_imp: np.ndarray,
    y_va: np.ndarray,
    meta_va: pd.DataFrame,
    feature_names: List[str],
    best_params: Dict[str, Any],
    chosen_epochs: int,
    seeds: List[int],
    models_dir: str = "models/zoo",
    moderate_pos_weight: float = 5.0,
) -> Dict[str, Any]:
    """
    Fit Tabular MLP across 5 seeds on full Train using chosen parameters and fixed epoch count.
    Predict on Val (2014B), evaluate with evaluate_model, and run paired bootstrap vs LightGBM v3.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_tr_imp.shape[1]

    use_cw = best_params.get("use_class_weight", False)
    pos_w = torch.tensor([moderate_pos_weight], device=device) if use_cw else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    seed_val_probs: List[np.ndarray] = []
    seed_rocs: List[float] = []
    seed_prs: List[float] = []
    seed_briers: List[float] = []
    seed_train_times: List[float] = []
    model_files: List[Path] = []
    first_model = None

    ds_tr = TensorDataset(
        torch.tensor(X_tr_imp, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )

    for s in seeds:
        gen = set_seed(s)
        loader_tr = DataLoader(
            ds_tr,
            batch_size=best_params["batch_size"],
            shuffle=True,
            generator=gen,
        )

        model = TabularMLP(
            input_dim=input_dim,
            hidden_units=best_params["hidden_units"],
            n_layers=best_params["n_layers"],
            dropout=best_params["dropout"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"],
        )

        t0 = time.perf_counter()
        for ep in range(1, chosen_epochs + 1):
            train_epoch(model, loader_tr, optimizer, criterion, device)
        t_fit = time.perf_counter() - t0
        seed_train_times.append(t_fit)

        m_file = models_path / f"mlp_seed{s}.pt"
        torch.save(model.state_dict(), str(m_file))
        model_files.append(m_file)

        if first_model is None:
            first_model = model

        probs = predict_mlp_probs(model, X_va_imp, device)
        seed_val_probs.append(probs)

        roc = roc_auc_score(y_va, probs)
        pr = average_precision_score(y_va, probs)
        brier = float(np.mean((probs - y_va) ** 2))
        seed_rocs.append(roc)
        seed_prs.append(pr)
        seed_briers.append(brier)

        log_run(
            run_name=f"neural/mlp/default/seed{s}",
            params={
                **best_params,
                "epochs": chosen_epochs,
                "seed": s,
                "k": 3,
                "target": "label_dropout_k3",
                "device": str(device),
            },
            metrics={"roc_auc": roc, "pr_auc": pr, "brier_score": brier, "fit_time_sec": t_fit},
            feature_list=feature_names,
            experiment_name="dropout",
        )

    headline_probs = np.mean(seed_val_probs, axis=0)

    # Measure latency on 1000 rows
    sample_data = X_va_imp[:1000]
    latency_ms_per_1k = measure_inference_latency_ms_per_1k(
        lambda data: predict_mlp_probs(first_model, data, device), sample_data
    )
    model_size_mb = get_model_size_mb(model_files[0])

    # Save validation predictions
    df_preds = meta_va[["id_student", "code_module", "code_presentation", "week"]].copy()
    df_preds["model"] = "mlp"
    df_preds["k"] = 3
    df_preds["target"] = "label_dropout_k3"
    df_preds["split"] = "val"
    df_preds["y_true"] = y_va
    df_preds["y_prob"] = headline_probs

    pred_path = write_predictions(
        df_preds,
        metadata={
            "best_params": best_params,
            "epochs": chosen_epochs,
            "seed_mean_roc": float(np.mean(seed_rocs)),
            "seed_mean_pr": float(np.mean(seed_prs)),
            "seed_std_pr": float(np.std(seed_prs)),
            "seed_mean_brier": float(np.mean(seed_briers)),
            "device": str(device),
        },
    )

    # Paired bootstrap vs LightGBM v3
    df_lgbm = pd.read_parquet("predictions/dropout/k3/lgbm_v3_val.parquet")
    boot_res = paired_bootstrap(
        y_true=y_va,
        y_prob_a=headline_probs,
        y_prob_b=df_lgbm["y_prob"].to_numpy(),
        cluster_ids=meta_va["id_student"],
        n_boot=500,
        seed=42,
    )

    eval_metrics = evaluate_model(
        y_true=y_va,
        y_prob=headline_probs,
        cluster_ids=meta_va["id_student"],
        threshold=0.035,
        n_resamples=500,
        seed=42,
    )

    return {
        "model_name": "PyTorch Tabular MLP",
        "best_params": best_params,
        "chosen_epochs": chosen_epochs,
        "inner_cv_pr_auc": best_params.get("inner_cv_pr_auc", 0.0),
        "seed_rocs": seed_rocs,
        "seed_prs": seed_prs,
        "seed_briers": seed_briers,
        "seed_mean_roc": float(np.mean(seed_rocs)),
        "seed_std_roc": float(np.std(seed_rocs)),
        "seed_mean_pr": float(np.mean(seed_prs)),
        "seed_std_pr": float(np.std(seed_prs)),
        "seed_mean_brier": float(np.mean(seed_briers)),
        "train_time_sec": float(np.mean(seed_train_times)),
        "latency_ms_per_1k": latency_ms_per_1k,
        "model_size_mb": model_size_mb,
        "device": str(device),
        "eval_metrics": eval_metrics,
        "paired_vs_lgbm": boot_res,
        "pred_path": str(pred_path),
    }
