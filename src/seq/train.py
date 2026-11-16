"""
Training and evaluation loop for Sequence Models.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.seq.config import GRADIENT_CLIP, MAX_EPOCHS, PATIENCE


def set_seed(seed: int = 42):
    """Seed random, numpy, and torch across CPU and CUDA for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_clf: nn.Module,
    criterion_reg: Optional[nn.Module] = None,
    lambda_reg: float = 0.3,
    target: str = "dropout",
    device: torch.device = torch.device("cpu"),
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    use_cuda = device.type == "cuda" and scaler is not None

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        static = batch["static"].to(device) if model.use_static else None

        if target == "dropout":
            y = batch["y_dropout"].to(device)
            loss_mask = torch.ones_like(y, dtype=torch.bool)
        elif target == "onset":
            y = batch["y_onset"].to(device)
            loss_mask = batch["onset_mask"].to(device)
        else:
            y = batch["y_dropout"].to(device)
            loss_mask = torch.ones_like(y, dtype=torch.bool)

        if not loss_mask.any():
            continue

        optimizer.zero_grad()

        if use_cuda:
            with torch.amp.autocast(device_type="cuda"):
                if model.multi_task:
                    logits, reg_preds = model(x, mask, static=static)
                    clf_loss = criterion_clf(logits[loss_mask], y[loss_mask])

                    s_mask = batch["score_mask"].to(device)
                    if s_mask.any() and criterion_reg is not None:
                        y_score = batch["y_score"].to(device)
                        reg_loss = criterion_reg(reg_preds[s_mask], y_score[s_mask])
                        loss = clf_loss + lambda_reg * reg_loss
                    else:
                        loss = clf_loss
                else:
                    logits = model(x, mask, static=static)
                    loss = criterion_clf(logits[loss_mask], y[loss_mask])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            if model.multi_task:
                logits, reg_preds = model(x, mask, static=static)
                clf_loss = criterion_clf(logits[loss_mask], y[loss_mask])

                s_mask = batch["score_mask"].to(device)
                if s_mask.any() and criterion_reg is not None:
                    y_score = batch["y_score"].to(device)
                    reg_loss = criterion_reg(reg_preds[s_mask], y_score[s_mask])
                    loss = clf_loss + lambda_reg * reg_loss
                else:
                    loss = clf_loss
            else:
                logits = model(x, mask, static=static)
                loss = criterion_clf(logits[loss_mask], y[loss_mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    target: str = "dropout",
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float]:
    """Score validation set."""
    model.eval()
    all_probs = []
    all_trues = []
    all_regs = []

    t0 = time.perf_counter()
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        static = batch["static"].to(device) if model.use_static else None

        if target == "dropout":
            y = batch["y_dropout"]
        elif target == "onset":
            y = batch["y_onset"]
        else:
            y = batch["y_dropout"]

        if model.multi_task:
            logits, reg_preds = model(x, mask, static=static)
            all_regs.append(reg_preds.cpu().numpy())
        else:
            logits = model(x, mask, static=static)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_trues.append(y.numpy())

    total_time = time.perf_counter() - t0
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_trues)
    reg_out = np.concatenate(all_regs) if all_regs else None

    latency_ms_per_1k = (total_time / max(len(y_prob), 1)) * 1000.0 * 1000.0
    return y_true, y_prob, reg_out, latency_ms_per_1k


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    target: str = "dropout",
    lambda_reg: float = 0.3,
    patience: int = 4,
    pos_weight: Optional[float] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    trial_prune_cb: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, Any]:
    """Full training loop with early stopping."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pw_tensor = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    criterion_clf = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    criterion_reg = nn.HuberLoss() if model.multi_task else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_score = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion_clf=criterion_clf,
            criterion_reg=criterion_reg,
            lambda_reg=lambda_reg,
            target=target,
            device=device,
            scaler=scaler,
        )

        y_true, y_prob, _, _ = evaluate_predictions(model, val_loader, target=target, device=device)

        if len(np.unique(y_true)) > 1:
            val_pr_auc = float(average_precision_score(y_true, y_prob))
        else:
            val_pr_auc = 0.0

        scheduler.step(val_pr_auc)
        ep_time = time.perf_counter() - t_ep

        if verbose:
            print(f"  Epoch {epoch:2d}/{epochs}: Loss={loss:.4f}, Val PR-AUC={val_pr_auc:.4f} ({ep_time:.1f}s)", flush=True)

        if val_pr_auc > best_score:
            best_score = val_pr_auc
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with PR={best_score:.4f}", flush=True)
                break

        if trial_prune_cb is not None:
            trial_prune_cb(epoch, val_pr_auc)

    total_train_time = time.perf_counter() - t_start

    # Restore best weights
    model.load_state_dict(best_weights)
    y_true, y_prob, reg_preds, latency = evaluate_predictions(model, val_loader, target=target, device=device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "best_epoch": best_epoch,
        "best_pr_auc": best_score,
        "train_time_sec": total_train_time,
        "latency_ms_per_1k": latency,
        "param_count": param_count,
        "model_state": best_weights,
        "val_y_true": y_true,
        "val_y_prob": y_prob,
        "val_reg_preds": reg_preds,
    }
