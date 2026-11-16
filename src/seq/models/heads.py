"""
Shared Heads and Mask-Aware Pooling modules for Sequence Models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPooling(nn.Module):
    """
    Mask-aware temporal pooling:
    - 'last': Extracts the last unmasked (real) step.
    - 'mean': Masked average over real steps.
    - 'mean_last': Concatenation of masked mean and last step.
    Handles all-masked rows safely without NaNs.
    """

    def __init__(self, mode: str = "last"):
        super().__init__()
        assert mode in {"last", "mean", "mean_last"}, f"Unknown pooling mode: {mode}"
        self.mode = mode

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h: (B, W, d)
        mask: (B, W) bool, True for real steps, False for padded
        """
        B, W, d = h.shape

        # Compute lengths of real steps per sequence (clamped to at least 1)
        lens = mask.sum(dim=1)  # (B,)
        safe_lens = torch.clamp(lens, min=1)

        # 1. Last real step: index is lens - 1 (or 0 if all masked)
        last_idx = torch.clamp(lens - 1, min=0)  # (B,)
        last_idx_expanded = last_idx.view(B, 1, 1).expand(B, 1, d)
        last_h = torch.gather(h, 1, last_idx_expanded).squeeze(1)  # (B, d)

        # Zero out completely masked rows if any
        all_masked = (lens == 0).unsqueeze(-1)
        last_h = torch.where(all_masked, torch.zeros_like(last_h), last_h)

        if self.mode == "last":
            return last_h

        # 2. Masked mean pooling
        mask_expanded = mask.unsqueeze(-1).float()  # (B, W, 1)
        sum_h = (h * mask_expanded).sum(dim=1)  # (B, d)
        mean_h = sum_h / safe_lens.unsqueeze(-1).float()
        mean_h = torch.where(all_masked, torch.zeros_like(mean_h), mean_h)

        if self.mode == "mean":
            return mean_h
        elif self.mode == "mean_last":
            return torch.cat([last_h, mean_h], dim=-1)

        return last_h


class StaticBranch(nn.Module):
    """
    Static/demographic feature processing branch (B4.1):
    Linear(F_static, 64) -> ReLU -> Dropout -> Linear(64, 32) -> ReLU
    """

    def __init__(self, in_features: int = 4, hidden_dim: int = 64, out_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        return self.net(static)


class ClassificationHead(nn.Module):
    """
    Standard classification head:
    Linear(in_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)  # (B,) logits


class RegressionHead(nn.Module):
    """
    Continuous target regression head (e.g. final_score):
    Linear(in_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)  # (B,) scalar predictions
