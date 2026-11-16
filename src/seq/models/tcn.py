"""
Temporal Convolutional Network (TCN) Architecture for student trajectory modeling.
Features:
- Causal 1D convolutions (left padding only, zero future leakage).
- Dilated residual blocks (dilations 1, 2, 4).
- Mask-aware global temporal pooling.
- Single-task and multi-task heads, optional static branch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.seq.models.heads import (
    ClassificationHead,
    MaskedPooling,
    RegressionHead,
    StaticBranch,
)


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution with left padding.
    Guarantees output at step t depends only on inputs <= t.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, W)
        out = self.conv(x)
        # Remove trailing right padding to maintain causal shape (B, C, W)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        return out


class TCNBlock(nn.Module):
    """Residual causal dilated convolution block with weight norm and dropout."""

    def __init__(self, channels: int, kernel_size: int = 2, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        return F.relu(out + res)


class StudentTCN(nn.Module):
    """
    Temporal Convolutional Network for student sequence classification and forecasting.
    """

    def __init__(
        self,
        in_features: int = 38,
        channels: int = 64,
        kernel_size: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        pooling: str = "mean",
        use_static: bool = False,
        static_dim: int = 4,
        static_out_dim: int = 32,
        multi_task: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.channels = channels
        self.use_static = use_static
        self.multi_task = multi_task

        # Input projection: Linear(F, channels)
        self.input_proj = nn.Linear(in_features, channels)

        # Dilated residual blocks
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(TCNBlock(channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
        self.tcn_blocks = nn.Sequential(*blocks)

        # Mask-aware temporal pooling
        self.pooler = MaskedPooling(mode=pooling if pooling in {"last", "mean", "mean_last"} else "mean")
        pooled_dim = channels * 2 if pooling == "mean_last" else channels

        # Static branch
        if use_static:
            self.static_branch = StaticBranch(
                in_features=static_dim,
                hidden_dim=64,
                out_dim=static_out_dim,
                dropout=dropout,
            )
            total_repr_dim = pooled_dim + static_out_dim
            self.fusion_dropout = nn.Dropout(dropout)
        else:
            self.static_branch = None
            total_repr_dim = pooled_dim

        # Prediction heads
        self.clf_head = ClassificationHead(
            in_dim=total_repr_dim,
            hidden_dim=64,
            dropout=dropout,
        )

        if multi_task:
            self.reg_head = RegressionHead(
                in_dim=total_repr_dim,
                hidden_dim=64,
                dropout=dropout,
            )
        else:
            self.reg_head = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, W, F)
        h = self.input_proj(x)  # (B, W, channels)

        # Permute to (B, channels, W) for Conv1d
        h_conv = h.transpose(1, 2)
        out_conv = self.tcn_blocks(h_conv)
        # Permute back to (B, W, channels)
        out = out_conv.transpose(1, 2)

        # Pooling
        pooled = self.pooler(out, mask)

        # Fusion
        if self.use_static and self.static_branch is not None and static is not None:
            s_out = self.static_branch(static)
            repr_vec = torch.cat([pooled, s_out], dim=-1)
            repr_vec = self.fusion_dropout(repr_vec)
        else:
            repr_vec = pooled

        clf_logit = self.clf_head(repr_vec)

        if self.multi_task and self.reg_head is not None:
            reg_pred = self.reg_head(repr_vec)
            return clf_logit, reg_pred

        return clf_logit
