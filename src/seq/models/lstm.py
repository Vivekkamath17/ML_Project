"""
LSTM Architecture for student trajectory modeling.
Supports:
- Sequence-only and hybrid fusion with static branch.
- Single-task classification and multi-task (classification + final_score regression).
- Mask-aware pooling (last or mean_last).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.seq.models.heads import (
    ClassificationHead,
    MaskedPooling,
    RegressionHead,
    StaticBranch,
)


class StudentLSTM(nn.Module):
    """
    LSTM Sequence Model for student attrition and engagement modeling.
    """

    def __init__(
        self,
        in_features: int = 38,
        d_model: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        pooling: str = "last",
        use_static: bool = False,
        static_dim: int = 4,
        static_out_dim: int = 32,
        multi_task: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling_mode = pooling
        self.use_static = use_static
        self.multi_task = multi_task

        # 1. Per-step input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.Dropout(dropout),
        )

        # 2. Causal LSTM (bidirectional=False)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # 3. Masked temporal pooling
        self.pooler = MaskedPooling(mode=pooling)
        pooled_dim = hidden_size * 2 if pooling == "mean_last" else hidden_size

        # 4. Optional static branch
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

        # 5. Output heads
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
        h = self.input_proj(x)
        out, _ = self.lstm(h)
        pooled = self.pooler(out, mask)

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
