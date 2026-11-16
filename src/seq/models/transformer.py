"""
Transformer Architecture for student trajectory modeling (B5).
Features:
- Learned positional embeddings.
- TransformerEncoder with key_padding_mask.
- Masked temporal mean pooling.
- Single-task and multi-task heads, optional static branch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.seq.models.heads import (
    ClassificationHead,
    MaskedPooling,
    RegressionHead,
    StaticBranch,
)


class StudentTransformer(nn.Module):
    """
    Transformer Encoder for sequence modeling.
    """

    def __init__(
        self,
        in_features: int = 38,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_len: int = 16,
        pooling: str = "mean",
        use_static: bool = False,
        static_dim: int = 4,
        static_out_dim: int = 32,
        multi_task: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.use_static = use_static
        self.multi_task = multi_task

        # Input projection and positional embeddings
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooler = MaskedPooling(mode=pooling if pooling in {"last", "mean", "mean_last"} else "mean")
        pooled_dim = d_model * 2 if pooling == "mean_last" else d_model

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
        B, W, F = x.shape
        h = self.input_proj(x)

        # Add positional embedding
        positions = torch.arange(W, device=x.device).unsqueeze(0).expand(B, W)
        h = h + self.pos_emb(positions)
        h = self.drop(h)

        # PyTorch transformer expects src_key_padding_mask where True = ignored/padded
        padding_mask = ~mask  # True where padded

        out = self.transformer(h, src_key_padding_mask=padding_mask)
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
