"""
Unit tests for Sequence Models forward pass, shape verification, and one-batch overfit.
Covers GRU, LSTM, TCN, and Transformer.
"""

import pytest
import torch
import torch.nn as nn

from src.seq.models import build_model


@pytest.mark.parametrize("model_name", ["gru", "lstm", "tcn", "transformer"])
def test_models_forward_shape_and_no_nan(model_name):
    """Verify forward pass output shape (B,) and absence of NaNs with partial masking."""
    B, W, F = 16, 8, 38
    model = build_model(model_name, in_features=F, hidden_size=32, d_model=32, channels=32, num_layers=1)
    model.eval()

    x = torch.randn(B, W, F)
    # Mask: first 4 steps padded for first half of batch
    mask = torch.ones(B, W, dtype=torch.bool)
    mask[:8, :4] = False

    with torch.no_grad():
        out = model(x, mask)

    assert out.shape == (B,), f"{model_name} output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), f"NaN in {model_name} output!"
    assert not torch.isinf(out).any(), f"Inf in {model_name} output!"


@pytest.mark.parametrize("model_name", ["gru", "lstm", "tcn"])
def test_models_hybrid_and_multitask(model_name):
    """Verify hybrid fusion with static branch and multi-task heads."""
    B, W, F = 8, 4, 38
    static_dim = 4
    model = build_model(
        model_name,
        in_features=F,
        hidden_size=32,
        d_model=32,
        channels=32,
        use_static=True,
        static_dim=static_dim,
        multi_task=True,
    )
    model.eval()

    x = torch.randn(B, W, F)
    mask = torch.ones(B, W, dtype=torch.bool)
    static = torch.randn(B, static_dim)

    with torch.no_grad():
        clf_out, reg_out = model(x, mask, static=static)

    assert clf_out.shape == (B,)
    assert reg_out.shape == (B,)
    assert not torch.isnan(clf_out).any()
    assert not torch.isnan(reg_out).any()


@pytest.mark.parametrize("model_name", ["gru", "lstm", "tcn"])
def test_models_one_batch_overfit(model_name):
    """Verify that a model can overfit a tiny batch of 8 samples (loss strictly drops)."""
    B, W, F = 8, 4, 38
    torch.manual_seed(42)
    model = build_model(model_name, in_features=F, hidden_size=32, d_model=32, channels=32, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn(B, W, F)
    mask = torch.ones(B, W, dtype=torch.bool)
    y = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.float32)

    model.train()
    initial_loss = None
    final_loss = None

    for epoch in range(25):
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.4, (
        f"{model_name} failed to overfit tiny batch: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
