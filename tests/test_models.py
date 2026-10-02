"""Unit tests for neural network modules, attention, and loss functions."""

import pytest
import torch

from uwb_moe.models.encoder import CIREncoder
from uwb_moe.models.gate import SoftGate
from uwb_moe.models.experts import Expert
from uwb_moe.models.attention import AnchorAttention
from uwb_moe.models.loss import DiversityLoss, LocalizationLoss
from uwb_moe.models.moe_model import UWBMoEModel, count_parameters


def test_cir_encoder():
    """Verify CIREncoder transforms multi-modal inputs into dense embedding."""
    batch_size = 8
    cir_len = 64
    embedding_dim = 16
    output_dim = 128

    encoder = CIREncoder(cir_length=cir_len, embedding_dim=embedding_dim, output_dim=output_dim)

    cir = torch.randn(batch_size, cir_len)
    td = torch.randn(batch_size, 1)
    td_offset = torch.randn(batch_size, 1)
    anchor_emb = torch.randn(batch_size, embedding_dim)

    out = encoder(cir, td, td_offset, anchor_emb)
    assert out.shape == (batch_size, output_dim)


def test_soft_gate():
    """Verify SoftGate generates valid probability distribution summing to 1."""
    gate = SoftGate(feature_dim=128, hidden_dim=64, num_experts=2)
    x = torch.randn(10, 128)
    weights = gate(x)

    assert weights.shape == (10, 2)
    assert torch.all(weights >= 0.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_expert():
    """Verify Expert layer dimension preservation."""
    expert = Expert(feature_dim=128, hidden_dim=128, output_dim=128)
    x = torch.randn(5, 128)
    out = expert(x)
    assert out.shape == (5, 128)


def test_anchor_attention():
    """Verify AnchorAttention masks inactive anchors properly."""
    attn = AnchorAttention(feature_dim=64)
    batch_size = 4
    n_anchors = 3

    x = torch.randn(batch_size, n_anchors, 64)
    # Zero out the 2nd anchor in all batches to simulate missing anchor
    x[:, 1, :] = 0.0

    fused, weights = attn(x, return_weights=True)
    assert fused.shape == (batch_size, 64)
    assert weights.shape == (batch_size, n_anchors, 1)
    # Attention on the missing anchor should be zero
    assert torch.allclose(weights[:, 1, :], torch.zeros(batch_size, 1), atol=1e-4)


def test_diversity_loss():
    """Verify DiversityLoss behavior: 0 for uniform, positive for collapsed."""
    div_fn = DiversityLoss()

    # Case 1: Perfectly balanced gate (50% LOS, 50% NLOS)
    balanced_gates = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    loss_balanced = div_fn(balanced_gates)
    assert torch.isclose(loss_balanced, torch.tensor(0.0), atol=1e-5)

    # Case 2: Expert collapse (100% LOS, 0% NLOS)
    collapsed_gates = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    loss_collapsed = div_fn(collapsed_gates)
    assert loss_collapsed.item() > 0.5


def test_uwb_moe_forward_and_backward(sample_batch):
    """Verify complete UWBMoEModel forward pass and gradient backpropagation."""
    model = UWBMoEModel(
        cir_length=64,
        n_anchors=4,
        embedding_dim=8,
        feature_dim=32,
        gate_hidden_dim=16,
        num_experts=2,
        expert_hidden_dim=32,
        position_head_hidden=16,
        output_dim=2,
    )

    cir = sample_batch["cir"]
    td = sample_batch["td"]
    td_offset = sample_batch["td_offset"]
    anchor_id = sample_batch["anchor_id"]
    target = sample_batch["target"]

    pred, div_loss = model(cir, td, td_offset, anchor_id, return_diversity_loss=True)
    assert pred.shape == (cir.shape[0], 2)
    assert div_loss.dim() == 0

    loss = (pred - target).pow(2).mean() + 0.01 * div_loss
    loss.backward()

    # Check gradients flowed to CNN and gate
    assert model.encoder.cnn[0].weight.grad is not None
    assert model.gate.gate[0].weight.grad is not None


def test_count_parameters():
    """Verify parameter count helper returns positive numbers."""
    model = UWBMoEModel(cir_length=64, n_anchors=4)
    counts = count_parameters(model)
    assert "Total" in counts
    assert counts["Total"] > 0
    assert counts["Trainable"] == counts["Total"]
