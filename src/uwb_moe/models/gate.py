"""Gating network for routing representations to specialized experts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftGate(nn.Module):
    """Soft routing network generating probability distributions over experts.

    Given a multi-modal feature vector, outputs normalized softmax routing weights
    across experts (e.g. Line-of-Sight vs. Non-Line-of-Sight conditions).
    """

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64, num_experts: int = 2):
        """Initialize SoftGate router.

        Args:
            feature_dim: Dimension of input feature representation.
            hidden_dim: Dimension of hidden projection layer.
            num_experts: Number of downstream expert branches (default 2: LOS and NLOS).
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute routing weights.

        Args:
            x: Input feature representations [N, feature_dim].

        Returns:
            Normalized gating weights [N, num_experts] summing to 1 across experts.
        """
        logits = self.gate(x)
        return F.softmax(logits, dim=-1)
