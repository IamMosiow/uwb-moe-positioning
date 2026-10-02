"""Specialized expert neural network blocks for UWB-MoE."""

import torch
import torch.nn as nn


class Expert(nn.Module):
    """Feedforward expert network specializing in channel condition regimes (LOS or NLOS)."""

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128, output_dim: int = 128):
        """Initialize Expert network.

        Args:
            feature_dim: Input feature dimension.
            hidden_dim: Hidden representation dimension.
            output_dim: Output representation dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process features through expert layers.

        Args:
            x: Input representations [N, feature_dim].

        Returns:
            Expert transformed representations [N, output_dim].
        """
        return self.net(x)
