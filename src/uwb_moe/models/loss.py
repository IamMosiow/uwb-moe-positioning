"""Loss functions for UWB localization and Mixture-of-Experts load balancing."""

from typing import Tuple
import torch
import torch.nn as nn


class DiversityLoss(nn.Module):
    """Load-balancing diversity loss to prevent expert collapse in Mixture-of-Experts.

    Penalizes deviations from uniform expert distribution across batches:
        L_div = K * sum(p_bar_i^2) - (sum(p_bar_i))^2
    where p_bar is the batch-averaged routing probability and K is the number of experts.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss from gating activations.

        Args:
            gate_weights: Tensor of shape [N, num_experts] containing softmax routing weights.

        Returns:
            Scalar diversity loss tensor.
        """
        # Average probability assigned to each expert across the batch
        mean_expert_prob = gate_weights.mean(dim=0)
        num_experts = float(gate_weights.shape[1])

        # Balanced: mean_expert_prob = 1/K for all i -> loss is 0.0
        # Collapsed: mean_expert_prob = 1.0 for one expert -> loss is K - 1
        diversity_loss = num_experts * torch.sum(mean_expert_prob ** 2) - torch.sum(mean_expert_prob).pow(2)
        return torch.clamp(diversity_loss, min=0.0)


class LocalizationLoss(nn.Module):
    """Composite loss combining 2D position regression with MoE diversity regularization."""

    def __init__(
        self,
        diversity_loss_coeff: float = 0.01,
        criterion: str = "mse",
    ):
        """Initialize LocalizationLoss.

        Args:
            diversity_loss_coeff: Weight multiplier for load balancing loss.
            criterion: 'mse' or 'smooth_l1'.
        """
        super().__init__()
        self.diversity_loss_coeff = diversity_loss_coeff

        if criterion == "smooth_l1":
            self.main_criterion = nn.SmoothL1Loss(reduction="mean")
        else:
            self.main_criterion = nn.MSELoss(reduction="mean")

        self.diversity_criterion = DiversityLoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gate_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Args:
            y_pred: Predicted 2D coordinates [B, 2].
            y_true: Ground truth 2D coordinates [B, 2].
            gate_weights: Gating probabilities [B * A, num_experts].

        Returns:
            Tuple of (total_loss, main_position_loss, diversity_loss).
        """
        main_loss = self.main_criterion(y_pred, y_true)
        diversity_loss = self.diversity_criterion(gate_weights)

        total_loss = main_loss + self.diversity_loss_coeff * diversity_loss
        return total_loss, main_loss, diversity_loss
