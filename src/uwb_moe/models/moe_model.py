"""End-to-end Ultra-Wideband Mixture of Experts (UWB-MoE) localization model."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from uwb_moe.models.encoder import CIREncoder
from uwb_moe.models.gate import SoftGate
from uwb_moe.models.experts import Expert
from uwb_moe.models.attention import AnchorAttention
from uwb_moe.models.loss import DiversityLoss


class UWBMoEModel(nn.Module):
    """Deep Mixture-of-Experts architecture for robust UWB 2D indoor positioning.

    Combines:
    1. Spatial Anchor Embeddings.
    2. 1D CNN Temporal CIR Encoders with multi-modal TD/TD-Offset fusion.
    3. Soft Gating Router allocating Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) experts.
    4. Active-masked Multi-Anchor Attention pooling.
    5. Coordinate Regression Head predicting Cartesian coordinates (X, Y).
    """

    def __init__(
        self,
        cir_length: int = 152,
        n_anchors: int = 8,
        embedding_dim: int = 16,
        feature_dim: int = 128,
        gate_hidden_dim: int = 64,
        num_experts: int = 2,
        expert_hidden_dim: int = 128,
        position_head_hidden: int = 64,
        output_dim: int = 2,
        diversity_loss_coeff: float = 0.01,
    ):
        """Initialize UWBMoEModel.

        Args:
            cir_length: Length of CIR magnitude envelope.
            n_anchors: Maximum number of physical anchors deployed in the room.
            embedding_dim: Dimension of learned anchor identifier embeddings.
            feature_dim: Unified feature representation dimension.
            gate_hidden_dim: Hidden dimension for routing gate network.
            num_experts: Number of condition experts (default 2: LOS and NLOS).
            expert_hidden_dim: Hidden dimension inside each expert block.
            position_head_hidden: Hidden dimension of coordinate regression head.
            output_dim: Dimension of coordinate output (default 2 for X, Y).
            diversity_loss_coeff: Weight multiplier for load balancing loss.
        """
        super().__init__()
        self.cir_length = cir_length
        self.n_anchors = n_anchors
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.diversity_loss_coeff = diversity_loss_coeff

        # Submodules
        self.anchor_embedding = nn.Embedding(n_anchors, embedding_dim)
        self.encoder = CIREncoder(
            cir_length=cir_length,
            embedding_dim=embedding_dim,
            output_dim=feature_dim,
        )
        self.gate = SoftGate(
            feature_dim=feature_dim,
            hidden_dim=gate_hidden_dim,
            num_experts=num_experts,
        )
        self.expert_los = Expert(
            feature_dim=feature_dim,
            hidden_dim=expert_hidden_dim,
            output_dim=feature_dim,
        )
        self.expert_nlos = Expert(
            feature_dim=feature_dim,
            hidden_dim=expert_hidden_dim,
            output_dim=feature_dim,
        )
        self.attention = AnchorAttention(feature_dim=feature_dim)
        self.position_head = nn.Sequential(
            nn.Linear(feature_dim, position_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(position_head_hidden, output_dim),
        )

        self._diversity_loss_fn = DiversityLoss()
        self._gate_values: list[torch.Tensor] = []
        self._record_gate_values: bool = False

    def enable_gate_recording(self, enabled: bool = True) -> None:
        """Enable or disable accumulation of gating activations for post-hoc analysis."""
        self._record_gate_values = enabled
        if not enabled:
            self.reset_gate_values()

    def reset_gate_values(self) -> None:
        """Clear recorded gate values."""
        self._gate_values.clear()

    def get_gate_values(self) -> torch.Tensor:
        """Retrieve all recorded NLOS gate activations concatenated as a 1D tensor."""
        if not self._gate_values:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(self._gate_values, dim=0)

    def forward(
        self,
        cir: torch.Tensor,
        td: torch.Tensor,
        td_offset: torch.Tensor,
        anchor_ids: torch.Tensor,
        return_diversity_loss: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass through the UWB-MoE pipeline.

        Args:
            cir: CIR magnitude tensor [B, A, L].
            td: Normalized Time Difference tensor [B, A, 1].
            td_offset: Normalized TD Offset tensor [B, A, 1].
            anchor_ids: Anchor indices [B, A].
            return_diversity_loss: If True, returns (pos_pred, diversity_loss).
                                   If False, returns pos_pred.

        Returns:
            pos: Predicted positions [B, 2].
            (optional) diversity_loss: Scalar load-balancing regularization loss.
        """
        batch_size, num_anchors, cir_len = cir.shape
        flat_size = batch_size * num_anchors

        # Vectorized processing across all anchors and batches
        cir_flat = cir.view(flat_size, cir_len)
        td_flat = td.view(flat_size, 1)
        td_offset_flat = td_offset.view(flat_size, 1)
        anchor_ids_flat = anchor_ids.view(flat_size)

        # 1. Anchor Embedding
        anchor_emb = self.anchor_embedding(anchor_ids_flat)

        # 2. Multi-modal Feature Encoding
        features = self.encoder(cir_flat, td_flat, td_offset_flat, anchor_emb)

        # 3. MoE Gating & Routing
        gate_weights = self.gate(features)  # [B * A, 2]

        if self._record_gate_values or not self.training:
            # NLOS expert is index 1
            self._gate_values.append(gate_weights[:, 1].detach().cpu())

        # 4. Expert Processing
        los_out = self.expert_los(features)
        nlos_out = self.expert_nlos(features)

        # Soft mixture fusion per anchor
        mixed = gate_weights[:, 0:1] * los_out + gate_weights[:, 1:2] * nlos_out

        # 5. Spatial Multi-Anchor Attention
        mixed_spatial = mixed.view(batch_size, num_anchors, self.feature_dim)
        fused = self.attention(mixed_spatial)  # [B, feature_dim]

        # 6. Coordinate Regression Head
        pos = self.position_head(fused)  # [B, 2]

        if return_diversity_loss:
            div_loss = self._diversity_loss_fn(gate_weights)
            return pos, div_loss

        return pos


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate and format parameter counts across submodules.

    Args:
        model: PyTorch model instance.

    Returns:
        Dictionary mapping component names to parameter counts.
    """
    counts: Dict[str, int] = {}
    for name, module in model.named_children():
        counts[name] = sum(p.numel() for p in module.parameters())

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    counts["Total"] = total
    counts["Trainable"] = trainable
    return counts
