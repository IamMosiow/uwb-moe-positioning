"""Spatial anchor attention mechanism for adaptive multi-anchor fusion."""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorAttention(nn.Module):
    """Multi-anchor attention mechanism with active anchor masking.

    Dynamically weights the contribution of each anchor node based on its estimated
    channel state and confidence, while strictly masking inactive/absent anchors.
    """

    def __init__(self, feature_dim: int = 128):
        """Initialize AnchorAttention.

        Args:
            feature_dim: Dimensionality of input anchor feature representations.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.attn = nn.Linear(feature_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute attention-weighted sum of anchor embeddings.

        Args:
            x: Anchor features tensor of shape [B, A, feature_dim].
            mask: Optional boolean tensor of shape [B, A] where True indicates invalid/masked anchor.
                  If None, auto-detected from all-zero feature vectors.
            return_weights: Whether to return attention weights [B, A, 1].

        Returns:
            fused: Fused spatial representation [B, feature_dim].
            (optional) weights: Attention weights [B, A, 1].
        """
        # Linear projection to compute unnormalized scalar attention score per anchor: [B, A]
        scores = self.attn(x).squeeze(-1)

        # Detect inactive anchors (all-zero features across dimension F)
        if mask is None:
            mask = x.abs().sum(dim=-1) == 0

        # Safe masking to prevent -inf / NaN if all anchors in a batch row are masked
        safe_mask = mask.clone()
        all_masked = safe_mask.all(dim=-1, keepdim=True)
        # If an entire sample has no active anchors, avoid completely masked row by keeping first
        safe_mask = safe_mask & (~all_masked)

        scores = scores.masked_fill(safe_mask, -1e4)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, A, 1]

        # Zero out weights on truly masked anchors
        weights = weights.masked_fill(mask.unsqueeze(-1), 0.0)

        fused = (x * weights).sum(dim=1)  # [B, feature_dim]

        if return_weights:
            return fused, weights
        return fused
