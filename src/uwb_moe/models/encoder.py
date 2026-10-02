"""Channel Impulse Response (CIR) feature extraction module."""

from typing import Sequence
import torch
import torch.nn as nn


class CIREncoder(nn.Module):
    """1D Convolutional Neural Network and multi-modal feature fusion encoder.

    Extracts temporal features from raw CIR magnitude envelopes using 1D convolutions,
    adapts pooling to a fixed dimension, and fuses them with time-difference telemetry
    (TD, TD_OFFSET) and learned anchor embeddings into a dense representation.
    """

    def __init__(
        self,
        cir_length: int = 152,
        embedding_dim: int = 16,
        channels: Sequence[int] = (16, 32),
        kernel_size: int = 5,
        pool_size: int = 16,
        output_dim: int = 128,
    ):
        """Initialize CIREncoder.

        Args:
            cir_length: Number of time samples in CIR envelope.
            embedding_dim: Dimensionality of learned anchor embedding.
            channels: Channel progression for 1D convolution layers.
            kernel_size: 1D convolution kernel size.
            pool_size: Target dimension for AdaptiveAvgPool1d.
            output_dim: Dimension of fused output embedding.
        """
        super().__init__()
        self.cir_length = cir_length
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        padding = kernel_size // 2
        layers = []
        in_c = 1
        for out_c in channels:
            layers.extend([
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            ])
            in_c = out_c

        layers.append(nn.AdaptiveAvgPool1d(pool_size))
        self.cnn = nn.Sequential(*layers)

        flattened_cnn_dim = channels[-1] * pool_size
        # Fuses flattened CIR features + 1 (TD) + 1 (TD_OFFSET) + embedding_dim (Anchor)
        fusion_dim = flattened_cnn_dim + 2 + embedding_dim
        self.fc = nn.Linear(fusion_dim, output_dim)

    def forward(
        self,
        cir: torch.Tensor,
        td: torch.Tensor,
        td_offset: torch.Tensor,
        anchor_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            cir: CIR magnitude tensor [B * A, L].
            td: Normalized Time Difference tensor [B * A, 1].
            td_offset: Normalized Time Difference Offset tensor [B * A, 1].
            anchor_emb: Learned anchor embedding tensor [B * A, embedding_dim].

        Returns:
            Fused feature representations [B * A, output_dim].
        """
        # cir: [N, L] -> [N, 1, L]
        x_cir = cir.unsqueeze(1)
        cnn_feat = self.cnn(x_cir).flatten(1)  # [N, channels[-1] * pool_size]

        # Multi-modal concatenation along feature dimension
        fused = torch.cat([cnn_feat, td, td_offset, anchor_emb], dim=1)
        return self.fc(fused)
