"""PyTorch Dataset and DataLoader builders for multi-anchor UWB telemetry."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class UWBDataset(Dataset):
    """PyTorch Dataset wrapping aligned multi-anchor UWB channel features and coordinates."""

    def __init__(
        self,
        features: Dict[str, Union[np.ndarray, torch.Tensor]],
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """Initialize UWBDataset.

        Args:
            features: Dictionary containing:
                - 'cir': [N, A, L] float32 tensor
                - 'td': [N, A, 1] float32 tensor
                - 'td_offset': [N, A, 1] float32 tensor
                - 'anchor_id': [N, A] int64 tensor
            targets: Optional [N, 2] target coordinates (x, y) in meters.
        """
        self.cir = self._to_tensor(features["cir"], dtype=torch.float32)
        self.td = self._to_tensor(features["td"], dtype=torch.float32)
        self.td_offset = self._to_tensor(features["td_offset"], dtype=torch.float32)
        self.anchor_id = self._to_tensor(features["anchor_id"], dtype=torch.long)

        if targets is not None:
            self.targets = self._to_tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        self._num_samples = len(self.cir)

    @staticmethod
    def _to_tensor(arr: Union[np.ndarray, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=dtype)
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=dtype)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "cir": self.cir[idx],
            "td": self.td[idx],
            "td_offset": self.td_offset[idx],
            "anchor_id": self.anchor_id[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def create_dataloaders(
    features: Dict[str, np.ndarray],
    targets: np.ndarray,
    batch_size: int = 32,
    val_split_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation DataLoaders from processed dataset arrays.

    Args:
        features: Dictionary of multi-anchor features.
        targets: 2D target position coordinates [N, 2].
        batch_size: Mini-batch size.
        val_split_ratio: Proportion of samples reserved for validation.
        seed: Random seed for deterministic train/val split.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning them.

    Returns:
        Tuple of (train_loader, val_loader). If val_split_ratio is 0.0, val_loader is None.
    """
    full_dataset = UWBDataset(features, targets)
    total_len = len(full_dataset)

    if val_split_ratio > 0.0:
        val_len = int(total_len * val_split_ratio)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(
            full_dataset, [train_len, val_len], generator=generator
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, None
