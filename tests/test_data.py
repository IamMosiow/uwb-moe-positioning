"""Unit tests for UWB data processor, normalization, and Dataset wrapper."""

from pathlib import Path
import numpy as np
import pytest
import torch

from uwb_moe.data.processor import UWBDataProcessor, PreprocessingStats
from uwb_moe.data.dataset import UWBDataset, create_dataloaders


def test_load_hdf5_to_dataframe(synthetic_hdf5: Path):
    """Verify raw HDF5 loading returns valid columns and records."""
    processor = UWBDataProcessor(cir_length=64)
    df = processor.load_hdf5_to_dataframe(synthetic_hdf5)

    assert not df.empty
    expected_cols = {"A_ID", "B_ID", "TD", "TD_OFFSET", "POS_X", "POS_Y", "CIR_R", "CIR_I"}
    assert expected_cols.issubset(set(df.columns))
    assert df["B_ID"].dtype == np.int32


def test_process_dataset_training(synthetic_hdf5: Path):
    """Verify data processing during training discovers anchors and normalizes features."""
    processor = UWBDataProcessor(cir_length=64)
    features, targets = processor.process_dataset(synthetic_hdf5, is_training=True)

    assert processor.n_anchors == 4
    assert len(processor.mac_to_idx) == 4
    assert processor.stats is not None

    n_samples = len(targets)
    assert targets.shape == (n_samples, 2)
    assert features["cir"].shape == (n_samples, 4, 64)
    assert features["td"].shape == (n_samples, 4, 1)
    assert features["td_offset"].shape == (n_samples, 4, 1)
    assert features["anchor_id"].shape == (n_samples, 4)

    # CIR values must be non-negative (log magnitude)
    assert np.all(features["cir"] >= 0.0)


def test_process_dataset_inference(synthetic_hdf5: Path, tmp_path: Path):
    """Verify inference processing respects fitted statistics without data leakage."""
    processor = UWBDataProcessor(cir_length=64)
    processor.process_dataset(synthetic_hdf5, is_training=True)

    stats_file = tmp_path / "stats.json"
    processor.save_stats(stats_file)

    # New processor instance loading stats
    eval_processor = UWBDataProcessor(cir_length=64)
    eval_processor.load_stats(stats_file)

    features, targets = eval_processor.process_dataset(synthetic_hdf5, is_training=False)
    assert features["cir"].shape[1] == 4
    assert len(targets) == len(features["cir"])


def test_uwb_dataset_and_dataloaders(synthetic_hdf5: Path):
    """Verify UWBDataset PyTorch integration and batch generation."""
    processor = UWBDataProcessor(cir_length=64)
    features, targets = processor.process_dataset(synthetic_hdf5, is_training=True)

    dataset = UWBDataset(features, targets)
    assert len(dataset) == len(targets)

    sample = dataset[0]
    assert isinstance(sample["cir"], torch.Tensor)
    assert isinstance(sample["target"], torch.Tensor)
    assert sample["cir"].shape == (4, 64)
    assert sample["target"].shape == (2,)

    # Test dataloader creation with 0.25 validation split
    train_loader, val_loader = create_dataloaders(
        features=features,
        targets=targets,
        batch_size=4,
        val_split_ratio=0.25,
        num_workers=0,
    )
    assert train_loader is not None
    assert val_loader is not None

    batch = next(iter(train_loader))
    assert batch["cir"].shape[0] <= 4
    assert batch["target"].shape == (batch["cir"].shape[0], 2)
