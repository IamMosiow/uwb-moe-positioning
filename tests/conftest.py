"""Test fixtures and synthetic UWB HDF5 generator."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import h5py
import numpy as np
import pytest
import torch


def create_synthetic_hdf5(
    filepath: str | Path,
    num_samples: int = 15,
    num_anchors: int = 4,
    cir_len: int = 64,
) -> Path:
    """Generate a valid mock UWB HDF5 dataset for unit tests."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    anchor_names = [f"00:11:22:33:44:0{i}" for i in range(num_anchors)]

    a_ids = []
    b_ids = []
    td_vals = []
    td_offset_vals = []
    pos_x_vals = []
    pos_y_vals = []
    cir_r_list = []
    cir_i_list = []

    for burst_id in range(num_samples):
        # Generate random position per burst
        pos_x = np.random.uniform(0.0, 10.0)
        pos_y = np.random.uniform(0.0, 10.0)

        # In each burst, 3 to 4 anchors respond
        active_k = np.random.randint(3, num_anchors + 1)
        chosen_anchors = np.random.choice(anchor_names, size=active_k, replace=False)

        for anc in chosen_anchors:
            a_ids.append(anc.encode("utf-8"))
            b_ids.append(burst_id)
            td_vals.append(np.random.normal(15.0, 3.0))
            td_offset_vals.append(np.random.normal(0.0, 0.5))
            pos_x_vals.append(pos_x)
            pos_y_vals.append(pos_y)

            # Synthetic CIR real and imaginary components
            cir_r = np.random.normal(0.0, 1.0, size=cir_len).astype(np.float32)
            cir_i = np.random.normal(0.0, 1.0, size=cir_len).astype(np.float32)
            cir_r_list.append(cir_r)
            cir_i_list.append(cir_i)

    with h5py.File(path, "w") as f:
        f.create_dataset("A_ID", data=a_ids)
        f.create_dataset("B_ID", data=np.array(b_ids, dtype=np.int32))
        f.create_dataset("TD", data=np.array(td_vals, dtype=np.float32))
        f.create_dataset("TD_OFFSET", data=np.array(td_offset_vals, dtype=np.float32))
        f.create_dataset("POS_X", data=np.array(pos_x_vals, dtype=np.float32))
        f.create_dataset("POS_Y", data=np.array(pos_y_vals, dtype=np.float32))
        f.create_dataset("CIR_R", data=np.array(cir_r_list, dtype=np.float32))
        f.create_dataset("CIR_I", data=np.array(cir_i_list, dtype=np.float32))

    return path


@pytest.fixture
def synthetic_hdf5(tmp_path: Path) -> Path:
    """Fixture providing a temporary synthetic HDF5 dataset."""
    hdf5_file = tmp_path / "synthetic_test_uwb.hdf5"
    return create_synthetic_hdf5(hdf5_file, num_samples=16, num_anchors=4, cir_len=64)


@pytest.fixture
def sample_batch():
    """Fixture providing a mock PyTorch mini-batch."""
    batch_size = 4
    num_anchors = 4
    cir_len = 64

    return {
        "cir": torch.randn(batch_size, num_anchors, cir_len, dtype=torch.float32),
        "td": torch.randn(batch_size, num_anchors, 1, dtype=torch.float32),
        "td_offset": torch.randn(batch_size, num_anchors, 1, dtype=torch.float32),
        "anchor_id": torch.randint(0, num_anchors, (batch_size, num_anchors), dtype=torch.long),
        "target": torch.randn(batch_size, 2, dtype=torch.float32),
    }
