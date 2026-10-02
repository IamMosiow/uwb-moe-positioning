"""Data processing pipeline for raw UWB HDF5 measurements."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import h5py
import numpy as np
import pandas as pd

from uwb_moe.utils.logging import get_logger

logger = get_logger("uwb_moe.processor")


@dataclass
class PreprocessingStats:
    """Normalization statistics and anchor mappings for reproducible inference."""

    mac_to_idx: Dict[str, int]
    n_anchors: int
    cir_length: int
    td_mean: float
    td_std: float
    td_offset_mean: float
    td_offset_std: float

    def to_json(self, path: str | Path) -> None:
        """Serialize preprocessing stats to JSON."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> PreprocessingStats:
        """Deserialize preprocessing stats from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class UWBDataProcessor:
    """Extracts, normalizes, and packages raw UWB HDF5 telemetry into aligned tensors."""

    def __init__(self, cir_length: int = 152):
        """Initialize processor.

        Args:
            cir_length: Number of CIR samples to truncate or pad per anchor measurement.
        """
        self.cir_length = cir_length
        self.mac_to_idx: Dict[str, int] = {}
        self.n_anchors = 0
        self.stats: Optional[PreprocessingStats] = None

    def load_hdf5_to_dataframe(self, filepath: str | Path) -> pd.DataFrame:
        """Parse raw HDF5 file containing anchor bursts and channel responses into a DataFrame.

        Args:
            filepath: Path to the .hdf5 file.

        Returns:
            pd.DataFrame with parsed anchor measurements, CIR arrays, and positions.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found at: {path.resolve()}")

        logger.info("Loading HDF5 data from %s", path)
        with h5py.File(path, "r") as f:
            raw_a_ids = f["A_ID"][:]
            a_ids = [
                val.decode("utf-8") if isinstance(val, (bytes, np.bytes_)) else str(val)
                for val in raw_a_ids
            ]

            cir_r_raw = f["CIR_R"][:]
            cir_i_raw = f["CIR_I"][:]

            df = pd.DataFrame({
                "A_ID": a_ids,
                "B_ID": f["B_ID"][:].astype(np.int32),
                "TD": f["TD"][:].astype(np.float32),
                "TD_OFFSET": f["TD_OFFSET"][:].astype(np.float32),
                "POS_X": f["POS_X"][:].astype(np.float32),
                "POS_Y": f["POS_Y"][:].astype(np.float32),
                "CIR_R": [np.asarray(r, dtype=np.float32) for r in cir_r_raw],
                "CIR_I": [np.asarray(i, dtype=np.float32) for i in cir_i_raw],
            })

        logger.info("Loaded %d rows across %d unique bursts.", len(df), df["B_ID"].nunique())
        return df

    def compute_cir_magnitude(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute log-scaled CIR magnitude: log1p(sqrt(CIR_R^2 + CIR_I^2))."""
        logger.debug("Computing CIR magnitude log envelope...")
        cir_mag_list = []
        for r, i in zip(df["CIR_R"], df["CIR_I"]):
            mag = np.log1p(np.sqrt(np.square(r) + np.square(i)))
            cir_mag_list.append(mag.astype(np.float32))
        df["CIR_MAG"] = cir_mag_list
        return df

    def process_dataset(
        self,
        filepath: str | Path,
        is_training: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Process an HDF5 dataset file into aligned, normalized multi-anchor tensors.

        Args:
            filepath: Path to HDF5 file.
            is_training: If True, computes and stores normalization stats and anchor mapping.
                         If False, uses previously fitted normalization stats and anchor mapping.

        Returns:
            Tuple containing:
                - Dictionary of features:
                    - 'cir': [N, n_anchors, cir_length] float32
                    - 'td': [N, n_anchors, 1] float32
                    - 'td_offset': [N, n_anchors, 1] float32
                    - 'anchor_id': [N, n_anchors] int64
                - Targets array:
                    - y: [N, 2] float32 (2D ground truth coordinates)
        """
        df = self.load_hdf5_to_dataframe(filepath)
        df = self.compute_cir_magnitude(df)

        if is_training:
            unique_anchors = sorted(df["A_ID"].unique())
            self.mac_to_idx = {mac: idx for idx, mac in enumerate(unique_anchors)}
            self.n_anchors = len(unique_anchors)
            logger.info("Fitted %d unique anchors: %s", self.n_anchors, self.mac_to_idx)
        else:
            if not self.mac_to_idx or self.n_anchors == 0:
                raise ValueError(
                    "Processor must either be run with is_training=True first "
                    "or have stats loaded via load_stats() before evaluation."
                )

        grouped = df.groupby("B_ID", sort=False)
        n_samples = len(grouped)
        num_anchors = self.n_anchors
        target_len = self.cir_length

        x_cir = np.zeros((n_samples, num_anchors, target_len), dtype=np.float32)
        x_td = np.zeros((n_samples, num_anchors, 1), dtype=np.float32)
        x_td_offset = np.zeros((n_samples, num_anchors, 1), dtype=np.float32)
        x_anchor = np.zeros((n_samples, num_anchors), dtype=np.int64)
        y = np.zeros((n_samples, 2), dtype=np.float32)

        for sample_idx, (_, group) in enumerate(grouped):
            # Ground truth coordinates are consistent per burst (from transmitter tag)
            y[sample_idx] = group.iloc[0][["POS_X", "POS_Y"]].values

            for _, row in group.iterrows():
                anchor_id = row["A_ID"]
                if anchor_id not in self.mac_to_idx:
                    continue

                a_idx = self.mac_to_idx[anchor_id]
                cir_mag = row["CIR_MAG"][:target_len]
                actual_len = len(cir_mag)

                x_cir[sample_idx, a_idx, :actual_len] = cir_mag
                x_td[sample_idx, a_idx, 0] = row["TD"]
                x_td_offset[sample_idx, a_idx, 0] = row["TD_OFFSET"]
                x_anchor[sample_idx, a_idx] = a_idx

        # Compute or apply normalization statistics
        eps = 1e-9
        if is_training:
            td_mean = float(x_td.mean())
            td_std = float(x_td.std() + eps)
            td_offset_mean = float(x_td_offset.mean())
            td_offset_std = float(x_td_offset.std() + eps)

            self.stats = PreprocessingStats(
                mac_to_idx=self.mac_to_idx,
                n_anchors=self.n_anchors,
                cir_length=self.cir_length,
                td_mean=td_mean,
                td_std=td_std,
                td_offset_mean=td_offset_mean,
                td_offset_std=td_offset_std,
            )
            logger.info(
                "Training Normalization Stats - TD mean: %.4f, std: %.4f | TD_OFFSET mean: %.4f, std: %.4f",
                td_mean, td_std, td_offset_mean, td_offset_std,
            )
        else:
            if self.stats is None:
                raise RuntimeError("Normalization stats must be available during inference.")
            td_mean = self.stats.td_mean
            td_std = self.stats.td_std
            td_offset_mean = self.stats.td_offset_mean
            td_offset_std = self.stats.td_offset_std

        x_td = (x_td - td_mean) / td_std
        x_td_offset = (x_td_offset - td_offset_mean) / td_offset_std

        features = {
            "cir": x_cir,
            "td": x_td,
            "td_offset": x_td_offset,
            "anchor_id": x_anchor,
        }
        return features, y

    def save_stats(self, path: str | Path) -> None:
        """Save fitted preprocessing stats to a JSON file."""
        if self.stats is None:
            raise ValueError("No preprocessing stats available to save. Call process_dataset() first.")
        self.stats.to_json(path)

    def load_stats(self, path: str | Path) -> None:
        """Load fitted preprocessing stats from a JSON file."""
        self.stats = PreprocessingStats.from_json(path)
        self.mac_to_idx = self.stats.mac_to_idx
        self.n_anchors = self.stats.n_anchors
        self.cir_length = self.stats.cir_length
        logger.info("Loaded preprocessing stats with %d anchors from %s", self.n_anchors, path)
