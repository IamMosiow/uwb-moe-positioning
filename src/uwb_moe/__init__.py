"""UWB-MoE: Modular Mixture of Experts for Ultra-Wideband Indoor Localization.

A production-grade PyTorch framework for indoor positioning using Channel Impulse
Response (CIR) and Time Difference (TD) measurements with dynamic anchor attention
and soft-gated Line-of-Sight (LOS) / Non-Line-of-Sight (NLOS) experts.
"""

import os
# Prevent OpenMP runtime conflict on Windows between PyTorch and MKL
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from uwb_moe.models.moe_model import UWBMoEModel
from uwb_moe.data.processor import UWBDataProcessor
from uwb_moe.data.dataset import UWBDataset, create_dataloaders
from uwb_moe.config import AppConfig, load_config

__version__ = "0.1.0"
__all__ = [
    "UWBMoEModel",
    "UWBDataProcessor",
    "UWBDataset",
    "create_dataloaders",
    "AppConfig",
    "load_config",
    "__version__",
]
