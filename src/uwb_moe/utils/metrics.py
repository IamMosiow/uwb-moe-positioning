"""Evaluation metrics for indoor localization systems."""

from dataclasses import dataclass
from typing import Dict, Union
import numpy as np
import torch


@dataclass
class LocalizationMetrics:
    """Container for localization evaluation results."""

    rmse: float
    mae: float
    median_error: float
    p90_error: float
    p95_error: float
    max_error: float
    mse: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "median_error": self.median_error,
            "p90_error": self.p90_error,
            "p95_error": self.p95_error,
            "max_error": self.max_error,
            "mse": self.mse,
        }

    def summary(self) -> str:
        """Formatted summary table of metrics."""
        return (
            f"Localization Metrics:\n"
            f"  - RMSE:         {self.rmse:.4f} m\n"
            f"  - MAE (Mean):   {self.mae:.4f} m\n"
            f"  - Median (p50): {self.median_error:.4f} m\n"
            f"  - 90th %-ile:   {self.p90_error:.4f} m\n"
            f"  - 95th %-ile:   {self.p95_error:.4f} m\n"
            f"  - Max Error:    {self.max_error:.4f} m"
        )


def compute_localization_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> LocalizationMetrics:
    """Compute comprehensive 2D localization error metrics.

    Args:
        y_true: Ground truth 2D coordinates [N, 2].
        y_pred: Predicted 2D coordinates [N, 2].

    Returns:
        LocalizationMetrics dataclass with RMSE, MAE, percentiles.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Euclidean distance error per sample: sqrt((x - x_hat)^2 + (y - y_hat)^2)
    euclidean_errors = np.linalg.norm(y_true - y_pred, axis=1)
    squared_errors = euclidean_errors ** 2

    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(euclidean_errors))
    median_error = float(np.median(euclidean_errors))
    p90_error = float(np.percentile(euclidean_errors, 90))
    p95_error = float(np.percentile(euclidean_errors, 95))
    max_error = float(np.max(euclidean_errors))

    return LocalizationMetrics(
        rmse=rmse,
        mae=mae,
        median_error=median_error,
        p90_error=p90_error,
        p95_error=p95_error,
        max_error=max_error,
        mse=mse,
    )
