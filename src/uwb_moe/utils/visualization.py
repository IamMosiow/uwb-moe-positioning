"""Plotting utilities for localization predictions, gate distributions, and error CDFs."""

from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use("Agg")  # Ensure headless-safe rendering across CLI, CI, and server environments
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_gate_distribution(
    gate_values: Union[np.ndarray, torch.Tensor],
    output_path: Optional[str | Path] = None,
    bins: int = 50,
    title: str = "NLOS Gate Value Distribution",
) -> None:
    """Plot and optionally save a histogram of expert gating activations.

    Args:
        gate_values: 1D array of gating values (e.g. NLOS probability).
        output_path: Optional path to save figure.
        bins: Number of histogram bins.
        title: Plot title.
    """
    if isinstance(gate_values, torch.Tensor):
        gate_values = gate_values.detach().cpu().numpy()

    gate_values = np.asarray(gate_values).ravel()
    if gate_values.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(gate_values, bins=bins, color="#3498db", edgecolor="#2c3e50", alpha=0.85)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Gate Weight (NLOS Expert)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=300)
    plt.close()


def plot_trajectory_comparison(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    output_path: Optional[str | Path] = None,
    title: str = "UWB 2D Localization: Ground Truth vs Prediction",
    max_points: int = 200,
) -> None:
    """Plot 2D coordinates comparison between ground truth and model predictions.

    Args:
        y_true: Ground truth positions [N, 2].
        y_pred: Predicted positions [N, 2].
        output_path: Optional path to save figure.
        title: Plot title.
        max_points: Subsample count to avoid cluttered plots.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if len(y_true) > max_points:
        indices = np.linspace(0, len(y_true) - 1, max_points, dtype=int)
        y_true = y_true[indices]
        y_pred = y_pred[indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true[:, 0], y_true[:, 1], c="#2ecc71", label="Ground Truth", alpha=0.7, s=30)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c="#e74c3c", label="Prediction", alpha=0.7, s=30, marker="x")

    for i in range(len(y_true)):
        plt.plot([y_true[i, 0], y_pred[i, 0]], [y_true[i, 1], y_pred[i, 1]], "gray", alpha=0.3, linewidth=0.8)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("X Coordinate (m)", fontsize=12)
    plt.ylabel("Y Coordinate (m)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=300)
    plt.close()


def plot_error_cdf(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    output_path: Optional[str | Path] = None,
    title: str = "Cumulative Distribution Function (CDF) of Localization Error",
) -> None:
    """Plot the Cumulative Distribution Function (CDF) of the positioning error.

    Args:
        y_true: Ground truth positions [N, 2].
        y_pred: Predicted positions [N, 2].
        output_path: Optional path to save figure.
        title: Plot title.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    errors = np.linalg.norm(y_true - y_pred, axis=1)
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_errors, cdf, color="#8e44ad", linewidth=2.0, label="UWB-MoE")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Position Error (m)", fontsize=12)
    plt.ylabel("Cumulative Probability", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=300)
    plt.close()
