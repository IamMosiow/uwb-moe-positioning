"""Evaluation and inference pipeline for UWB-MoE localization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from uwb_moe.config import AppConfig, load_config
from uwb_moe.data.processor import UWBDataProcessor
from uwb_moe.data.dataset import UWBDataset
from uwb_moe.models.moe_model import UWBMoEModel
from uwb_moe.utils.logging import setup_logger
from uwb_moe.utils.metrics import compute_localization_metrics, LocalizationMetrics
from uwb_moe.utils.visualization import (
    plot_gate_distribution,
    plot_trajectory_comparison,
    plot_error_cdf,
)


class Evaluator:
    """Manages inference, metric calculation, and diagnostic visualization."""

    def __init__(
        self,
        model: UWBMoEModel,
        data_loader: DataLoader,
        device: torch.device,
        output_dir: str | Path = "outputs/eval",
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("uwb_moe.evaluator")

    def run(self) -> Tuple[LocalizationMetrics, np.ndarray, np.ndarray, np.ndarray]:
        """Run inference over entire dataset and compute metrics.

        Returns:
            Tuple of (metrics, y_true, y_pred, gate_values).
        """
        self.model.eval()
        self.model.enable_gate_recording(True)
        self.model.reset_gate_values()

        y_true_list = []
        y_pred_list = []

        self.logger.info("Executing evaluation across %d batches...", len(self.data_loader))

        with torch.no_grad():
            for batch in self.data_loader:
                cir = batch["cir"].to(self.device)
                td = batch["td"].to(self.device)
                td_offset = batch["td_offset"].to(self.device)
                anchor_id = batch["anchor_id"].to(self.device)

                pos_pred = self.model(cir, td, td_offset, anchor_id, return_diversity_loss=False)

                y_pred_list.append(pos_pred.cpu().numpy())
                if "target" in batch:
                    y_true_list.append(batch["target"].cpu().numpy())

        y_pred = np.concatenate(y_pred_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0) if y_true_list else np.empty((0, 2))

        gate_values = self.model.get_gate_values().cpu().numpy()
        self.logger.info("Collected %d NLOS gate values across dataset.", gate_values.size)

        metrics = compute_localization_metrics(y_true, y_pred) if len(y_true) > 0 else None

        if metrics is not None:
            self.logger.info("\n%s", metrics.summary())

        # Generate evaluation visualizations
        hist_path = self.output_dir / "gate_values_histogram.png"
        plot_gate_distribution(gate_values, output_path=hist_path)
        self.logger.info("Saved NLOS gate histogram to %s", hist_path)

        if metrics is not None:
            traj_path = self.output_dir / "trajectory_comparison.png"
            plot_trajectory_comparison(y_true, y_pred, output_path=traj_path)
            self.logger.info("Saved trajectory comparison plot to %s", traj_path)

            cdf_path = self.output_dir / "error_cdf.png"
            plot_error_cdf(y_true, y_pred, output_path=cdf_path)
            self.logger.info("Saved error CDF plot to %s", cdf_path)

        return metrics, y_true, y_pred, gate_values


def evaluate_pipeline(
    config_path: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    test_hdf5_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[LocalizationMetrics, np.ndarray, np.ndarray]:
    """Execute end-to-end evaluation pipeline with a trained checkpoint.

    Args:
        config_path: Path to YAML config.
        checkpoint_path: Path to saved model checkpoint (.pt).
        test_hdf5_path: Path to evaluation HDF5 dataset.
        overrides: Configuration overrides.

    Returns:
        Tuple of (metrics, y_true, y_pred).
    """
    config = load_config(config_path, overrides)
    logger = setup_logger("uwb_moe.eval_pipeline")

    if checkpoint_path is None:
        checkpoint_path = config.eval.checkpoint_path or (
            Path(config.train.checkpoint_dir) / "best_model.pt"
        )
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file.resolve()}")

    # Resolve device
    if config.train.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.train.device)

    # Load preprocessing stats
    processor = UWBDataProcessor(cir_length=config.data.cir_length)
    stats_file = Path(config.train.checkpoint_dir) / "preprocessing_stats.json"
    if stats_file.is_file():
        processor.load_stats(stats_file)
    else:
        logger.warning("No preprocessing stats found at %s. Fitting on evaluation data.", stats_file)

    if test_hdf5_path is None:
        test_file = Path(config.data.raw_data_dir) / config.data.test_file
        if not test_file.is_file() and Path(config.data.test_file).is_file():
            test_file = Path(config.data.test_file)
    else:
        test_file = Path(test_hdf5_path)

    features, targets = processor.process_dataset(
        test_file,
        is_training=(not stats_file.is_file()),
    )

    eval_dataset = UWBDataset(features, targets)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # Initialize model and load weights
    model = UWBMoEModel(
        cir_length=config.model.cir_length,
        n_anchors=processor.n_anchors,
        embedding_dim=config.model.embedding_dim,
        feature_dim=config.model.feature_dim,
        gate_hidden_dim=config.model.gate_hidden_dim,
        num_experts=config.model.num_experts,
        expert_hidden_dim=config.model.expert_hidden_dim,
        position_head_hidden=config.model.position_head_hidden,
        output_dim=config.model.output_dim,
        diversity_loss_coeff=config.model.diversity_loss_coeff,
    )

    logger.info("Loading model weights from %s", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    evaluator = Evaluator(
        model=model,
        data_loader=eval_loader,
        device=device,
        output_dir=config.eval.output_dir,
    )

    metrics, y_true, y_pred, _ = evaluator.run()
    return metrics, y_true, y_pred


def main() -> None:
    """CLI entrypoint for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate UWB-MoE Localization Model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--data", type=str, default=None, help="Path to test HDF5 dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/eval", help="Evaluation output folder")
    args = parser.parse_args()

    evaluate_pipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        test_hdf5_path=args.data,
    )


if __name__ == "__main__":
    main()
