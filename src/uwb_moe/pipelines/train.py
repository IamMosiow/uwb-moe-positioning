"""Modular training pipeline with checkpointing, validation, and early stopping."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from uwb_moe.config import AppConfig, load_config
from uwb_moe.data.processor import UWBDataProcessor
from uwb_moe.data.dataset import create_dataloaders
from uwb_moe.models.moe_model import UWBMoEModel, count_parameters
from uwb_moe.utils.logging import setup_logger
from uwb_moe.utils.seed import seed_everything


class Trainer:
    """Orchestrates model optimization, validation, and checkpoint persistence."""

    def __init__(
        self,
        model: UWBMoEModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        config: AppConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss(reduction="mean")
        self.logger = setup_logger(
            "uwb_moe.trainer",
            log_file=config.logging.log_file,
            level=config.logging.level,
        )

        self.checkpoint_dir = Path(config.train.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_rmse = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Execute a single training epoch.

        Args:
            epoch: Current epoch index (1-based).

        Returns:
            Tuple of (average_total_loss, localization_rmse).
        """
        self.model.train()
        total_loss = 0.0
        total_squared_error = 0.0
        num_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            cir = batch["cir"].to(self.device)
            td = batch["td"].to(self.device)
            td_offset = batch["td_offset"].to(self.device)
            anchor_id = batch["anchor_id"].to(self.device)
            target = batch["target"].to(self.device)
            batch_sz = target.size(0)

            self.optimizer.zero_grad()
            pred, diversity_loss = self.model(cir, td, td_offset, anchor_id, return_diversity_loss=True)
            main_loss = self.criterion(pred, target)

            loss = main_loss + self.model.diversity_loss_coeff * diversity_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_sz
            total_squared_error += main_loss.item() * batch_sz
            num_samples += batch_sz

            if (batch_idx + 1) % self.config.train.log_interval == 0:
                self.logger.debug(
                    "Epoch %d [%d/%d] - Batch Loss: %.6f | Div Loss: %.6f",
                    epoch, batch_idx + 1, len(self.train_loader), loss.item(), diversity_loss.item(),
                )

        avg_loss = total_loss / max(1, num_samples)
        rmse = float(np.sqrt(total_squared_error / max(1, num_samples)))
        return avg_loss, rmse

    def validate(self) -> Tuple[float, float]:
        """Evaluate model on validation dataset.

        Returns:
            Tuple of (average_total_loss, localization_rmse).
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        total_squared_error = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                cir = batch["cir"].to(self.device)
                td = batch["td"].to(self.device)
                td_offset = batch["td_offset"].to(self.device)
                anchor_id = batch["anchor_id"].to(self.device)
                target = batch["target"].to(self.device)
                batch_sz = target.size(0)

                pred, diversity_loss = self.model(cir, td, td_offset, anchor_id, return_diversity_loss=True)
                main_loss = self.criterion(pred, target)
                loss = main_loss + self.model.diversity_loss_coeff * diversity_loss

                total_loss += loss.item() * batch_sz
                total_squared_error += main_loss.item() * batch_sz
                num_samples += batch_sz

        avg_loss = total_loss / max(1, num_samples)
        rmse = float(np.sqrt(total_squared_error / max(1, num_samples)))
        return avg_loss, rmse

    def save_checkpoint(self, epoch: int, val_rmse: float, is_best: bool = False) -> Path:
        """Save training state checkpoint."""
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_rmse": val_rmse,
            "config": self.config,
        }

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        if not self.config.train.save_best_only:
            torch.save(checkpoint_data, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            self.logger.info("Saved new best model checkpoint to %s (Val RMSE: %.4f m)", best_path, val_rmse)
            return best_path

        return path

    def run(self) -> Dict[str, list[float]]:
        """Run complete training cycle.

        Returns:
            Dictionary containing epoch-wise loss and RMSE logs.
        """
        history: Dict[str, list[float]] = {
            "train_loss": [],
            "train_rmse": [],
            "val_loss": [],
            "val_rmse": [],
        }

        self.logger.info("Beginning training for %d epochs on device: %s", self.config.train.epochs, self.device)
        param_counts = count_parameters(self.model)
        self.logger.info("Total parameters: %s | Trainable: %s", f"{param_counts['Total']:,}", f"{param_counts['Trainable']:,}")

        for epoch in range(1, self.config.train.epochs + 1):
            train_loss, train_rmse = self.train_epoch(epoch)
            history["train_loss"].append(train_loss)
            history["train_rmse"].append(train_rmse)

            if self.val_loader is not None:
                val_loss, val_rmse = self.validate()
                history["val_loss"].append(val_loss)
                history["val_rmse"].append(val_rmse)

                self.logger.info(
                    "Epoch %2d/%2d | Train Loss: %.6f | Train RMSE: %.4f m | Val Loss: %.6f | Val RMSE: %.4f m",
                    epoch, self.config.train.epochs, train_loss, train_rmse, val_loss, val_rmse,
                )

                if val_rmse < self.best_val_rmse:
                    self.best_val_rmse = val_rmse
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, val_rmse, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.train.early_stopping_patience:
                        self.logger.info("Early stopping triggered after %d epochs without improvement.", self.patience_counter)
                        break
            else:
                self.logger.info(
                    "Epoch %2d/%2d | Train Loss: %.6f | Train RMSE: %.4f m",
                    epoch, self.config.train.epochs, train_loss, train_rmse,
                )
                self.save_checkpoint(epoch, train_rmse, is_best=True)

        return history


def train_pipeline(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[UWBMoEModel, Dict[str, list[float]]]:
    """High-level training entry point: loads data, builds model, and executes training.

    Args:
        config_path: Path to YAML config.
        overrides: Dictionary of config overrides.

    Returns:
        Tuple of (trained_model, history_dict).
    """
    config = load_config(config_path, overrides)
    seed_everything(config.train.seed)

    logger = setup_logger("uwb_moe.pipeline", log_file=config.logging.log_file, level=config.logging.level)

    # Resolve device
    if config.train.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.train.device)

    # 1. Process dataset
    processor = UWBDataProcessor(cir_length=config.data.cir_length)
    train_file = Path(config.data.raw_data_dir) / config.data.train_file
    if not train_file.is_file():
        # Fallback to current directory if running in root
        if Path(config.data.train_file).is_file():
            train_file = Path(config.data.train_file)
        else:
            raise FileNotFoundError(f"Training dataset file not found at {train_file.resolve()}")

    features, targets = processor.process_dataset(train_file, is_training=True)

    # Save preprocessing metadata for downstream inference
    stats_path = Path(config.train.checkpoint_dir) / "preprocessing_stats.json"
    processor.save_stats(stats_path)
    logger.info("Saved preprocessing stats to %s", stats_path)

    # 2. Build DataLoaders
    train_loader, val_loader = create_dataloaders(
        features=features,
        targets=targets,
        batch_size=config.data.batch_size,
        val_split_ratio=config.data.val_split_ratio,
        seed=config.train.seed,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # 3. Instantiate model
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

    # 4. Build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    # 5. Execute training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        device=device,
    )

    history = trainer.run()
    return model, history


def main() -> None:
    """CLI entrypoint for training."""
    parser = argparse.ArgumentParser(description="Train UWB-MoE Localization Model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides.setdefault("train", {})["epochs"] = args.epochs
    if args.lr is not None:
        overrides.setdefault("train", {})["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size

    train_pipeline(config_path=args.config, overrides=overrides)


if __name__ == "__main__":
    main()
