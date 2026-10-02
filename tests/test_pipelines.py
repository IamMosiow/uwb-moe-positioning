"""Integration tests for end-to-end training and evaluation pipelines."""

from pathlib import Path
import pytest
import torch

from uwb_moe.config import AppConfig, DataConfig, ModelConfig, TrainConfig, EvalConfig, LoggingConfig
from uwb_moe.pipelines.train import Trainer
from uwb_moe.pipelines.evaluate import Evaluator
from uwb_moe.data.processor import UWBDataProcessor
from uwb_moe.data.dataset import create_dataloaders
from uwb_moe.models.moe_model import UWBMoEModel


def test_pipeline_integration(synthetic_hdf5: Path, tmp_path: Path):
    """Run an end-to-end integration test of processing, training, checkpointing, and evaluation."""
    processor = UWBDataProcessor(cir_length=64)
    features, targets = processor.process_dataset(synthetic_hdf5, is_training=True)

    checkpoint_dir = tmp_path / "checkpoints"
    eval_dir = tmp_path / "eval"

    cfg = AppConfig(
        data=DataConfig(cir_length=64, batch_size=4, val_split_ratio=0.25, num_workers=0),
        model=ModelConfig(
            cir_length=64,
            n_anchors=4,
            embedding_dim=8,
            feature_dim=32,
            gate_hidden_dim=16,
            expert_hidden_dim=32,
            position_head_hidden=16,
        ),
        train=TrainConfig(
            epochs=2,
            checkpoint_dir=str(checkpoint_dir),
            device="cpu",
            save_best_only=True,
        ),
        eval=EvalConfig(
            output_dir=str(eval_dir),
        ),
        logging=LoggingConfig(level="WARNING"),
    )

    train_loader, val_loader = create_dataloaders(
        features=features,
        targets=targets,
        batch_size=cfg.data.batch_size,
        val_split_ratio=cfg.data.val_split_ratio,
        num_workers=0,
    )

    model = UWBMoEModel(
        cir_length=cfg.model.cir_length,
        n_anchors=processor.n_anchors,
        embedding_dim=cfg.model.embedding_dim,
        feature_dim=cfg.model.feature_dim,
        gate_hidden_dim=cfg.model.gate_hidden_dim,
        num_experts=cfg.model.num_experts,
        expert_hidden_dim=cfg.model.expert_hidden_dim,
        position_head_hidden=cfg.model.position_head_hidden,
        output_dim=cfg.model.output_dim,
        diversity_loss_coeff=cfg.model.diversity_loss_coeff,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=cfg,
        device=torch.device("cpu"),
    )

    history = trainer.run()
    assert len(history["train_loss"]) == 2
    assert (checkpoint_dir / "best_model.pt").is_file()

    # Test evaluation
    evaluator = Evaluator(
        model=model,
        data_loader=val_loader,
        device=torch.device("cpu"),
        output_dir=eval_dir,
    )
    metrics, y_true, y_pred, gate_vals = evaluator.run()

    assert metrics is not None
    assert metrics.rmse >= 0.0
    assert (eval_dir / "gate_values_histogram.png").is_file()
    assert (eval_dir / "trajectory_comparison.png").is_file()
    assert (eval_dir / "error_cdf.png").is_file()
