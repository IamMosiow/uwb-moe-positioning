"""Configuration schema and loader for UWB-MoE localization system."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class DataConfig:
    """Dataset and preprocessing configurations."""

    dataset_url: str = (
        "https://www.iis.fraunhofer.de/content/dam/iis/de/doc/lv/dataanalytics/uwb_dataset.zip"
    )
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    train_file: str = "uwb_test_data.hdf5"
    test_file: str = "uwb_test_data.hdf5"
    cir_length: int = 152
    val_split_ratio: float = 0.2
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    save_preprocessed: bool = False


@dataclass
class ModelConfig:
    """Neural network architecture hyperparameters."""

    cir_length: int = 152
    n_anchors: int = 8
    embedding_dim: int = 16
    cnn_channels: list[int] = field(default_factory=lambda: [16, 32])
    cnn_kernel_size: int = 5
    pool_size: int = 16
    feature_dim: int = 128
    gate_hidden_dim: int = 64
    num_experts: int = 2
    expert_hidden_dim: int = 128
    position_head_hidden: int = 64
    output_dim: int = 2
    diversity_loss_coeff: float = 0.01


@dataclass
class TrainConfig:
    """Training loop hyperparameters and optimization settings."""

    seed: int = 42
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    checkpoint_dir: str = "outputs/checkpoints"
    save_best_only: bool = True
    early_stopping_patience: int = 10
    log_interval: int = 200


@dataclass
class EvalConfig:
    """Evaluation and visualization settings."""

    checkpoint_path: Optional[str] = None
    output_dir: str = "outputs/eval"
    plot_histogram: bool = True
    histogram_bins: int = 50
    plot_trajectories: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_file: Optional[str] = "outputs/training.log"
    verbose: bool = True


@dataclass
class AppConfig:
    """Consolidated master application configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update nested dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(config_path: str | Path | None = None, overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    """Load configuration from a YAML file with optional nested dictionary overrides.

    Args:
        config_path: Path to the YAML configuration file. If None, default config is returned.
        overrides: Nested dictionary of overrides to apply on top of file configuration.

    Returns:
        AppConfig instance populated with loaded parameters.
    """
    raw_dict: dict[str, Any] = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                raw_dict = loaded

    if overrides:
        _deep_update(raw_dict, overrides)

    data_cfg = DataConfig(**raw_dict.get("data", {}))
    model_cfg = ModelConfig(**raw_dict.get("model", {}))
    train_cfg = TrainConfig(**raw_dict.get("train", {}))
    eval_cfg = EvalConfig(**raw_dict.get("eval", {}))
    logging_cfg = LoggingConfig(**raw_dict.get("logging", {}))

    return AppConfig(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        eval=eval_cfg,
        logging=logging_cfg,
    )


def save_config(config: AppConfig, output_path: str | Path) -> None:
    """Serialize an AppConfig instance to a YAML file.

    Args:
        config: The AppConfig instance to save.
        output_path: Path where the YAML file should be written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
