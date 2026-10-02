# UWB-MoE: Deep Mixture of Experts for Ultra-Wideband Indoor Localization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/IamMosiow/uwb-moe-positioning/actions/workflows/ci.yml/badge.svg)](https://github.com/IamMosiow/uwb-moe-positioning/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, production-grade PyTorch framework for indoor positioning using Ultra-Wideband (UWB) Channel Impulse Response (CIR) envelopes and Time Difference (TD) measurements. 

**UWB-MoE** addresses severe multipath and Non-Line-of-Sight (NLOS) distortions by pairing temporal 1D convolutional feature extraction with soft-routed regime experts (LOS vs. NLOS), active-masked multi-anchor attention pooling, and load-balancing diversity regularization.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Anchor CIR Envelopes      |
                      |   + Time Differences (TD)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       CIREncoder (1D CNN)   |
                      |  Conv1d -> ReLU -> Conv1d   |
                      |  -> AdaptiveAvgPool1d       |
                      |  -> Concat with TD & Emb    |
                      +--------------+--------------+
                                     |
                         +-----------+-----------+
                         |                       |
                         v                       v
               +-------------------+   +-------------------+
               |     SoftGate      |   |   Condition       |
               | (Routing Network) |   |   Experts         |
               +---------+---------+   |  - LOS Expert     |
                         |             |  - NLOS Expert    |
                         | (Weights)   +---------+---------+
                         |                       |
                         +-----------+-----------+
                                     |
                                     v
                      +-----------------------------+
                      |     Soft Mixture Fusion     |
                      | z = w_LOS * e_LOS           |
                      |     + w_NLOS * e_NLOS       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Multi-Anchor Attention    |
                      |   - Dynamic Masking         |
                      |   - Attention Pooling       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Position Regression      |
                      |    Head: (X, Y) Coordinates|
                      +-----------------------------+
```

---

## Key Features

- **Multi-Modal CIR Feature Extraction**: Combines 1D CNN representations of CIR envelopes with normalized Time Differences ($\tau$), TD offsets ($\Delta\tau$), and learned spatial anchor embeddings.
- **Regime-Specialized Experts (MoE)**: Soft gating router dynamically assigns Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) experts to mitigate ranging biases.
- **Load Balancing Diversity Regularization**: Prevents expert starvation through a differentiable diversity regularization loss.
- **Dynamic Anchor Attention with Safe Masking**: Handles variable or missing anchor constellations per transmission burst without numerical instability.
- **Leakage-Free Preprocessing**: Normalization statistics and anchor mappings are fitted on training data and persisted to JSON for reproducible test-set inference.
- **Config-Driven Architecture**: Strongly typed dataclasses with declarative YAML configuration files and CLI overrides.
- **Comprehensive Evaluation**: Automated metric generation (RMSE, MAE, 50th/90th/95th percentiles, max error), error CDFs, and gating activation histograms.

---

## Directory Layout

```
.
├── configs/
│   ├── default.yaml            # Master production configuration
│   └── synthetic.yaml          # Fast smoke test / CI configuration
├── data/
│   ├── raw/                    # Raw HDF5 / zip datasets (gitignored)
│   └── processed/              # Normalized tensors (optional)
├── docs/
│   └── architecture.md         # Detailed mathematical formulation
├── notebooks/
│   └── demo_uwb_moe.ipynb      # Step-by-step interactive demonstration
├── outputs/
│   ├── checkpoints/            # Model weights & preprocessing metadata
│   └── eval/                   # Plots, histograms, and metric summaries
├── src/
│   └── uwb_moe/
│       ├── __init__.py         # Package root
│       ├── config.py           # Dataclasses and YAML loader
│       ├── data/
│       │   ├── download.py     # Fraunhofer dataset downloader
│       │   ├── processor.py    # UWBDataProcessor & stats serialization
│       │   └── dataset.py      # PyTorch UWBDataset & DataLoader factory
│       ├── models/
│       │   ├── encoder.py      # CIREncoder (1D CNN + feature fusion)
│       │   ├── gate.py         # SoftGate router
│       │   ├── experts.py      # Expert MLP blocks
│       │   ├── attention.py    # Active-masked spatial attention
│       │   ├── loss.py         # DiversityLoss & LocalizationLoss
│       │   └── moe_model.py    # Complete UWBMoEModel & parameter counter
│       ├── pipelines/
│       │   ├── train.py        # Trainer class & training loop CLI
│       │   └── evaluate.py     # Evaluator class & diagnostics CLI
│       └── utils/
│           ├── logging.py      # Structured console & file logging
│           ├── seed.py         # Multi-framework deterministic seeding
│           ├── metrics.py      # RMSE, MAE, percentile calculations
│           └── visualization.py# Gate histograms, CDFs, trajectory plots
├── tests/
│   ├── conftest.py             # Synthetic fixtures & HDF5 generator
│   ├── test_data.py            # Preprocessing & Dataset unit tests
│   ├── test_models.py          # Forward/backward & loss unit tests
│   └── test_pipelines.py       # End-to-end integration tests
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules for ML workflows
├── pyproject.toml              # Build system, CLI console scripts, tools
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Installation

### Option A: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/IamMosiow/uwb-moe-positioning.git
cd "uwb-moe-positioning"

# Create and activate environment
conda create -n uwb_moe python=3.10 -y
conda activate uwb_moe

# Install PyTorch with CUDA support (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies and editable package
pip install -e ".[dev]"
```

### Option B: Using Python `venv`

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart

### 1. Download Dataset
Download and extract the Fraunhofer IIS UWB dataset (`uwb_dataset.zip`):

```bash
# Using CLI entry point
uwb-download --dest-dir data/raw

# Or via Python module
python -m uwb_moe.data.download --dest-dir data/raw
```

### 2. Train the Model
Train the UWB-MoE model using default settings or custom overrides:

```bash
# Train with default configuration
uwb-train --config configs/default.yaml

# Override epochs and batch size directly via CLI
uwb-train --config configs/default.yaml --epochs 30 --batch-size 64 --lr 0.0005
```

Checkpoints and preprocessing metadata are saved to `outputs/checkpoints/best_model.pt` and `outputs/checkpoints/preprocessing_stats.json`.

### 3. Evaluate and Visualize
Run inference on evaluation data and generate metric summaries, error CDFs, and gate histograms:

```bash
uwb-eval --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt
```

Outputs generated in `outputs/eval/`:
- `gate_values_histogram.png`: Frequency distribution of NLOS routing probabilities.
- `trajectory_comparison.png`: 2D scatter comparison of ground truth vs predicted coordinates.
- `error_cdf.png`: Cumulative Distribution Function of positioning error.

---

## Python API Usage

You can easily embed `uwb_moe` into your custom scripts or research notebooks:

```python
import torch
from uwb_moe.config import load_config
from uwb_moe.data.processor import UWBDataProcessor
from uwb_moe.data.dataset import create_dataloaders
from uwb_moe.models.moe_model import UWBMoEModel
from uwb_moe.pipelines.train import Trainer
from uwb_moe.pipelines.evaluate import Evaluator

# 1. Load config
config = load_config("configs/default.yaml")

# 2. Process data
processor = UWBDataProcessor(cir_length=config.data.cir_length)
features, targets = processor.process_dataset("data/raw/uwb_test_data.hdf5", is_training=True)

# 3. Create loaders
train_loader, val_loader = create_dataloaders(
    features=features,
    targets=targets,
    batch_size=config.data.batch_size,
    val_split_ratio=0.2,
)

# 4. Initialize model & optimizer
model = UWBMoEModel(
    cir_length=config.model.cir_length,
    n_anchors=processor.n_anchors,
    embedding_dim=config.model.embedding_dim,
    diversity_loss_coeff=config.model.diversity_loss_coeff,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 5. Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    config=config,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
history = trainer.run()

# 6. Evaluate
evaluator = Evaluator(model, val_loader, device=trainer.device)
metrics, y_true, y_pred, gate_vals = evaluator.run()
print(metrics.summary())
```

---

## Testing

Run the full automated test suite (including unit tests for modules, datasets, and end-to-end integration):

```bash
pytest tests/ -v
```

All tests execute on CPU using synthetic fixtures without requiring large external dataset downloads.

---

## Configuration Reference

Key configuration options in `configs/default.yaml`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data.cir_length` | `int` | `152` | Time sample length of CIR magnitude envelope |
| `data.val_split_ratio` | `float` | `0.2` | Fraction of bursts reserved for validation |
| `data.batch_size` | `int` | `32` | Training and evaluation batch size |
| `model.embedding_dim` | `int` | `16` | Learned anchor identifier embedding dimension |
| `model.feature_dim` | `int` | `128` | Multi-modal fused latent dimension |
| `model.num_experts` | `int` | `2` | Number of specialized expert branches (LOS / NLOS) |
| `model.diversity_loss_coeff` | `float` | `0.01` | Regularization coefficient $\lambda$ for load balancing |
| `train.epochs` | `int` | `20` | Number of training epochs |
| `train.early_stopping_patience` | `int` | `10` | Epochs without validation improvement before early stopping |

---

## Citation

If you use this codebase, methodology, or models in your academic research, please cite this work using the following BibTeX entries:

### Thesis & Software
```bibtex
@mastersthesis{mousavi2026uwbmoe,
  author       = {Seyyed Mostafa Mousavi},
  title        = {Deep Mixture of Experts for Ultra-Wideband (UWB) Indoor Localization},
  school       = {Your University / Institution},
  year         = {2026},
  url          = {https://github.com/IamMosiow/uwb-moe-positioning}
}

@software{mousavi2026uwbmoe_code,
  author       = {Seyyed Mostafa Mousavi},
  title        = {UWB-MoE: Modular Mixture-of-Experts for Ultra-Wideband Indoor Positioning},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/IamMosiow/uwb-moe-positioning}}
}
```

### Dataset Citation
If utilizing the benchmark dataset, please also cite the Fraunhofer IIS UWB dataset:
```bibtex
@misc{fraunhofer_uwb_dataset,
  author       = {{Fraunhofer Institute for Integrated Circuits (IIS)}},
  title        = {UWB Indoor Localization Dataset with Channel Impulse Response (CIR)},
  year         = {2021},
  howpublished = {\url{https://www.iis.fraunhofer.de/en/ff/lv/dataanalytics/uwb-dataset.html}}
}
```

---

## Acknowledgments

- **Fraunhofer IIS**: For providing the publicly accessible open-source UWB channel impulse response localization dataset.
- Developed with PyTorch, NumPy, Pandas, and Matplotlib.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
