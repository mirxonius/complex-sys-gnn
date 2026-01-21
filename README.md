# Equivariant Graph Neural Networks for Molecular Dynamics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Master's thesis project in Physics: A modular API for training and testing equivariant and regular graph neural networks on physical systems, with current focus on molecular dynamics.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Models](#models)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Overview

This repository implements **SE(3)-equivariant Graph Neural Networks** for learning molecular dynamics from the MD17 dataset. The models preserve rotational and translational symmetries, making them particularly suitable for physical systems.

### What is Equivariance?

Equivariance means that when the input is transformed (e.g., rotated), the output transforms in a predictable way:

```
f(R·x) = R·f(x)    (for SE(3) equivariant models)
```

This is crucial for molecular systems where predictions should be independent of the coordinate system orientation.

### Scientific Context

Molecular dynamics simulations are fundamental to understanding:
- Drug discovery and protein folding
- Materials science and catalyst design
- Chemical reaction mechanisms
- Quantum chemistry

Traditional simulation methods (DFT, MD) are computationally expensive. Machine learning models can accelerate these simulations by orders of magnitude while maintaining accuracy.

---

## Key Features

✅ **SE(3)-Equivariant Architectures**: O(3) Graph Attention Networks and MACE models <br>
✅ **Modular Design**: Easy to extend to new tasks and datasets <br>
✅ **Multiple Loss Functions**: MSE, MAE, Huber, and custom combinations <br>
✅ **PyTorch Lightning**: Organized training with automatic logging <br>
✅ **Extrapolation Testing**: Evaluate generalization to unseen molecules <br>
✅ **TensorBoard Integration**: Real-time training visualization <br>
✅ **Comprehensive Metrics**: R², RMSE, MAE for robust evaluation <br>

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  MD17 Dataset                                                │
│  ├─ Benzene (C₆H₆)           ~600k conformations           │
│  ├─ Ethanol (C₂H₅OH)         ~600k conformations           │
│  ├─ Uracil (C₄H₄N₂O₂)        ~600k conformations           │
│  ├─ Aspirin (C₉H₈O₄)         ~600k conformations           │
│  └─ Paracetamol (C₈H₉NO₂)    ~600k conformations (test)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Graph Construction                                          │
│  ├─ Radius-based edges (default: 1.875 Å)                  │
│  ├─ One-hot atom types (H, C, N, O)                        │
│  └─ 3D positions + atomic forces                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Architecture                                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ O(3) Graph Attention Network                          │  │
│  │ ├─ Node Encoder (Spherical Harmonics + Radial Basis) │  │
│  │ ├─ Attention Layers (Tensor Products)                │  │
│  │ └─ Decoder (Output Irreps)                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                         OR                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ MACE Network                                          │  │
│  │ ├─ Multi-body Interactions                           │  │
│  │ └─ Message Passing                                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Predictions                                                 │
│  ├─ Atomic Forces (3D vectors per atom)                    │
│  └─ Molecular Energy (scalar per molecule)                 │
└─────────────────────────────────────────────────────────────┘
```

### O(3) Graph Attention Network Architecture

The core model uses **irreducible representations** (irreps) from e3nn to maintain equivariance:

```
Input: Molecular Graph
│
├─ Node Features: One-hot atom types [H, C, N, O]
├─ Edge Features: Spatial vectors rᵢⱼ = xᵢ - xⱼ
└─ Positions: 3D coordinates
        │
        ▼
┌──────────────────────────┐
│  Node Encoder            │
│  ├─ Spherical Harmonics  │  Y^l_m(r̂ᵢⱼ)
│  ├─ Radial Basis         │  ϕₖ(||rᵢⱼ||)
│  └─ Tensor Product       │  ⊗
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  O(3) Attention Layers   │  (×N layers)
│  ┌────────────────────┐  │
│  │ Query Projection   │  │
│  │ Key TP             │  │  Q·K^T → attention scores
│  │ Value TP           │  │
│  │ Softmax + Cutoff   │  │
│  └────────────────────┘  │
│  + Residual Connection   │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Decoder                 │
│  Linear(irreps → output) │
│  • Forces: "3x1o"        │  (3D vectors)
│  • Energy: "1x0e"        │  (scalar)
└──────────────────────────┘
```

**Irreducible Representations Used:**
- `0e`: Scalars (rotation-invariant)
- `1o`: Vectors (rotate like positions)
- `2e`: Rank-2 even tensors
- `lmax`: Maximum spherical harmonic degree (typically 2-4)

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/mirxonius/complex-sys-gnn.git
cd complex-sys-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # For GPU
# OR
pip install -r requirements_cpu.txt  # For CPU only

# Verify installation
python -c "import torch; import e3nn; print('Setup successful!')"
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0.1 | Deep learning framework |
| PyTorch Geometric | 2.1 | Graph neural networks |
| e3nn | 0.4.4 | E(3)-equivariant operations |
| PyTorch Lightning | ≥2.0.3 | Training framework |
| mace-layer | latest | MACE model implementation |
| tensorboard | ≥2.15.1 | Training visualization |

---

## Quick Start

### 1. Download Dataset

The MD17 dataset will be automatically downloaded on first run. Alternatively, prepare your data directory:

```bash
mkdir -p data/md17
# Place multimolecule_index.json and paracetamol_index.json in data/md17/
```

### 2. Train a Model

```bash
# Train O(3) Transformer on multi-molecule force prediction
python src/train.py \
  --task multi_molecule_forces \
  --model o3_transformer \
  --model_args_json model_args/md17/model_args.json \
  --experiment_name my_first_experiment \
  --num_epochs 20 \
  --batch_size 128 \
  --lr 1e-2 \
  --data_dir data/md17
```

### 3. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### 4. Quick Test with Small Dataset

```bash
python src/train.py \
  --task multi_molecule_forces \
  --model o3_transformer \
  --model_args_json model_args/md17/model_args.json \
  --extra_small True \
  --num_epochs 5
```

---

## Dataset

### MD17: Molecular Dynamics Benchmark

The **revised MD17 dataset** contains ab-initio molecular dynamics trajectories computed with DFT (PBE functional):

| Molecule | Formula | Atoms | Configurations | Training | Validation | Test |
|----------|---------|-------|----------------|----------|------------|------|
| Benzene | C₆H₆ | 12 | ~600k | 450k | 50k | 100k |
| Ethanol | C₂H₅OH | 9 | ~600k | 450k | 50k | 100k |
| Uracil | C₄H₄N₂O₂ | 12 | ~600k | 450k | 50k | 100k |
| Aspirin | C₉H₈O₄ | 21 | ~600k | 450k | 50k | 100k |
| Paracetamol* | C₈H₉NO₂ | 20 | ~600k | - | - | 100k |

\* Used exclusively for extrapolation testing

### Data Format

Each sample contains:
- **Positions**: `[num_atoms, 3]` Cartesian coordinates (Å)
- **Atomic Numbers**: `[num_atoms]` Element identifiers
- **Forces**: `[num_atoms, 3]` Atomic forces (kcal/mol/Å)
- **Energy**: Scalar molecular energy (kcal/mol)

### Graph Construction

Molecular graphs are constructed using:
- **Nodes**: Atoms with one-hot encoded types
- **Edges**: Radius-based connectivity (default 1.875 Å)
- **Edge Features**: Spatial vectors between atoms

---

## Models

### 1. O(3) Graph Attention Network

**Architecture**: SE(3)-equivariant attention with spherical harmonics

**Key Parameters**:
```json
{
  "num_layers": 4,
  "hidden_irreps": "32x0e + 32x1o + 32x2e",
  "output_irreps": "1x1o",
  "lmax": 2,
  "num_basis": 32,
  "max_radius": 2.5
}
```

**Best For**: Small to medium molecules, interpretability

### 2. MACE Network

**Architecture**: Multi-Atomic Cluster Expansion with many-body interactions

**Key Parameters**:
```json
{
  "num_layers": 2,
  "hidden_irreps": "128x0e + 128x1o",
  "correlation": 3,
  "max_radius": 2.5
}
```

**Best For**: Large molecules, high accuracy

### 3. Standard GAT (Baseline)

**Architecture**: Non-equivariant Graph Attention Network

**Purpose**: Baseline comparison to demonstrate equivariance benefits

---

## Training

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | required | Task name (e.g., multi_molecule_forces) |
| `--model` | str | required | Model name from JSON config |
| `--model_args_json` | str | required | Path to model config JSON |
| `--experiment_name` | str | "extra_small" | Experiment identifier |
| `--num_epochs` | int | 20 | Training epochs |
| `--lr` | float | 1e-2 | Learning rate |
| `--batch_size` | int | 128 | Batch size |
| `--loss` | str | "mse" | Loss function (mse/mae/mse_mae/huber) |
| `--data_dir` | str | ... | MD17 data directory |
| `--extrapolate` | bool | False | Test on paracetamol |
| `--training_noise` | bool | False | Add Gaussian noise |
| `--extra_small` | bool | False | Use 400 samples for testing |
| `--compile` | bool | False | Use torch.compile |

### Example Training Commands

#### Multi-Molecule Force Prediction
```bash
python src/train.py \
  --task multi_molecule_forces \
  --model o3_transformer \
  --model_args_json model_args/md17/model_args.json \
  --experiment_name force_prediction \
  --num_epochs 50 \
  --batch_size 64 \
  --lr 1e-2 \
  --loss mse_mae
```

#### Single Molecule Training
```bash
python src/train.py \
  --task benzene_forces \
  --model mace_model \
  --model_args_json model_args/md17/model_args.json \
  --experiment_name benzene_mace \
  --num_epochs 100
```

#### Extrapolation Testing
```bash
python src/train.py \
  --task multi_molecule_forces \
  --model o3_transformer \
  --model_args_json model_args/md17/model_args.json \
  --extrapolate True \
  --experiment_name extrapolation_test
```

### Training Tips

1. **Learning Rate**: Start with 1e-2, decrease if loss plateaus
2. **Batch Size**: Larger batches (128-256) often improve stability
3. **Model Size**: Increase `hidden_irreps` for better accuracy (e.g., `64x0e + 64x1o + 64x2e`)
4. **lmax**: Higher values (3-4) capture finer angular features but increase compute
5. **Regularization**: Weight decay 1e-3 is applied by default

---

## Results

### Performance Metrics

Models are evaluated using:
- **MAE**: Mean Absolute Error (kcal/mol/Å for forces)
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination


### Visualization

Training progress can be monitored in TensorBoard:

```bash
tensorboard --logdir logs/
```

**Available Plots**:
- Loss curves (train/validation/test)
- R² score over epochs
- MAE and RMSE metrics
- Learning rate schedule

---

## Project Structure

```
complex-sys-gnn/
├── src/
│   ├── train.py                    # Main training script
│   ├── config_defaults.py          # Task and model enums
│   ├── models/
│   │   ├── lightning_model_wrapper.py  # PyTorch Lightning wrapper
│   │   ├── equivariant_gat.py         # O(3) GAT implementation
│   │   ├── blocks.py                   # Equivariant building blocks
│   │   ├── mace_model.py              # MACE network
│   │   └── gat_model.py               # Standard GAT baseline
│   ├── datasets/
│   │   └── Md17_dataset.py            # MD17 data loaders
│   └── utils/
│       ├── setup_utils.py             # Model/dataset factories
│       ├── metric_calc.py             # Regression metrics
│       ├── model_utils.py             # Graph operations
│       ├── loss_utils.py              # Custom loss functions
│       └── equivariance_utils.py      # Equivariance testing
├── model_args/
│   └── md17/
│       ├── model_args.json            # All model configs
│       ├── o3_transformer_args.json   # O3 GAT configs
│       ├── mace_model_args.json       # MACE configs
│       └── regular_gat_args.json      # Baseline configs
├── data/
│   └── md17/                          # Dataset storage
│       ├── multimolecule_index.json   # Train/val/test splits
│       └── paracetamol_index.json     # Extrapolation splits
├── logs/                              # TensorBoard logs
├── requirements.txt                   # GPU dependencies
├── requirements_cpu.txt               # CPU dependencies
└── README.md                          # This file
```

---

## Advanced Usage

### Custom Model Configuration

Create a new JSON entry in `model_args/md17/model_args.json`:

```json
{
  "my_custom_model": {
    "model_type": "equivariant_gat",
    "model_args": {
      "num_layers": 6,
      "hidden_irreps": "64x0e + 64x1o + 64x2e + 64x3o",
      "output_irreps": "1x1o",
      "lmax": 3,
      "num_basis": 64,
      "max_radius": 3.0
    }
  }
}
```

Then train with:
```bash
python src/train.py --model my_custom_model ...
```

### Equivariance Verification

Test model equivariance:

```python
from utils.equivariance_utils import test_equivariance
from models.equivariant_gat import O3GraphAttentionNetwork

model = O3GraphAttentionNetwork(...)
test_equivariance(model, num_samples=100)
```

### Data Augmentation

Add Gaussian noise during training:

```bash
python src/train.py --training_noise True ...
```

---

## Future Work

🔮 **Planned Extensions**:
- [ ] Smooth Particle Hydrodynamics (SPH) systems
- [ ] Mesh-based simulations
- [ ] Electron density prediction
- [ ] Protein structure prediction
- [ ] Materials property prediction
- [ ] Multi-task learning framework

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{mirkovic2024equivariant,
  title={Deep Learning Methods For Modeling Complex Systems},
  author={Mirkovic, Filip},
  year={2024},
  school={[Faculty of Science, University of Zagreb]},
  type={Master's Thesis}
}
```

### Related Papers

- **e3nn**: [e3nn: Euclidean Neural Networks](https://arxiv.org/abs/2207.09453)
- **MACE**: [MACE: Higher Order Equivariant Message Passing Neural Networks](https://arxiv.org/abs/2206.07697)
- **MD17**: [Machine Learning of Accurate Energy-conserving Molecular Force Fields](https://arxiv.org/abs/1611.04678)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **e3nn** library by Mario Geiger and Tess Smidt
- **MACE** implementation by Ilyes Batatia et al.
- **MD17 dataset** by Stefan Chmiela et al.
- **PyTorch Geometric** team

---

## Contact

For questions or collaborations:
- GitHub Issues: [complex-sys-gnn/issues](https://github.com/mirxonius/complex-sys-gnn/issues)
---

**Last Updated**: January 2026
