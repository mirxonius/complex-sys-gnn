"""
Training script for equivariant Graph Neural Networks on molecular dynamics.

This script provides a command-line interface for training and testing GNN models
on the MD17 molecular dynamics dataset. Supports both equivariant (O3, MACE) and
standard (GAT) architectures.
"""

from pathlib import Path
import os
from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models.lightning_model_wrapper import LightningModelWrapper
from utils.setup_utils import set_up_model, set_up_dataset, set_up_metric, set_up_loss

# Command-line argument parser
parser = ArgumentParser()
parser.add_argument("--experiment_name", default="extra_small", type=str,
                    help="Name of the experiment for logging")
parser.add_argument("--num_epochs", type=int, default=20,
                    help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-2,
                    help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size for training")
parser.add_argument("--radius", type=float, default=None,
                    help="Radius for molecular graph construction (Angstroms)")
parser.add_argument("--task", type=str,
                    help="Task to perform (e.g., multi_molecule_forces)")
parser.add_argument("--compile", type=bool, default=False,
                    help="Whether to use torch.compile for model optimization")
parser.add_argument("--model", type=str,
                    help="Model name (must be defined in model_args_json)")
parser.add_argument("--model_args_json", type=str,
                    help="Path to JSON file containing model parameters")
parser.add_argument("--data_dir", type=str,
                    default="/home/fmirkovic/user_data/fmirkovic/diplomski_datasets/molecules/md17",
                    help="Directory containing MD17 dataset")
parser.add_argument("--loss", type=str, default="mse",
                    help="Loss function (mse, mae, mse_mae, huber, huber_scalar)")
parser.add_argument("--extrapolate", type=bool, default=False,
                    help="Test on paracetamol for extrapolation evaluation")
parser.add_argument("--training_noise", type=bool, default=False,
                    help="Add Gaussian noise during training")
parser.add_argument("--extra_small", type=bool, default=False,
                    help="Use reduced dataset for quick testing")
log_dir = Path("./logs/")

if __name__ == "__main__":
    args = parser.parse_args()

    # Initialize model from JSON configuration
    model = set_up_model(args.model, args.model_args_json)
    metric_calc_kwargs = set_up_metric(args.task)
    loss = set_up_loss(args.loss)

    # Wrap model in PyTorch Lightning module
    model = LightningModelWrapper(
        model=model, lr=args.lr, compile=args.compile, loss_fn=loss, **metric_calc_kwargs
    )
    print("MODEL IS READY")

    # Load datasets
    train_set, valid_set, test_set = set_up_dataset(
        task=args.task, dataset_data_dir=args.data_dir, extra_small=args.extra_small
    )

    # Optionally load paracetamol dataset for extrapolation testing
    if args.extrapolate:
        _, __, paracetamol_test_set = set_up_dataset(
            task="paracetamol", dataset_data_dir=args.data_dir, training_noise=args.training_noise
        )
        paracetamol_loader = DataLoader(paracetamol_test_set, batch_size=args.batch_size)

    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=4
    )
    print("DATASET LEN:", len(train_set))

    # Setup logging directory
    log_dir = Path(log_dir, args.task)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize logger and trainer
    logger = TensorBoardLogger(log_dir, name=args.model + f"/{args.experiment_name}")
    trainer = pl.Trainer(
        logger=logger, max_epochs=args.num_epochs, accelerator="gpu", devices=[0]
    )

    # Train and test
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

    # Extrapolation test if requested
    if args.extrapolate:
        trainer.test(model, paracetamol_loader)


