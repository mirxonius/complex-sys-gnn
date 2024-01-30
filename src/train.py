from pathlib import Path
import os
from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

from models.lightning_model_wrapper import LightningModelWrapper
from utils.setup_utils import set_up_model, set_up_dataset, set_up_metric, set_up_loss


parser = ArgumentParser()
parser.add_argument("--experiment_name", default="equivariant_gat", type=str)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--radius", type=float, default=None)
parser.add_argument("--task", type=str)
parser.add_argument("--compile", type=bool, default=False)
parser.add_argument("--model", type=str)
parser.add_argument(
    "--model_args_json",
    type=str,
    help=".json file containing model parameters, since they can be very long.",
)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--loss", type=str, default="mse")
parser.add_argument("--extrapolate",type=bool,default=False)
log_dir = Path("./logs/")

if __name__ == "__main__":
    args = parser.parse_args()

    model = set_up_model(args.model, args.model_args_json)
    metric_calc_kwargs = set_up_metric(args.task)
    loss = set_up_loss(args.loss)
    model = LightningModelWrapper(
        model=model, lr=args.lr, compile=args.compile, **metric_calc_kwargs
    )
    print("MODEL IS READY")
    train_set, valid_set, test_set = set_up_dataset(
        task=args.task,
        dataset_data_dir=args.data_dir,
    )
    if args.extrapolate:
        _, __, test_set = set_up_dataset(
            task="paracetamol", dataset_data_dir=args.data_dir
        )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=4,
    )

    log_dir = Path(log_dir, args.task)
    os.makedirs(log_dir, exist_ok=True)

    # profiler = "simple
    logger = TensorBoardLogger(log_dir, name=args.model)
    trainer = pl.Trainer(
        logger=logger, max_epochs=args.num_epochs, accelerator="gpu", devices=[0]
    )
    trainer.fit(model, train_loader, valid_loader)

    trainer.test(model, test_loader)



