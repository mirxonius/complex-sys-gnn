from pathlib import Path
from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models.lightning_model_wrapper import LightningModelWrapper
from utils.setup_utils import set_up_model, set_up_dataset


parser = ArgumentParser()
parser.add_argument("--experiment_name", default="test", type=str)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
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
parser.add_argument(
    "--dataset_index_dir",
    type=str,
    default=None,
    help="If provided the directory should contain three .pt files named train_idx.pt, valid_idx.pt and test_idx.pt",
)

log_dir = Path("./logs/")

if __name__ == "__main__":
    args = parser.parse_args()

    model = set_up_model(args.model, args.model_args_json)
    model = LightningModelWrapper(model=model, lr=args.lr, compile=args.compile)

    train_set, valid_set, test_set = set_up_dataset(
        task=args.task,
        dataset_root_dir=args.data_dir,
        dataset_index_dir=args.dataset_index_dir,
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
    log_dir.mkdir(exist_ok=True)
    logger = TensorBoardLogger(log_dir, name=args.experiment_name)
    trainer = pl.Trainer(logger=logger, max_epochs=args.num_epochs, accelerator="gpu")

    trainer.fit(model, train_loader, valid_loader)

    trainer.test(model, test_loader)
