from pathlib import Path
from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


parser = ArgumentParser()
parser.add_argument("--experiment_name", default="MACE", type=str)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--radius", type=float, default=None)


log_dir = Path("..logs/")

if __name__ == "__main__":
    args = parser.parse_args()

    model = NotImplemented

    train_set = NotImplemented
    val_set = NotImplemented
    test_set = NotImplemented

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=4,
    )

    logger = TensorBoardLogger(log_dir, name=args.experiment_name)
    trainer = pl.Trainer(logger=logger, max_epochs=args.num_epochs, accelerator="gpu")

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)
