from typing import Iterable
from collections import defaultdict
import torch
from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl
from torch_geometric.data import Data


from utils.metric_calc import RegressionMetricCalc


# TODO: add comments and test code
# NOTE: maybe change scheduler to one_cycle_lr
class LightningModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-2,
        compile: bool = False,
        target_idx: int | list[int] = 0,
    ):
        super().__init__()

        self.model = model

        if compile:
            self.model = torch.compile(self.model)

        if isinstance(target_idx, Iterable):
            self.target_idx = torch.LongTensor(target_idx)
        else:
            self.target_idx = torch.LongTensor([target_idx])

        self.metric_calculator = RegressionMetricCalc(num_outputs=1)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr
        self.step_outputs = defaultdict(
            lambda: {"y_pred": [], "y_true": [], "loss": []}
        )
        # self.save_hyperparameters()

    def forward(self, graph: Data):
        return self.model(graph)

    def _step(self, graph, step_type="train"):
        prediction = self.forward(graph)
        loss = self.loss_fn(prediction, graph.y[:, self.target_idx])
        self.step_outputs[step_type]["y_pred"].append(prediction)
        self.step_outputs[step_type]["y_true"].append(graph.y[:, self.target_idx])
        self.step_outputs[step_type]["loss"].append(loss)
        return loss

    def on_epoch_end(self, epoch_type):
        avg_loss = torch.stack(
            [x for x in self.step_outputs[epoch_type]["loss"]]
        ).mean()
        y_pred = torch.cat([x for x in self.step_outputs[epoch_type]["y_pred"]])
        y_true = torch.cat([x for x in self.step_outputs[epoch_type]["y_true"]])
        for k in self.step_outputs[epoch_type].keys():
            self.step_outputs[epoch_type][k].clear()
        metric_dict = self.metric_calculator(y_pred, y_true, epoch_type)
        metric_dict.update({f"{epoch_type}_MSE_loss": avg_loss})
        self.log_dict(metric_dict, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end("valid")

    def on_test_epoch_end(self) -> None:
        self.on_epoch_end("test")

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._step(batch, "valid")

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1 - 1e-3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
