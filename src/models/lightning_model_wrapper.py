from typing import Iterable

import torch
from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl
from torch_geometric.data import Data


from config_defaults import SupportedModels
from utils.metric_calc import RegressionMetricCalc

from models.gat_model import GATModel
from models.gate_equiv_model import GateEquivariantModel
from models.mace_model import MaceModel

model_dict = {
    SupportedModels.mace_model.value: MaceModel,
    SupportedModels.gat_model.value: GATModel,
    SupportedModels.gate_equiv_model.value: GateEquivariantModel,
}


# TODO: add comments and test code
# NOTE: maybe change scheduler to one_cycle_lr
class LightningModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_params: dict,
        lr: float = 1e-2,
        compile: bool = False,
        target_idx: int | list[int] = 0,
    ):
        super().__init__()
        try:
            self.model = model_dict[model_name](**model_params)
        except:
            print(f"{model_name} is not a supported model.")

        if compile:
            self.model = torch.compile(self.model)

        if isinstance(target_idx, Iterable):
            self.target_idx = torch.LongTensor(target_idx)
        else:
            self.target_idx = torch.LongTensor([target_idx])

        self.metric_calculator = RegressionMetricCalc(num_outputs=1)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, graph: Data):
        return self.model(graph)

    def _step(self, graph, step_type="train"):
        prediction = self.forward(graph)
        loss = self.loss_fn(prediction, graph.y[:, self.target_idx])
        metric_dict = self.metric_calculator(
            prediction, graph.y[:, self.target_idx], step_type
        )
        metric_dict.update({step_type + "_MSE_loss": loss})
        self.log_dict(
            metric_dict, on_epoch=True, on_step=True, batch_size=graph.batch.max()
        )
        return loss

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