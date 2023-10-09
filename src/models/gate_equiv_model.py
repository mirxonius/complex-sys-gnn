import torch
from e3nn import nn, o3, math


from utils.model_utils import MeanOnGraph


class GateEquivariantModel(torch.nn.Module):
    def __init__(
        self, final_prediction_irreps: str | o3.Irreps = "1x0e", aggregate: bool = True
    ):
        pass

    def forward(self, graph):
        pass
