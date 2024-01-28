import torch
from e3nn import o3
from torch_geometric.data import Data

from .blocks import O3AttentionLayer, NodeEncoder
from utils.model_utils import MeanOnGraph, IdentityOnGraph


class O3GraphAttentionNetwork(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_irreps: str | o3.Irreps,
        output_irreps: str | o3.Irreps,
        hidden_irreps: str | o3.Irreps,
        lmax: int,
        num_basis: int,
        aggregate: bool = False,
        max_radius: float = 2.5,
        num_atom_types: int = 4,
        embedding_size: int = 64,
    ):
        self.max_radius = max_radius
        super().__init__()
        self.aggregate = MeanOnGraph() if aggregate else IdentityOnGraph()
        self.embedding_layer = torch.nn.Linear(num_atom_types, embedding_size)
        layers = []
        for i in range(num_layers):
            layers.append(
                O3AttentionLayer(
                    input_irreps=hidden_irreps if i > 0 else input_irreps,
                    key_irreps=hidden_irreps,
                    query_irreps=hidden_irreps,
                    value_irreps=hidden_irreps if i < num_layers - 1 else output_irreps,
                    lmax=lmax,
                    num_basis=num_basis,
                    max_radius=self.max_radius,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, graph: Data) -> torch.Tensor:
        graph.x = self.embedding_layer(graph)
        for layer in self.layers:
            updated_node_features = layer(graph)
            graph.x = updated_node_features
        return self.aggregate(updated_node_features, batch_index=graph.batch)
