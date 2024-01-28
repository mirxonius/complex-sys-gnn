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
        self.embedding_layer = NodeEncoder(
            num_atom_types=num_atom_types,
            embedding_irreps=input_irreps,
            lmax=lmax,
            atom_embedding_size=embedding_size,
            max_radius=max_radius,
        )
        layers = []
        for i in range(num_layers):
            layers.append(
                O3AttentionLayer(
                    input_irreps=hidden_irreps,
                    key_irreps=hidden_irreps,
                    query_irreps=hidden_irreps,
                    value_irreps=hidden_irreps,
                    lmax=lmax,
                    num_basis=num_basis,
                    max_radius=self.max_radius,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.decoder = o3.Linear(irreps_in=hidden_irreps, irreps_out=output_irreps)

    def forward(self, graph: Data) -> torch.Tensor:
        graph.x = self.embedding_layer(graph)
        for layer in self.layers:
            updated_node_features = layer(graph)
            graph.x += updated_node_features
        graph.x = self.decoder(graph.x)
        return self.aggregate(updated_node_features, batch_index=graph.batch)
