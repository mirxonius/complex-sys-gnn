import torch
from torch_geometric.nn import GAT
from torch_geometric.data import Data

from utils.model_utils import make_fcnn, MeanOnGraph, IdentityOnGraph


class GATModel(torch.nn.Module):
    def __init__(
        self,
        gat_params: dict,
        final_output_dim: int = 1,
        head_hidden_layers: int = 3,
        use_normalization_in_head: bool = False,
        aggregate: bool = False,
        num_atom_types: int = 4,
        embedding_size: int = 64,
    ):
        super().__init__()
        self.atom_embedding = torch.nn.Linear(num_atom_types, embedding_size)
        self.gat = GAT(**gat_params)
        self.head = make_fcnn(
            in_shape=gat_params["hidden_channels"],
            hidden_shape=gat_params["hidden_channels"],
            out_shape=final_output_dim,
            num_hidden_layers=head_hidden_layers,
            use_normalization=use_normalization_in_head,
        )
        self.aggregate = MeanOnGraph() if aggregate else IdentityOnGraph()

    def forward(self, graph: Data):
        print("DATA: ", graph)
        src, dst = graph.edge_index
        vec = graph.pos[dst] - graph.pos[src]
        graph.x = self.atom_embedding(graph.z)
        y_pred = self.gat(
            x=graph.x,
            edge_attr=vec,
            edge_index=graph.edge_index,
        )
        y_pred = self.head(y_pred)
        return self.aggregate(y_pred, batch_index=graph.batch)
