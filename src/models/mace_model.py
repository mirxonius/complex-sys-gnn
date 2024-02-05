import torch
from e3nn import o3
from torch_geometric.data import Data
from mace_layer import MACE_layer

from utils.model_utils import MeanOnGraph


class MaceNet(torch.nn.Module):
    """
    Implementation of a two layer MACE GNN.
    """

    def __init__(
        self,
        mace_params: dict,
        num_atom_types: int = 4,
        final_prediction_irreps: str | o3.Irreps = "1x0e",
        aggregate: bool = True,
    ):
        super().__init__()

        self.embedding_layer = torch.nn.Linear(num_atom_types, mace_params["n_dims_in"])

        second_layer_params = {**mace_params}
        second_layer_params["n_dims_in"] = o3.Irreps(mace_params["hidden_irreps"]).dim
        self.learable_node_features1 = torch.nn.Linear(
            in_features=mace_params["n_dims_in"],
            out_features=o3.Irreps(mace_params["node_feats_irreps"]).dim,
        )
        self.message_passing1 = MACE_layer(**mace_params)

        self.learable_node_features2 = torch.nn.Linear(
            in_features=second_layer_params["n_dims_in"],
            out_features=o3.Irreps(second_layer_params["node_feats_irreps"]).dim,
        )
        self.message_passing2 = MACE_layer(**second_layer_params)
        if aggregate:
            self.aggregate = MeanOnGraph(dim=0, keepdim=True)
        else:
            self.aggregate = torch.nn.Identity()

        self.projection = o3.FullyConnectedTensorProduct(
            irreps_in1=o3.Irreps(second_layer_params["hidden_irreps"]),
            irreps_in2=o3.Irreps(second_layer_params["hidden_irreps"]),
            irreps_out=final_prediction_irreps,
        )

    def forward(self, graph: Data):
        feats = self.learable_node_features1(graph.x)
        src, dst = graph.edge_index
        vec = graph.pos[dst] - graph.pos[src]
        updated_nodes = self.message_passing1(
            vectors=vec,
            node_attrs=graph.x,
            node_feats=feats,
            edge_feats=graph.edge_attr,
            edge_index=graph.edge_index,
        )
        feats = self.learable_node_features2(updated_nodes)
        updated_nodes = self.message_passing2(
            vectors=vec,
            node_attrs=updated_nodes,
            node_feats=feats,
            edge_feats=graph.edge_attr,
            edge_index=graph.edge_index,
        )
        updated_nodes = self.projection(updated_nodes, updated_nodes)
        return self.aggregate(updated_nodes, batch_index=graph.batch)
