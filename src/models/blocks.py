import torch
from e3nn import o3, nn, math, io
from e3nn.util.jit import compile_mode
from torch_scatter import scatter
from torch_geometric.data import Data

from utils.model_utils import softmax_on_graph


@compile_mode("script")
class O3EquivConv(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self):
        pass


@compile_mode("script")
class O3AttentionLayer(torch.nn.Module):
    def __init__(
        self,
        input_irreps: str | o3.Irreps,
        key_irreps: str | o3.Irreps,
        query_irreps: str | o3.Irreps,
        value_irreps: str | o3.Irreps,
        lmax: int = 2,
        num_basis: int = 32,
        max_radius: float = 2.5,
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.max_radius = max_radius
        self.irreps_sph = o3.Irreps.spherical_harmonics(lmax=lmax)
        self.tp_value = o3.FullyConnectedTensorProduct(
            irreps_in1=input_irreps,
            irreps_in2=self.irreps_sph,
            irreps_out=value_irreps,
            shared_weights=False,
        )
        self.value_basis_net = nn.FullyConnectedNet(
            [self.num_basis, 16, self.tp_value.weight_numel],
            act=torch.nn.functional.silu,
        )

        self.tp_key = o3.FullyConnectedTensorProduct(
            irreps_in1=input_irreps,
            irreps_in2=self.irreps_sph,
            irreps_out=key_irreps,
            shared_weights=False,
        )
        self.key_basis_net = nn.FullyConnectedNet(
            [self.num_basis, 16, self.tp_key.weight_numel],
            act=torch.nn.functional.silu,
        )

        # Calculates similarity metric between keys and queries
        self.similarity_tp = o3.FullyConnectedTensorProduct(
            irreps_in1=query_irreps, irreps_in2=key_irreps, irreps_out="0e"
        )

        self.query_projection = o3.Linear(
            irreps_in=input_irreps, irreps_out=query_irreps
        )

    def forward(self, graph: Data):
        src, dst = graph.edge_index
        vec = graph.pos[src] - graph.pos[dst]
        vec_len = vec.norm(dim=1)

        radial_embedding = math.soft_one_hot_linspace(
            vec_len,
            start=0.0,
            end=self.max_radius,
            number=self.num_basis,
            basis="bessel",
            cutoff=True,
        )
        # edge_weight_cutoff = math.soft_unit_step(10 * (1 - vec_len / self.max_radius))

        radial_embedding = radial_embedding.mul(self.num_basis**0.5)
        vec_sph = o3.spherical_harmonics(
            self.irreps_sph, vec, normalize=True, normalization="component"
        )

        query = self.query_projection(graph.x)
        key = self.tp_key(graph.x[src], vec_sph, self.key_basis_net(radial_embedding))
        values = self.tp_value(
            graph.x[src], vec_sph, self.value_basis_net(radial_embedding)
        )

        attn_score = softmax_on_graph(
            input=self.similarity_tp(query[dst], key), index=dst, dim=0
        )
        return scatter(
            src=attn_score.relu().sqrt() * values,
            index=dst,
            dim=0,
            dim_size=len(graph.x),
        )
