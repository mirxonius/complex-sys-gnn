import torch
from e3nn import o3, nn, math
from e3nn.util.jit import compile_mode
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_scatter import scatter

from utils.model_utils import softmax_on_graph


@compile_mode("script")
class O3EquivConv(torch.nn.Module):
    def __init__(
        self,
        irreps_in: str | o3.Irreps,
    ) -> None:
        super().__init__()

    def forward(self):
        pass


class NodeEncoder(torch.nn.Module):
    def __init__(
        self,
        num_atom_types: int = 4,
        atom_embedding_size: int = 64,
        embedding_irreps: str | o3.Irreps = "32x0e + 32x1o + 32x2e",
        num_basis: int = 32,
        max_radius: float = 2.5,
        lmax: int = 2,
    ):
        super().__init__()
        self.irreps_sph = o3.Irreps.spherical_harmonics(lmax=lmax)
        self.atom_embedding = torch.nn.Linear(num_atom_types, atom_embedding_size)

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=o3.Irreps([(atom_embedding_size, (0, 0))]),
            irreps_in2=self.irreps_sph,
            irreps_out=embedding_irreps,
        )
        self.num_basis = num_basis
        self.max_radius = max_radius
        self.radial_embedding_net = nn.FullyConnectedNet(
            [num_basis, 32, self.tp.weight_numel],
            act=torch.nn.functional.silu,
        )

    def forward(self, graph: Data) -> torch.Tensor:
        graph.x = self.atom_embedding(graph.z)
        src, dst = graph.edge_index
        vec = graph.pos[src] - graph.pos[dst]
        vec_len = vec.norm(dim=1)
        vec_sph = o3.spherical_harmonics(
            self.irreps_sph, vec, normalize=True, normalization="component"
        )
        radial_embedding = math.soft_one_hot_linspace(
            vec_len,
            start=0.0,
            end=self.max_radius,
            number=self.num_basis,
            basis="bessel",
            cutoff=True,
        )
        radial_embedding = self.radial_embedding_net(radial_embedding)
        node_emb = self.tp(graph.x[src], vec_sph, radial_embedding)
        return scatter(src=node_emb, index=dst, dim=0)


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
        num_neighbors: int = 32,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
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
            [self.num_basis, 32, self.tp_value.weight_numel],
            act=torch.nn.functional.silu,
        )

        self.tp_key = o3.FullyConnectedTensorProduct(
            irreps_in1=input_irreps,
            irreps_in2=self.irreps_sph,
            irreps_out=key_irreps,
            shared_weights=False,
        )
        self.key_basis_net = nn.FullyConnectedNet(
            [self.num_basis, 32, self.tp_key.weight_numel],
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
        edge_weight_cutoff = math.soft_unit_step(10 * (1 - vec_len / self.max_radius))

        radial_embedding = radial_embedding.mul(self.num_basis**0.5)
        vec_sph = o3.spherical_harmonics(
            self.irreps_sph, vec, normalize=True, normalization="component"
        )

        query = self.query_projection(graph.x)
        key = self.tp_key(graph.x[src], vec_sph, self.key_basis_net(radial_embedding))
        values = self.tp_value(
            graph.x[src], vec_sph, self.value_basis_net(radial_embedding)
        )
        similarity = self.similarity_tp(query[src], key)
        attn_score = softmax_on_graph(
            input=edge_weight_cutoff.unsqueeze(-1) * similarity,
            index=src,
            dim=0,
        )

        return (
            scatter(
                src=attn_score.relu().sqrt() * values,
                index=dst,
                dim=0,
                dim_size=len(graph.x),
            )
            / self.num_neighbors
        )
