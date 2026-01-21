"""
Building blocks for O(3)-equivariant graph neural networks.

Contains node encoders and attention layers using e3nn.
"""

import torch
from e3nn import o3, nn, math
from e3nn.util.jit import compile_mode
from torch_scatter import scatter
from torch_geometric.data import Data

from utils.model_utils import softmax_on_graph


@compile_mode("script")
class O3EquivConv(torch.nn.Module):
    """Placeholder for O(3)-equivariant convolution layer (not yet implemented)."""

    def __init__(self, irreps_in: str | o3.Irreps) -> None:
        super().__init__()

    def forward(self):
        pass


class NodeEncoder(torch.nn.Module):
    """
    Encodes nodes using spherical harmonics and radial basis functions.

    Combines atomic type embeddings with geometric information from edge
    vectors to create rotationally equivariant node features.
    """
    def __init__(
        self,
        num_atom_types: int = 4,
        atom_embedding_size: int = 64,
        embedding_irreps: str | o3.Irreps = "32x0e + 32x1o + 32x2e",
        lmax: int = 2,
        num_basis: int = 32,
        max_radius: float = 2.0,
    ):
        """
        Initialize node encoder.

        Args:
            num_atom_types: Number of atom types (H, C, N, O = 4)
            atom_embedding_size: Dimension of atom type embeddings
            embedding_irreps: Output irreducible representations
            lmax: Maximum degree of spherical harmonics
            num_basis: Number of radial basis functions
            max_radius: Maximum interaction radius
        """
        super().__init__()
        self.max_radius = max_radius
        self.irreps_sph = o3.Irreps.spherical_harmonics(lmax=lmax)

        # Project spherical harmonics to desired irreps
        self.spherical_projection = o3.Linear(
            irreps_in=self.irreps_sph, irreps_out=embedding_irreps
        )

        # Atom type embedding
        self.atom_embedding = torch.nn.Linear(
            num_atom_types, self.spherical_projection.irreps_out.dim
        )

        # Radial basis network
        self.num_basis = num_basis
        self.radial_embedding_net = nn.FullyConnectedNet(
            [self.num_basis, 32, self.spherical_projection.irreps_out.dim],
            act=torch.nn.functional.silu,
        )

    def forward(self, graph: Data) -> Data:
        """
        Encode nodes using edge geometry and atom types.

        Args:
            graph: Molecular graph with positions, edge_index, and atom types (z)

        Returns:
            Node embeddings with shape [num_nodes, embedding_dim]
        """
        src, dst = graph.edge_index
        vec = graph.pos[src] - graph.pos[dst]  # Edge vectors
        vec_len = vec.norm(dim=1)

        # Compute spherical harmonics from edge directions
        vec_sph = o3.spherical_harmonics(
            self.irreps_sph, vec, normalize=True, normalization="component"
        )
        spherical_embedding = self.spherical_projection(vec_sph)

        # Radial basis functions (smooth cutoff at max_radius)
        radial_embedding = math.soft_one_hot_linspace(
            vec_len,
            start=0.0,
            end=self.max_radius,
            number=self.num_basis,
            basis="bessel",
            cutoff=True,
        )

        # Combine radial, spherical, and atom type information
        atom_embedding = self.atom_embedding(graph.z)[src]
        radial_embedding = self.radial_embedding_net(radial_embedding)
        node_emb = radial_embedding * spherical_embedding * atom_embedding

        # Aggregate edge features to nodes
        return scatter(src=node_emb, index=dst, dim=0)


@compile_mode("script")
class O3AttentionLayer(torch.nn.Module):
    """
    O(3)-equivariant attention layer using tensor products.

    Implements attention mechanism while preserving rotational equivariance
    through e3nn tensor products and spherical harmonics.
    """
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
        """
        Initialize O3 attention layer.

        Args:
            input_irreps: Input irreducible representations
            key_irreps: Key irreps for attention
            query_irreps: Query irreps for attention
            value_irreps: Value irreps for message passing
            lmax: Maximum degree of spherical harmonics
            num_basis: Number of radial basis functions
            max_radius: Maximum interaction radius
            num_neighbors: Typical number of neighbors (for normalization)
        """
        super().__init__()
        self.num_neighbors = num_neighbors
        self.num_basis = num_basis
        self.max_radius = max_radius
        self.irreps_sph = o3.Irreps.spherical_harmonics(lmax=lmax)

        # Value tensor product (node features × spherical harmonics → values)
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

        # Key tensor product
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

        # Similarity metric between keys and queries (→ scalar)
        self.similarity_tp = o3.FullyConnectedTensorProduct(
            irreps_in1=query_irreps, irreps_in2=key_irreps, irreps_out="0e"
        )

        # Query projection
        self.query_projection = o3.Linear(
            irreps_in=input_irreps, irreps_out=query_irreps
        )

    def forward(self, graph: Data):
        """
        Compute equivariant attention and aggregate messages.

        Args:
            graph: Molecular graph with node features (x), positions (pos), and edges

        Returns:
            Updated node features after attention aggregation
        """
        src, dst = graph.edge_index
        vec = graph.pos[src] - graph.pos[dst]
        vec_len = vec.norm(dim=1)

        # Radial basis with smooth cutoff
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

        # Spherical harmonics from edge directions
        vec_sph = o3.spherical_harmonics(
            self.irreps_sph, vec, normalize=True, normalization="component"
        )

        # Compute queries, keys, and values
        query = self.query_projection(graph.x)
        key = self.tp_key(graph.x[src], vec_sph, self.key_basis_net(radial_embedding))
        values = self.tp_value(
            graph.x[src], vec_sph, self.value_basis_net(radial_embedding)
        )

        # Attention scores (with distance-based cutoff)
        similarity = self.similarity_tp(query[src], key)
        attn_score = softmax_on_graph(
            input=edge_weight_cutoff.unsqueeze(-1) * similarity,
            index=src,
            dim=0,
        )

        # Aggregate weighted values to destination nodes
        return (
            scatter(
                src=attn_score.relu().sqrt() * values,
                index=dst,
                dim=0,
                dim_size=len(graph.x),
            )
            / self.num_neighbors
        )
