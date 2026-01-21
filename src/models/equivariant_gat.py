"""
O(3)-equivariant Graph Attention Network using e3nn.

Implements SE(3)-equivariant attention for molecular property prediction.
"""

import torch
from e3nn import o3
from torch_geometric.data import Data

from .blocks import O3AttentionLayer, NodeEncoder
from utils.model_utils import MeanOnGraph, IdentityOnGraph


class O3GraphAttentionNetwork(torch.nn.Module):
    """
    SE(3)-equivariant Graph Attention Network.

    Uses spherical harmonics and tensor products to maintain rotational
    and translational equivariance for molecular systems.
    """
    def __init__(
        self,
        num_layers: int,
        output_irreps: str | o3.Irreps,
        hidden_irreps: str | o3.Irreps,
        lmax: int,
        num_basis: int,
        aggregate: bool = False,
        max_radius: float = 2.5,
        num_atom_types: int = 4,
        embedding_size: int = 64,
    ):
        """
        Initialize O3 Graph Attention Network.

        Args:
            num_layers: Number of attention layers
            output_irreps: Output irreducible representations (e.g., '1x1o' for 3D vectors)
            hidden_irreps: Hidden layer irreps (e.g., '32x0e + 32x1o + 32x2e')
            lmax: Maximum spherical harmonic degree
            num_basis: Number of radial basis functions
            aggregate: If True, aggregate node features to graph level
            max_radius: Maximum interaction radius in Angstroms
            num_atom_types: Number of distinct atom types
            embedding_size: Dimension of initial atom embeddings
        """
        self.max_radius = max_radius
        super().__init__()

        # Aggregation for graph-level vs node-level predictions
        self.aggregate = MeanOnGraph() if aggregate else IdentityOnGraph()

        # Initial embedding and encoding layers
        self.embedding_layer = torch.nn.Linear(num_atom_types, embedding_size)
        self.node_encoder = NodeEncoder(
            embedding_irreps=hidden_irreps,
            num_basis=num_basis,
            max_radius=self.max_radius,
        )

        # Stack of equivariant attention layers
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

        # Output projection
        self.decoder = o3.Linear(irreps_in=hidden_irreps, irreps_out=output_irreps)

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            graph: Molecular graph with node positions, features, and edges

        Returns:
            Predictions (node-level or graph-level depending on aggregate setting)
        """
        # Encode nodes with spherical harmonics and radial basis
        graph.x = self.node_encoder(graph)

        # Apply attention layers with residual connections
        for layer in self.layers:
            updated_node_features = layer(graph)
            graph.x += updated_node_features

        # Decode to output irreps
        graph.x = self.decoder(graph.x)

        # Return node-level or graph-level predictions
        return self.aggregate(graph.x, batch_index=graph.batch)
