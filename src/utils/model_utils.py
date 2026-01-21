"""
Utility functions and modules for graph neural networks.

Includes aggregation operations, softmax on graphs, and network builders.
"""

import torch
from torch_scatter import scatter


def softmax_on_graph(
    input: torch.Tensor, index: torch.LongTensor, dim: int = 0
) -> torch.Tensor:
    """
    Numerically stable softmax for graph-structured data.

    Computes softmax over messages grouped by index (e.g., edges → nodes).

    Args:
        input: Values to normalize
        index: Grouping indices (e.g., destination node for each edge)
        dim: Dimension to reduce over

    Returns:
        Normalized values with softmax applied per group
    """
    stabilizer = scatter(src=input, index=index, dim=dim, reduce="max")
    exp = torch.exp(input - stabilizer[index])
    Z = scatter(src=exp, index=index, dim=dim, reduce="sum")
    Z[Z == 0] = 1
    return exp / Z[index]


class Mean(torch.nn.Module):
    """Simple mean aggregation module."""

    def __init__(self, dim=0, keepdim=False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class MeanOnGraph(torch.nn.Module):
    """
    Mean aggregation for graph data.

    Aggregates node features to graph-level by averaging over batch indices.
    """

    def __init__(self, dim=0, keepdim=False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, node_features, batch_index=None):
        """
        Aggregate node features to graph level.

        Args:
            node_features: Node-level features [num_nodes, feature_dim]
            batch_index: Batch assignment for each node

        Returns:
            Graph-level features (averaged over nodes in each graph)
        """
        if batch_index is None:
            return torch.mean(node_features, dim=self.dim, keepdim=self.keepdim)
        return scatter(
            src=node_features, dim=self.dim, index=batch_index, reduce="mean"
        )


class IdentityOnGraph(torch.nn.Module):
    """
    Identity operation (no aggregation).

    Used for node-level predictions where no graph-level pooling is needed.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, input: torch.Tensor, batch_index: torch.LongTensor = None
    ) -> torch.Tensor:
        return input


def make_fcnn(
    in_shape,
    hidden_shape,
    out_shape,
    num_hidden_layers=1,
    activation=torch.nn.ReLU(),
    use_normalization=True,
):
    """
    Build a fully connected neural network.

    Args:
        in_shape: Input dimension
        hidden_shape: Hidden layer dimension
        out_shape: Output dimension
        num_hidden_layers: Number of hidden layers
        activation: Activation function
        use_normalization: Whether to use batch normalization

    Returns:
        Sequential neural network
    """
    layer_sizes = [in_shape] + [hidden_shape] * num_hidden_layers + [out_shape]
    layers = []
    for i in range(len(layer_sizes) - 1):
        if use_normalization:
            layers.append(torch.nn.BatchNorm1d(layer_sizes[i]))
        if i < len(layer_sizes) - 1:
            layers.append(activation)
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    return torch.nn.Sequential(*layers)
