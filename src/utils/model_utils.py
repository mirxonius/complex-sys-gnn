import torch
from torch_scatter import scatter


def softmax_on_graph(
    input: torch.Tensor, index: torch.LongTensor, dim: int = 0
) -> torch.Tensor:
    """
    Performs numerically stable softmax on graph i.e.
    aggregates messages from neighboring nodes where
    the message weights are calculated using softmax.
    """
    stabilizer = scatter(src=input, index=index, dim=dim, reduce="max")
    exp = torch.exp(input - stabilizer[index])
    Z = scatter(src=exp, index=index, dim=dim, reduce="sum")
    Z[Z == 0] = 1
    return exp / Z[index]


class Mean(torch.nn.Module):
    def __init__(self, dim=0, keepdim=False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class MeanOnGraph(torch.nn.Module):
    def __init__(self, dim=0, keepdim=False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, node_features, batch_index=None):
        if batch_index is None:
            return torch.mean(node_features, dim=self.dim, keepdim=self.keepdim)
        return scatter(
            src=node_features, dim=self.dim, index=batch_index, reduce="mean"
        )


class IdentityOnGraph(torch.nn.Module):
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
    use_normalization=False,
):
    layer_sizes = [in_shape] + [hidden_shape] * num_hidden_layers + [out_shape]
    layers = []
    for i in range(len(layer_sizes) - 1):
        if use_normalization:
            layers.append(torch.nn.BatchNorm1d(layer_sizes[i]))
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 1:
            layers.append(activation)
    return torch.nn.Sequential(*layers)
