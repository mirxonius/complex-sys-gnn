import torch
from torch_scatter import scatter


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
