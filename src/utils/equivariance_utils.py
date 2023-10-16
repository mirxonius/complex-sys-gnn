from typing import Callable
import torch
from torch_geometric.data import Data
from e3nn import o3
from tqdm import tqdm


def estimate_equivariance_SO3(
    function: Callable,
    input: Data,
    input_irreps: str | o3.Irreps,
    output_irreps: str | o3.Irreps,
    num_samples: int = 100,
    edge_irreps: str | o3.Irreps = None,
):
    """
    For a given function or module (Callable) and input,
    this estimates whether the SO(3) equivariance is satisfied  .
    The proper equivaraince test implies integration over the
    entire SO(3) group, so instead we estimate these integrals
    with Monte Carlo methods.
    NOTE: Not yet supported for data with edge_features/atributes.
    """
    if isinstance(input_irreps, str):
        input_irreps = o3.Irreps(input_irreps)
    if isinstance(output_irreps, str):
        output_irreps = o3.Irreps(output_irreps)
    if isinstance(edge_irreps, str):
        edge_irreps = o3.Irreps(edge_irreps)

    mean_abs_difference = 0.0
    for _ in tqdm(range(num_samples)):
        random_rotaion = o3.rand_matrix()
        D_input = input_irreps.D_from_matrix(random_rotaion)

        D_output = output_irreps.D_from_matrix(random_rotaion)
        input_R = Data(
            x=input.x @ D_input.T,
            pos=input.pos @ random_rotaion.T,
            edge_index=input.edge_index,
            # edge_attr=input.edge_attr
            # if input.edge_attr is not None
            # else input.edge_attr @ D_input_edge.T,
        )
        output_before = function(input_R)
        output_after = function(input) @ D_output.T
        mean_abs_difference = torch.abs(output_before - output_after)

    return (
        torch.allclose(mean_abs_difference, torch.tensor(0.0), atol=1e-4, rtol=1e-4),
        mean_abs_difference / num_samples,
    )
