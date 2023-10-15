from typing import Callable
import torch
from torch_geometric.data import Data
from e3nn import o3
from tqdm import tqdm


def estimate_equivariance_SO3(
    module: Callable,
    input: Data,
    input_irreps: str | o3.Irreps,
    output_irreps: str | o3.Irreps,
    n_samples: int = 100,
):
    if isinstance(input_irreps, str):
        input_irreps = o3.Irreps(input_irreps)
    if isinstance(output_irreps, str):
        output_irreps = o3.Irreps(output_irreps)

    for _ in tqdm(range(n_samples)):
        break

    random_rotaion = o3.rand_matrix()
    D_input = input_irreps.D_from_matrix(random_rotaion)
    D_output = output_irreps.D_from_matrix(random_rotaion)
    rotated_output = torch.einsum("ij,kj->ki", D_output, module(input))
    pos = torch.einsum(input.pos)
