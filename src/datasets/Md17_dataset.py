"""
MD17 molecular dynamics dataset loaders.

Provides PyTorch datasets for the revised MD17 benchmark with support for
multi-molecule training and extrapolation testing.
"""

from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch_geometric.datasets import MD17
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph

from utils.dataset_utils import RadnomNoise


class MultiMoleculeDataset(Dataset):
    """
    Dataset combining multiple molecules from MD17.

    Supports benzene, ethanol, uracil, and aspirin for multi-molecule training
    with configurable train/valid/test splits via JSON index files.
    """
    def __init__(
        self,
        data_dir: Path | str,
        index_file: str | Path = "multimolecule_index.json",
        split: str = "train",
        radius: float = 1.875,
        molecules: list[str] = ["benzene", "ethanol", "uracil", "aspirin"],
        training_noise: bool = False,
        extra_small: bool = False,
    ) -> None:
        """
        Initialize multi-molecule dataset.

        Args:
            data_dir: Directory containing MD17 data and index JSON file
            index_file: JSON file mapping molecules to train/valid/test indices
            split: One of 'train', 'valid', or 'test'
            radius: Graph edge radius in Angstroms (default: 1.875)
            molecules: List of molecules to include
            training_noise: Whether to add Gaussian noise during training
            extra_small: Use small subset (400 samples) for quick testing
        """
        super().__init__()
        self.radius = radius

        # Load index file (maps molecules to train/valid/test indices)
        with open(Path(data_dir, index_file), "r") as indexing_data:
            self.indexing = json.load(indexing_data)[split]

        # Atomic number table: H=1, C=6, N=7, O=8
        self.z_table = [1, 6, 7, 8]
        self.z_to_index_map = np.vectorize(self.z_to_index)

        # Load and concatenate datasets for selected molecules
        datasets = []
        for molecule in molecules:
            datasets.append(
                Subset(
                    MD17(root=data_dir, name=f"revised {molecule}"),
                    indices=self.indexing[molecule],
                )
            )
        self.data = ConcatDataset(datasets)

        # Optional: Use small subset for quick testing
        if extra_small:
            self.data = Subset(
                self.data,
                indices=list(range(0, 100))
                + list(range(1000, 1100))
                + list(range(2000, 2100))
                + list(range(3000, 3100)),
            )

        # Optional: Add Gaussian noise for data augmentation
        if training_noise:
            self.force_transform = RadnomNoise()
            self.pos_transform = RadnomNoise()
        else:
            self.force_transform = torch.nn.Identity()
            self.pos_transform = torch.nn.Identity()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Data:
        """
        Get a molecular graph with radius-based edges and one-hot atom types.

        Args:
            index: Dataset index

        Returns:
            Graph with pos, force, z (one-hot), and edge_index
        """
        example: Data = self.data[index]

        # Construct graph edges based on spatial radius
        example.edge_index = radius_graph(example.pos, r=self.radius, loop=False)

        # One-hot encode atomic numbers
        Z_ind = self.z_to_index_map(example.z)
        example.z = torch.nn.functional.one_hot(
            torch.tensor(Z_ind), num_classes=4
        ).float()

        # Apply optional noise transforms
        example.pos = self.pos_transform(example.pos)
        example.force = self.force_transform(example.force)
        return example

    def z_to_index(self, Z: int) -> int:
        """
        Map atomic number to index for one-hot encoding.

        Args:
            Z: Atomic number (1=H, 6=C, 7=N, 8=O)

        Returns:
            Index in [0, 1, 2, 3]
        """
        return self.z_table.index(Z)


class ParacetamolDataset(Dataset):
    """
    Paracetamol dataset for extrapolation testing.

    Used to test generalization to unseen molecules after training on
    benzene, ethanol, uracil, and aspirin.
    """
    def __init__(
        self,
        data_dir: str | Path,
        index_file: str | Path = "paracetamol_index.json",
        split="test",
        radius: float = 1.875,
        extra_small=None,
    ) -> None:
        """
        Initialize paracetamol dataset.

        Args:
            data_dir: Directory containing MD17 data and index JSON
            index_file: JSON file with train/valid/test indices
            split: One of 'train', 'valid', or 'test'
            radius: Graph edge radius in Angstroms
            extra_small: Not used (for compatibility)
        """
        super().__init__()
        self.radius = radius

        # Load index file
        with open(Path(data_dir, index_file), "r") as indexing_data:
            self.indexing = json.load(indexing_data)[split]

        # Atomic number table
        self.z_table = [1, 6, 7, 8]
        self.z_to_index_map = np.vectorize(self.z_to_index)

        # Load paracetamol data
        self.data = Subset(
            MD17(root=data_dir, name="revised paracetamol"),
            indices=self.indexing,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Data:
        """Get a molecular graph."""
        example: Data = self.data[index]

        # Construct graph edges
        example.edge_index = radius_graph(example.pos, r=self.radius, loop=False)

        # One-hot encode atomic numbers
        Z_ind = self.z_to_index_map(example.z)
        example.z = torch.nn.functional.one_hot(
            torch.tensor(Z_ind), num_classes=4
        ).float()
        return example

    def z_to_index(self, Z: int) -> int:
        """Map atomic number to index for one-hot encoding."""
        return self.z_table.index(Z)
