from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch_geometric.datasets import MD17
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph


class BenzeneEthanolUracilDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | str,
        index_file: str | Path,
        split: str = "train",
        radius: float = 1.875,
    ) -> None:
        """
        Args:
            root_dir: Directory where data is stored for each molecule alongside an index_file.json
            index_file: .json file path used to index molecules from benzene, ethanol and uracil datasets.
            split: 'train', 'valid' 'test'
            radius: distrance within the molecules interact in angstrom
        """
        super().__init__()
        self.radius = radius
        with open(Path(data_dir, index_file), "r") as indexing_data:
            """
            Example indexing data:
                indexing_data =  {
                "train":{
                "benzene":[0,2,11,832,...], indices taken from benzene dataset for training
                "uracil": [1,2,3,72,...],
                "ethanol": [2,22,123,...]
                },
                "valid":{"benzene":[...],"uracil":[...],"ethanol":[...]},
                "test":{"benzene":[...],"uracil":[...],"ethanol":[...]}
            }
            """
            self.indexing = json.load(indexing_data)[split]
        # Table containing all possible atomic numbers that can occur
        # in dataset: H, O, N, C
        self.z_table = [1, 6, 7, 8]
        self.z_to_index_map = np.vectorize(self.z_to_index)

        benzene = Subset(
            MD17(root=data_dir, name="revised benzene"),
            indices=self.indexing["benzene"],
        )
        uracil = Subset(
            MD17(root=data_dir, name="revised uracil"), indices=self.indexing["uracil"]
        )
        ethanol = Subset(
            MD17(root=data_dir, name="revised ethanol"),
            indices=self.indexing["ethanol"],
        )
        self.data = ConcatDataset([benzene, uracil, ethanol])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Data:
        example: Data = self.data[index]
        example.edge_index = radius_graph(example.pos, r=self.radius, loop=False)
        Z_ind = self.z_to_index_map(example.z)
        example.z = torch.nn.functional.one_hot(torch.tensor(Z_ind), num_classes=4)
        return example

    def z_to_index(self, Z: int) -> int:
        """
        Used for one hot encoding
        of atomic numbers.
        NOTE: This function is never explicitly called, rather
              we use its vectorized form declared in the __init__ function.
        """
        return self.z_table.index(Z)


class ParacetamolDataset(Dataset):
    def __init__(self) -> None:
        self.z_table = [1, 6, 7, 8]
        self.z_to_index_map = np.vectorize(self.z_to_index)

    def z_to_index(self, Z: int) -> int:
        """
        Used for one hot encoding
        of atomic numbers.
        NOTE: This function is never explicitly called, rather
              we use its vectorized form declared in the __init__ function.
        """
        return self.z_table.index(Z)
