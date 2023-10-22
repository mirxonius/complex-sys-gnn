from typing import Optional, Callable, Iterable
from pathlib import Path
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.datasets import QM9


class QM9_dataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        transforms: Optional[Callable] = lambda x: x,
        indices: str | Path | Iterable = None,
    ) -> None:
        super().__init__()
        self.data = QM9(root_dir)
        self.transforms = transforms
        if isinstance(indices, Path) or isinstance(indices, str):
            indices = torch.load(indices)
        self.indices = (
            torch.arange(len(self.data), dtype=int) if indices is None else indices
        )
        self.data = Subset(self.data, self.indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        graph = self.data[index]
        graph = self.transforms(graph)
        return graph
