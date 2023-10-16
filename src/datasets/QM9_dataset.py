from typing import Optional, Callable
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9


class QM9_dataset(Dataset):
    def __init__(
        self, root_dir: str | Path, transforms: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.ds = QM9(root_dir)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index) -> Any:
        return NotImplemented
