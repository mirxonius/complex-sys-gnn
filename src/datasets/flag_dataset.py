from typing import Optional, Callable
from pathlib import Path
from torch.utils.data import Dataset


class FlagDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        trasfroms: Optional[Callable] = None,
        from_tfrecord: bool = True,
    ):
        pass

    def __len__(self):
        return NotImplemented

    def __getitem__(self, index):
        return NotImplemented
