import json
from pathlib import Path
import os
import torch

from models.equivariant_gat import O3GraphAttentionNetwork
from models.gat_model import GATModel
from models.mace_model import MaceNet

from datasets import QM9_dataset
from config_defaults import SupportedModels
from config_defaults import dataset_dict


def set_up_model(model_name, model_args_json):
    with open(model_args_json, "r") as file:
        model_kwargs = json.load(file)

    if model_name == SupportedModels.equivariant_gat.value:
        model = O3GraphAttentionNetwork(**model_kwargs)

    elif model_name == SupportedModels.gat_model.value:
        model = GATModel(**model_kwargs)

    elif model_name == SupportedModels.mace_model.value:
        model = MaceNet(**model_kwargs)

    else:
        print(dataset_dict)
        raise ValueError(f"{model_name} is not supported a supported model.")

    return model


def set_up_dataset(
    task, dataset_root_dir: str | Path = None, dataset_index_dir: str | Path = None
):
    if dataset_root_dir is None:
        dataset_root_dir = Path("../../data/", task)
    if dataset_index_dir is not None:
        train_idx = torch.load(Path(dataset_index_dir, "train_idx.pt"))
        valid_idx = torch.load(Path(dataset_index_dir, "talid_idx.pt"))
        test_idx = torch.load(Path(dataset_index_dir, "test_idx.pt"))
        try:
            train_set = dataset_dict[task](root_dir=dataset_root_dir, indices=train_idx)
            valid_set = dataset_dict[task](root_dir=dataset_root_dir, indices=valid_idx)
            test_set = dataset_dict[task](root_dir=dataset_root_dir, indices=test_idx)

        except:
            raise KeyError(f"Task {task} is not defined")
    else:
        try:
            full_dataset = dataset_dict[task](root_dir=dataset_root_dir)
            train_set, valid_set = torch.utils.data.random_split(
                full_dataset, lengths=[0.7, 0.3]
            )
            valid_set, test_set = torch.utils.data.random_split(
                valid_set, lengths=[0.6, 0.4]
            )
        except:
            raise KeyError(f"Task {task} is not defined")
    return train_set, valid_set, test_set
