import json
from pathlib import Path
import os
import sys
import torch

from models.equivariant_gat import O3GraphAttentionNetwork
from models.gat_model import GATModel
from models.mace_model import MaceNet

from config_defaults import SupportedModels
from config_defaults import dataset_dict, Tasks


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
    task, dataset_data_dir: str | Path = None, dataset_index_dir: str | Path = None
):
    if dataset_data_dir is None:
        dataset_data_dir = Path("../../data/md17")

        # train_set = dataset_dict[task](data_dir=dataset_data_dir, split="train")
        # valid_set = dataset_dict[task](data_dir=dataset_data_dir, split="valid")
        # test_set = dataset_dict[task](data_dir=dataset_data_dir, split="valid")

    try:
        train_set = dataset_dict[task](data_dir=dataset_data_dir, split="train")
        valid_set = dataset_dict[task](data_dir=dataset_data_dir, split="valid")
        test_set = dataset_dict[task](data_dir=dataset_data_dir, split="valid")

    except:
        raise KeyError(f"Task {task} is not defined")

    return train_set, valid_set, test_set


def set_up_metric(task):
    match task:
        case Tasks.tri_molecule_forces.value:
            return {"num_outputs": 3}
