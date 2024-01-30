import json
from pathlib import Path
import os
import sys
import torch

from models.equivariant_gat import O3GraphAttentionNetwork
from models.gat_model import GATModel
from models.mace_model import MaceNet
from utils.loss_utils import MSE_MAE_Loss

from config_defaults import dataset_dict,task_dataset_kwargs, Tasks, SupportedLosses, SupportedModels


def set_up_model(model_name, model_args_json):
    with open(model_args_json, "r") as file:
        try:
            model_info = json.load(file)[model_name]
        except:
            raise KeyError(f"{model_name} is not found.")
    model_type = model_info["model_type"]
    model_args = model_info["model_args"]
    if model_type == SupportedModels.equivariant_gat.value:
        model = O3GraphAttentionNetwork(**model_args)

    elif model_type == SupportedModels.gat_model.value:
        model = GATModel(**model_args)

    elif model_type == SupportedModels.mace_model.value:
        model = MaceNet(**model_args)

    else:
        raise ValueError(f"{model_name} is not supported a supported model.")

    return model


def set_up_dataset(
    task, dataset_data_dir: str | Path = None
):
    if dataset_data_dir is None:
        dataset_data_dir = Path("../../data/md17")

        # train_set = dataset_dict[task](data_dir=dataset_data_dir, split="train")
        # valid_set = dataset_dict[task](data_dir=dataset_data_dir, split="valid")
        # test_set = dataset_dict[task](data_dir=dataset_data_dir, split="valid")
    #try:
    dataset_kwargs = task_dataset_kwargs[task]
    train_set = dataset_dict[task](data_dir=dataset_data_dir, split="train",**dataset_kwargs)
    valid_set = dataset_dict[task](data_dir=dataset_data_dir, split="valid",**dataset_kwargs)
    test_set = dataset_dict[task](data_dir=dataset_data_dir, split="valid",**dataset_kwargs)

    #except:
    #    raise KeyError(f"Task {task} is not defined")

    return train_set, valid_set, test_set


def set_up_metric(task):
    match task:
        case Tasks.tri_molecule_forces.value:
            return {"num_outputs": 3}
        case Tasks.benzene_forces.value:
            return {"num_outputs": 3}
        case _:
            raise ValueError(f"{task} does not have metirc calculator set up implemented.")


def set_up_loss(loss: str):
    match loss:
        case SupportedLosses.mae.value:
            return torch.nn.L1Loss()
        case SupportedLosses.mse.value:
            return torch.nn.MSELoss()
        case SupportedLosses.mse_mae.value:
            return MSE_MAE_Loss()
        case SupportedLosses.huber.value:
            return torch.nn.HuberLoss()
        case _:
            raise ValueError(f"{loss} is not supported!")
