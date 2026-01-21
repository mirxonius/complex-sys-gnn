"""
Setup utilities for models, datasets, metrics, and losses.

Provides factory functions to initialize components based on configuration files.
"""

import json
from pathlib import Path
import torch
from models.equivariant_gat import O3GraphAttentionNetwork
from models.gat_model import GATModel
from models.mace_model import MaceNet
from utils.loss_utils import MSE_MAE_Loss, HuberScalarLoss

from config_defaults import (
    dataset_dict,
    task_dataset_kwargs,
    Tasks,
    SupportedLosses,
    SupportedModels,
)


def set_up_model(model_name, model_args_json):
    """
    Initialize a model from JSON configuration.

    Args:
        model_name: Name of the model (must exist in model_args_json)
        model_args_json: Path to JSON file with model configurations

    Returns:
        Initialized PyTorch model

    Raises:
        KeyError: If model_name not found in JSON
        ValueError: If model_type is not supported
    """
    with open(model_args_json, "r") as file:
        try:
            model_info = json.load(file)[model_name]
        except KeyError:
            raise KeyError(f"{model_name} is not found in {model_args_json}.")

    model_type = model_info["model_type"]
    model_args = model_info["model_args"]

    # Instantiate model based on type
    if model_type == SupportedModels.equivariant_gat.value:
        model = O3GraphAttentionNetwork(**model_args)
    elif model_type == SupportedModels.gat_model.value:
        model = GATModel(**model_args)
    elif model_type == SupportedModels.mace_model.value:
        model = MaceNet(**model_args)
    else:
        raise ValueError(f"{model_type} is not a supported model type.")

    return model


def set_up_dataset(
    task,
    dataset_data_dir: str | Path = None,
    training_noise: bool = False,
    extra_small: bool = False,
):
    """
    Initialize train/valid/test datasets for a task.

    Args:
        task: Task name (e.g., 'multi_molecule_forces')
        dataset_data_dir: Path to MD17 data directory
        training_noise: Whether to add Gaussian noise to training data
        extra_small: Use small subset for quick testing

    Returns:
        Tuple of (train_set, valid_set, test_set)
    """
    if dataset_data_dir is None:
        dataset_data_dir = Path("../../data/md17")

    dataset_kwargs = task_dataset_kwargs[task]

    # Training noise not applicable to paracetamol (extrapolation test set)
    if task != Tasks.paracetamol.value:
        dataset_kwargs["training_noise"] = training_noise
    else:
        extra_small = False

    # Create datasets
    train_set = dataset_dict[task](
        data_dir=dataset_data_dir, split="train", extra_small=extra_small, **dataset_kwargs
    )
    valid_set = dataset_dict[task](
        data_dir=dataset_data_dir, split="valid", **dataset_kwargs
    )
    test_set = dataset_dict[task](
        data_dir=dataset_data_dir, split="valid", **dataset_kwargs
    )

    return train_set, valid_set, test_set


def set_up_metric(task):
    """
    Get metric configuration for a task.

    Args:
        task: Task name

    Returns:
        Dictionary of metric calculator kwargs

    Raises:
        ValueError: If task doesn't have metric calculator defined
    """
    match task:
        case Tasks.multi_molecule_forces.value:
            return {"num_outputs": 3}  # 3D force vectors
        case Tasks.benzene_forces.value:
            return {"num_outputs": 3}
        case _:
            raise ValueError(
                f"{task} does not have metric calculator setup implemented."
            )


def set_up_loss(loss: str):
    """
    Initialize a loss function.

    Args:
        loss: Loss function name (mse, mae, mse_mae, huber, huber_scalar)

    Returns:
        PyTorch loss module

    Raises:
        ValueError: If loss function is not supported
    """
    match loss:
        case SupportedLosses.mae.value:
            return torch.nn.L1Loss()  # Mean Absolute Error
        case SupportedLosses.mse.value:
            return torch.nn.MSELoss()  # Mean Squared Error
        case SupportedLosses.mse_mae.value:
            return MSE_MAE_Loss()  # Combined MSE + MAE
        case SupportedLosses.huber.value:
            return torch.nn.HuberLoss()  # Robust to outliers
        case SupportedLosses.huber_scalar.value:
            return HuberScalarLoss(alpha=2.0)  # Scalar Huber loss
        case _:
            raise ValueError(f"{loss} is not a supported loss function!")
