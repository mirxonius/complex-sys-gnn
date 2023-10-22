import json
from models.equivariant_gat import O3GraphAttentionNetwork
from models.gat_model import GATModel
from models.mace_model import MaceNet

from config_defaults import SupportedModels


def set_up_model(model_name, model_args_json):
    with open(model_args_json, "r") as file:
        model_kwargs = json.load(file)

    if model_name == SupportedModels.equivariant_gat:
        model = O3GraphAttentionNetwork(**model_kwargs)

    elif model_name == SupportedModels.gat_model:
        model = GATModel(**model_kwargs)

    elif model_name == SupportedModels.mace_model:
        model = MaceNet(**model_kwargs)

    else:
        raise ValueError(f"{model_name} is not supported a supported model.")

    return model
