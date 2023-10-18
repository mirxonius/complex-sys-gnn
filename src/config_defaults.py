from enum import Enum


class SupportedModels(Enum):
    gat_model = "gat_model"
    gate_equiv_model = "gate_equiv_model"
    mace_model = "mace_model"
    equivariant_gat = "equivariant_gat"


class SupportedDatasets(Enum):
    QM9 = "QM9"
    top_tagging = "top_tagging"
    flag = "flag"
    aero = "aero"


batch_size = 128
num_epochs = 20
lr = 1e-2
