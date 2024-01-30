from enum import Enum
from datasets.Md17_dataset import ParacetamolDataset, MultiMoleculeDataset


class Tasks(Enum):
    multi_molecule_forces = "multi_molecule_forces"
    multi_molecule_energy = "multi_molecule_energy"
    benzene_forces = "benzene_forces"
    benzene_energy = "benzene_energy"
    uracil_forces = "uracail_forces"
    uracil_energy = "uracail_energy"
    ethanol_forces = "ethaned_forces"
    ethanol_energy = "ethanol_energy"
    paracetamol = "paracetamol"


dataset_dict = {
    Tasks.multi_molecule_energy.value: MultiMoleculeDataset,
    Tasks.multi_molecule_forces.value: MultiMoleculeDataset,
    Tasks.paracetamol.value: ParacetamolDataset,
    Tasks.benzene_forces.value:MultiMoleculeDataset,
    Tasks.ethanol_forces.value:MultiMoleculeDataset,
    Tasks.uracil_forces.value:MultiMoleculeDataset,

}


task_dataset_kwargs = {
    Tasks.multi_molecule_energy.value: {},
    Tasks.multi_molecule_forces.value: {},
    Tasks.paracetamol.value: {},
    Tasks.benzene_forces.value:{"molecules":["benzene"]},
    Tasks.uracil_forces.value:{"molecules":["uracil"]},
    Tasks.ethanol_forces.value:{"molecules":["ethanol"]}


}

class SupportedLosses(Enum):
    mae = "mae"
    mse = "mse"
    mse_mae = "mse_mae"
    huber = "huber"


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
