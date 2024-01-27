from enum import Enum
from datasets.Md17_dataset import ParacetamolDataset, BenzeneEthanolUracilDataset


class Tasks(Enum):
    tri_molecule_forces = "tri_molecule_forces"
    tri_molecule_energy = "tri_molecule_energy"
    benzene_forces = "benzene_forces"
    benzene_energy = "benzene_energy"
    uracail_forces = "uracail_forces"
    uracail_energy = "uracail_energy"
    ethanol_forces = "ethaned_forces"
    ethanol_energy = "ethanol_energy"
    paracetamol = "paracetamol"


dataset_dict = {
    Tasks.tri_molecule_energy.value: BenzeneEthanolUracilDataset,
    Tasks.tri_molecule_forces.value: BenzeneEthanolUracilDataset,
    Tasks.paracetamol: ParacetamolDataset,
}


class SupportedLosses(Enum):
    mae = "mae"
    mse = "mse"
    mse_mae = "mse_mae"


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
