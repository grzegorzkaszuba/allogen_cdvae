import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import numpy as np

from torch_geometric.data import Data

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop, preprocess_ad_hoc)

from cdvae.common.data_utils import get_scaler_from_data_list

class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode,
                 **kwargs):


        if 'crystal_phases' in kwargs:
            self.crystal_phases = kwargs.get('crystal_phases')
        else:
            self.crystal_phases = None
        if 'preprocess_limit' in kwargs:
            preprocess_limit = kwargs.get('preprocess_limit')
        else:
            preprocess_limit = None
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)[:preprocess_limit]
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            preprocess_limit=preprocess_limit,
            crystal_phases=self.crystal_phases)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
            phase=data_dict['phase']
        )
        return data

    def relabel(self, new_labels, reset_scaler=True, model=None):
        for i, cd in enumerate(self.cached_data):
            cd[self.prop] = np.float64(str(new_labels[i]))
        if reset_scaler:
            self.scaler = get_scaler_from_data_list(
            self.cached_data,
            key=self.prop)
            if model is None:
                raise ValueError('If you replace the scaler of the dataset, you must change the model scaler as well')
            else:
                model.scaler = self.scaler.copy()
                model.reinitialize_fc_property()

    def get_properties(self):
        properties = []
        for cd in self.cached_data:
            properties.append(cd[self.prop])
        return properties

    def get_compositions(self, atomic_numbers):
        compositions = []
        for cd in self.cached_data:
            composition = tuple(np.bincount(cd['graph_arrays'][1], minlength=29)[atomic_numbers].tolist())
            compositions.append(composition)
        return compositions


    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class AdHocCrystDataset(Dataset):
    def __init__(self, name, cif_data: list, prop_data, phase_data, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):

        if 'preprocess_limit' in kwargs:
            preprocess_limit = kwargs.get('preprocess_limit')
        else:
            preprocess_limit = None
        super().__init__()
        self.path = ''
        self.cif_data = cif_data
        self.name = name
        prop_name = kwargs.get('prop_name') if 'prop_name' in kwargs else 'prop'
        self.prop = prop_name if prop_data is not None else None
        # construct df ad hoc
        if self.prop is None:
            df_data = {'material_id': [i for i in range(len(cif_data))], 'cif': cif_data, 'phase': phase_data}
        else:
            df_data = {'material_id': [i for i in range(len(cif_data))], 'cif': cif_data,  prop_name: prop_data, 'phase': phase_data}
        self.df = pd.DataFrame(df_data)
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        if self.prop is None:
            self.cached_data = preprocess_ad_hoc(
                self.df,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                preprocess_limit=preprocess_limit)
        else:
            self.cached_data = preprocess_ad_hoc(
                self.df,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[self.prop],
                preprocess_limit=preprocess_limit)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = kwargs.get('lattice_scaler').copy() if 'lattice_scaler' in kwargs else None
        self.scaler = kwargs.get('scaler').copy() if 'scaler' in kwargs else None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']
        # scaler is set in DataModule set stage
        phase = data_dict['phase']
        if self.scaler is not None and data_dict[self.prop] is not None:
            prop = self.scaler.transform(data_dict[self.prop])

            # atom_coords are fractional coordinates
            # edge_index is incremented during batching
            # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
            data = Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                lengths=torch.Tensor(lengths).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                edge_index=torch.LongTensor(
                    edge_indices.T).contiguous(),  # shape (2, num_edges)
                to_jimages=torch.LongTensor(to_jimages),
                num_atoms=num_atoms,
                num_bonds=edge_indices.shape[0],
                num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
                y=prop.view(1, -1),
                phase=torch.Tensor(phase)
            )

        else:
            data = Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                lengths=torch.Tensor(lengths).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                edge_index=torch.LongTensor(
                    edge_indices.T).contiguous(),  # shape (2, num_edges)
                to_jimages=torch.LongTensor(to_jimages),
                num_atoms=num_atoms,
                num_bonds=edge_indices.shape[0],
                num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
                phase=torch.Tensor(phase)
            )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"

class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        if 'y' not in data_dict.keys():
            data = Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                lengths=torch.Tensor(lengths).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                edge_index=torch.LongTensor(
                    edge_indices.T).contiguous(),  # shape (2, num_edges)
                to_jimages=torch.LongTensor(to_jimages),
                num_atoms=num_atoms,
                num_bonds=edge_indices.shape[0],
                num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            )
        else:
            data = Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                lengths=torch.Tensor(lengths).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                edge_index=torch.LongTensor(
                    edge_indices.T).contiguous(),  # shape (2, num_edges)
                to_jimages=torch.LongTensor(to_jimages),
                num_atoms=num_atoms,
                num_bonds=edge_indices.shape[0],
                num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
                y=torch.Tensor(data_dict['y']).view(-1, 1)
            )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
