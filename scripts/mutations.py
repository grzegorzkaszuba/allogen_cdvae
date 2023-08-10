import torch
import numpy as np
import ase
from ase.io import read
from io import StringIO
from amir_lammps import convert_cif_to_lammps, lammps_data_to_cif, lmp_elastic_calculator, lmp_energy_calculator,\
    lammps_pot, lammps_in
import pandas as pd
import copy
import os
import re
from itertools import combinations
from pymatgen.core import Structure
from conditions import Condition, filter_step_data
import random

print(os.getcwd())
class ASEMutation:
    POSSIBLE_ATOMS = [24, 26, 28]
    def __init__(self):
        pass

    def apply(self, atoms):
        pass

    @staticmethod
    def extract_atom_list(atoms):
        pass

    @staticmethod
    def apply_atom_list(atoms):
        pass


class Transposition(ASEMutation):
    def __init__(self, n_atoms=None):
        self.n_atoms = n_atoms

    def apply(self, atoms):
        atom_types = self.extr
        atoms_used = np.random.sample(torch.arange(len(atom_types)), self.n_atoms*2)
        for i, j in zip(atoms_used[::2], atoms_used[1::2]):
            atom_types[i], atom_types[j] = atom_types[j], atom_types[i]

class Transmutation(ASEMutation):
    def __init__(self, n_atoms, probability):
        self.atom_types = 3
        self.possible_shifts = torch.arange(len())


def struct_localsearch(cif_file, output_file):
    out_file = torch.load(
        'C:\\Users\\GrzegorzKaszuba\\PycharmProjects\\cdvae-main\\hydra\\singlerun\\2023-07-15\\cdvae_new_elastic\\retrain_exp2_max0p9\\3\\lammps_results')
    experiment_path = os.path.join(os.getcwd(), 'localsearch_experiment')
    os.makedirs(experiment_path, exist_ok=True)
    input_cif = out_file['cif_lmp'][35]

    lammps_cfg = {
        'lammps_path': '"C:\\Users\\GrzegorzKaszuba\\AppData\Local\\LAMMPS 64-bit 15Jun2023\\Bin\\lammps-shell.exe"',
        'pot_file': lammps_pot,
        'input_template': lammps_in,
        'pot': 'eam/alloy'}

    cif_file = StringIO(input_cif)
    atoms = read(cif_file, format="cif")
    best_res = 0

    n_steps = 50
    n_samples = 10

    current_state = copy.deepcopy(atoms)
    init_dir = os.path.join(experiment_path)
    os.makedirs(init_dir, exist_ok=True)
    for j in range(n_steps):
        best_prop = best_res
        best_cif = None
        step_dir = os.path.join(init_dir, str(j))
        os.makedirs(step_dir, exist_ok=True)
        cif_dir = os.path.join(step_dir, 'raw_cif')
        os.makedirs(cif_dir, exist_ok=True)
        lammps_dir = os.path.join(step_dir, 'raw_lammps')
        os.makedirs(lammps_dir, exist_ok=True)
        relaxed_cif_dir = os.path.join(step_dir, 'relaxed_cif')
        os.makedirs(relaxed_cif_dir, exist_ok=True)
        relaxed_lammps_dir = os.path.join(step_dir, 'relaxed_lammps')
        step_res = {'initial_energies': [],
                    'final_energies': [],
                    'prop': [],
                    'cif_lmp': []}



        cur_arr = np.array(atoms.numbers)


        samples = [pair for pair in combinations(range(cur_arr.shape[0]), 2) if atoms.numbers[pair[0]] != atoms.numbers[pair[1]]]
        samples = random.shuffle(samples)[:n_samples]


        for i, j in samples:
            arr = arr.copy()
            arr[i], arr[j] = arr[j], arr[i]

            sampled_atoms = copy.deepcopy(atoms)
            sampled_atoms.numbers = arr

            lattice = sampled_atoms.get_cell()
            species = sampled_atoms.get_chemical_symbols()
            coords = sampled_atoms.get_scaled_positions()

            structure = Structure(lattice, species, coords)
            structure.to(
                filename=os.path.join(cif_dir, f'{k}.cif'),
                fmt='cif')

            lammpsdata = convert_cif_to_lammps(cif_dir, lammps_dir)
            initial_energy, final_energy = lmp_energy_calculator(lammps_dir, relaxed_lammps_dir, lammps_cfg=lammps_cfg,
                                                                 silent=True)
            elastic = lmp_elastic_calculator(lammps_dir, lammps_cfg=lammps_cfg, silent=True)
            out_cif = lammps_data_to_cif([f'{i}.data' for i in range(n_samples)], lammps_dir, relaxed_lammps_dir,
                                         savedir=relaxed_cif_dir)
            structure_names = [n.split('.')[0] for n in
                               sorted(os.listdir(relaxed_lammps_dir), key=lambda s: int(re.search('\d+', s).group()))]
            initial_energies = [initial_energy[name] for name in structure_names]
            final_energies = [final_energy[name] for name in structure_names]
            elastic_vectors = [elastic[name] for name in structure_names]
            prop_lmp = [el[3] for el in elastic_vectors]
            step_data = {'initial_energies': initial_energies,
                         'final_energies': final_energies,
                         'elastic_vectors': prop_lmp,
                         'index': list(range(len(prop_lmp)))}

            conditions = [Condition('final_energies', -250, -200),
                          Condition('elastic_vectors', 0, 500)]

            filtered_step_data = filter_step_data(step_data, conditions)
            for l in range(len(filtered_step_data['elastic_vectors'])):
                if filtered_step_data['elastic_vectors'][l] > best_prop:
                    best_prop = filtered_step_data['elastic_vectors'][l]
                    best_cif = filtered_step_data['index'][l]
        if best_prop > best_res:
            patience = 0
            atoms = read(os.path.join(relaxed_cif_dir, str(best_cif) + '.data.cif'), format="cif")
        else:
            patience += 1
            if patience > 3:
                return
        print(f'step: {j}, best_step: {best_cif}, best_prop: {best_prop}')


def expand_dataset(dataset, out_directory, lammps_cfg, property_name='elastic_vector', n_steps=20, n_samples=20):
    cifs = []
    props = []
    datapoint_indices = []
    for i, data in enumerate(dataset.cached_data):
        print(f'i: {i}')
        cif_str = data['cif']

        cif_file = StringIO(cif_str)
        atoms = read(cif_file, format="cif")
        best_res = 0
        best_cif = None

        current_state = copy.deepcopy(atoms)
        struct_dir = os.path.join(out_directory, str(i))
        os.makedirs(struct_dir, exist_ok=True)
        for j in range(n_steps):
            print(f'j: {j}')
            best_prop = 0
            best_cif = None
            step_dir = os.path.join(struct_dir, str(j))
            os.makedirs(step_dir, exist_ok=True)
            cif_dir = os.path.join(step_dir, 'raw_cif')
            os.makedirs(cif_dir, exist_ok=True)
            lammps_dir = os.path.join(step_dir, 'raw_lammps')
            os.makedirs(lammps_dir, exist_ok=True)
            relaxed_cif_dir = os.path.join(step_dir, 'relaxed_cif')
            os.makedirs(relaxed_cif_dir, exist_ok=True)
            relaxed_lammps_dir = os.path.join(step_dir, 'relaxed_lammps')
            step_res = {'initial_energies': [],
                        'final_energies': [],
                        'prop': [],
                        'cif_lmp': []}

            cur_arr = np.array(atoms.numbers)
            samples = [pair for pair in combinations(range(cur_arr.shape[0]), 2) if
                       atoms.numbers[pair[0]] != atoms.numbers[pair[1]]]
            random.shuffle(samples)
            samples = samples[:n_samples]

            for k, (i, j) in enumerate(samples):
                print(f'k: {k}, i, j: {i, j}')
                arr = cur_arr.copy()
                arr[i], arr[j] = arr[j], arr[i]

                sampled_atoms = copy.deepcopy(atoms)
                sampled_atoms.numbers = arr

                lattice = sampled_atoms.get_cell()
                species = sampled_atoms.get_chemical_symbols()
                coords = sampled_atoms.get_scaled_positions()

                structure = Structure(lattice, species, coords)
                structure.to(
                    filename=os.path.join(cif_dir, f'{k}.cif'),
                    fmt='cif')

            lammpsdata = convert_cif_to_lammps(cif_dir, lammps_dir)
            initial_energy, final_energy = lmp_energy_calculator(lammps_dir, relaxed_lammps_dir, lammps_cfg=lammps_cfg,
                                                                 silent=True)
            elastic = lmp_elastic_calculator(lammps_dir, lammps_cfg=lammps_cfg, silent=True)
            out_cif = lammps_data_to_cif([f'{i}' for i in range(n_samples)], lammps_dir, relaxed_lammps_dir,
                                         savedir=relaxed_cif_dir)
            structure_names = [n.split('.')[0] for n in
                               sorted(os.listdir(relaxed_lammps_dir), key=lambda s: int(re.search('\d+', s).group()))]
            initial_energies = [initial_energy[name] for name in structure_names]
            final_energies = [final_energy[name] for name in structure_names]
            elastic_vectors = [elastic[name] for name in structure_names]
            prop_lmp = [el[3] for el in elastic_vectors]
            step_data = {'initial_energies': initial_energies,
                         'final_energies': final_energies,
                         'elastic_vectors': prop_lmp,
                         'index': list(range(len(prop_lmp)))}

            conditions = [Condition('final_energies', -250, -200),
                          Condition('elastic_vectors', 0, 500)]

            filtered_step_data = filter_step_data(step_data, conditions)
            for l in range(len(filtered_step_data['elastic_vectors'])):
                if filtered_step_data['elastic_vectors'][l] > best_prop:
                    best_prop = filtered_step_data['elastic_vectors'][l]
                    best_cif = filtered_step_data['index'][l]
                    print(f'step: {j}, best_prop: {best_prop}')

            if best_prop > best_res:
                patience = 0
                atoms = read(os.path.join(relaxed_cif_dir, str(best_cif) + '.cif'), format="cif")
            else:
                patience += 1
                if patience > 3:
                    break


        if best_cif is not None:
            cifs.append(best_cif)
            props.append(best_prop)
            datapoint_indices.append(i)
    df_data = {'structure_number': datapoint_indices,
                'cif': cifs,
        property_name: props}
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(out_directory, 'dataset.csv'), index=False)




if __name__ == '__main__':
    out_file = torch.load('C:\\Users\\GrzegorzKaszuba\\PycharmProjects\\cdvae-main\\hydra\\singlerun\\2023-07-15\\cdvae_new_elastic\\retrain_exp2_max0p9\\3\\lammps_results')
    experiment_path = os.path.join(os.getcwd(), 'localsearch_experiment')
    os.makedirs(experiment_path, exist_ok=True)
    input_cif = out_file['cif_lmp'][35]

    lammps_cfg = {
        'lammps_path': '"C:\\Users\\GrzegorzKaszuba\\AppData\Local\\LAMMPS 64-bit 15Jun2023\\Bin\\lammps-shell.exe"',
        'pot_file': lammps_pot,
        'input_template': lammps_in,
        'pot': 'eam/alloy'}

    cif_file = StringIO(input_cif)
    atoms = read(cif_file, format="cif")
    best_res = 0


    n_initializations = 5
    n_steps = 50
    n_samples = 10
    for i in range(n_initializations):
        current_state = copy.deepcopy(atoms)
        init_dir = os.path.join(experiment_path, str(i))
        os.makedirs(init_dir, exist_ok=True)
        for j in range(n_steps):
            best_prop = 0
            best_cif = None
            step_dir = os.path.join(init_dir, str(j))
            os.makedirs(step_dir, exist_ok=True)
            cif_dir = os.path.join(step_dir, 'raw_cif')
            os.makedirs(cif_dir, exist_ok=True)
            lammps_dir = os.path.join(step_dir, 'raw_lammps')
            os.makedirs(lammps_dir, exist_ok=True)
            relaxed_cif_dir = os.path.join(step_dir, 'relaxed_cif')
            os.makedirs(relaxed_cif_dir, exist_ok=True)
            relaxed_lammps_dir =os.path.join(step_dir, 'relaxed_lammps')
            step_res = {'initial_energies': [],
                              'final_energies': [],
                              'prop': [],
                              'cif_lmp': []}
            for k in range(n_samples):
                arr = np.array(atoms.numbers)
                indices_0 = np.where(arr == 26)[0]
                indices_1 = np.where(arr == 28)[0]
                index_0 = np.random.choice(indices_0)
                index_1 = np.random.choice(indices_1)
                arr[index_0], arr[index_1] = arr[index_1], arr[index_0]

                sampled_atoms = copy.deepcopy(atoms)
                sampled_atoms.numbers = arr

                lattice = sampled_atoms.get_cell()
                species = sampled_atoms.get_chemical_symbols()
                coords = sampled_atoms.get_scaled_positions()

                structure = Structure(lattice, species, coords)
                structure.to(
                    filename=os.path.join(cif_dir, f'{k}.cif'),
                    fmt='cif')


            lammpsdata = convert_cif_to_lammps(cif_dir, lammps_dir)
            initial_energy, final_energy = lmp_energy_calculator(lammps_dir, relaxed_lammps_dir, lammps_cfg=lammps_cfg, silent=True)
            elastic = lmp_elastic_calculator(lammps_dir, lammps_cfg=lammps_cfg, silent=True)
            out_cif = lammps_data_to_cif([f'{i}.data' for i in range(n_samples)], lammps_dir, relaxed_lammps_dir, savedir=relaxed_cif_dir)
            structure_names = [n.split('.')[0] for n in
                               sorted(os.listdir(relaxed_lammps_dir), key=lambda s: int(re.search('\d+', s).group()))]
            initial_energies = [initial_energy[name] for name in structure_names]
            final_energies = [final_energy[name] for name in structure_names]
            elastic_vectors = [elastic[name] for name in structure_names]
            prop_lmp = [el[3] for el in elastic_vectors]
            step_data = {'initial_energies': initial_energies,
                         'final_energies': final_energies,
                         'elastic_vectors': prop_lmp,
                         'index': list(range(len(prop_lmp)))}

            conditions = [Condition('final_energies', -250, -200),
                          Condition('elastic_vectors', 0, 500)]

            filtered_step_data = filter_step_data(step_data, conditions)
            for l in range(len(filtered_step_data['elastic_vectors'])):
                if filtered_step_data['elastic_vectors'][l] > best_prop:
                    best_prop = filtered_step_data['elastic_vectors'][l]
                    best_cif = filtered_step_data['index'][l]
            atoms = read(os.path.join(relaxed_cif_dir, str(best_cif)+'.data.cif'), format="cif")
            print(f'init: {i}, step: {j}, best_step: {best_cif}, best_prop: {best_prop}')


