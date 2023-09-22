import torch
import numpy as np
import ase
from ase.io import read
from io import StringIO
from amir_lammps import convert_cif_to_lammps, lammps_data_to_cif, lmp_elastic_calculator, lmp_energy_calculator, \
    lammps_pot, lammps_in
import pandas as pd
import copy
import os
import re
from itertools import combinations
from pymatgen.core import Structure
from conditions import Condition, filter_step_data
import random
import yaml
import time

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
        atoms_used = np.random.sample(torch.arange(len(atom_types)), self.n_atoms * 2)
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

    with open('subprocess_calls.yaml', 'r') as file:
        lmp_path = yaml.safe_load(file).get('lammps_call')
        if type(lmp_path) == str:
            lmp_path = [lmp_path]
    lammps_cfg = {
        'lammps_path': lmp_path,
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

        samples = [pair for pair in combinations(range(cur_arr.shape[0]), 2) if
                   atoms.numbers[pair[0]] != atoms.numbers[pair[1]]]
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
                                                                 silent=False)
            elastic = lmp_elastic_calculator(lammps_dir, lammps_cfg=lammps_cfg, silent=False)
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


def simulate_ground_structure(atoms, struct_dir, lammps_cfg):
    step_dir = os.path.join(struct_dir, 'ground_structure')
    os.makedirs(step_dir, exist_ok=True)
    cif_dir = os.path.join(step_dir, 'raw_cif')
    os.makedirs(cif_dir, exist_ok=True)
    lammps_dir = os.path.join(step_dir, 'raw_lammps')
    os.makedirs(lammps_dir, exist_ok=True)
    relaxed_cif_dir = os.path.join(step_dir, 'relaxed_cif')
    os.makedirs(relaxed_cif_dir, exist_ok=True)
    relaxed_lammps_dir = os.path.join(step_dir, 'relaxed_lammps')

    lattice = atoms.get_cell()
    species = atoms.get_chemical_symbols()
    coords = atoms.get_scaled_positions()

    structure = Structure(lattice, species, coords)
    structure.to(
        filename=os.path.join(cif_dir, 'g.cif'),
        fmt='cif')

    lammpsdata = convert_cif_to_lammps(cif_dir, lammps_dir)
    initial_energy, final_energy = lmp_energy_calculator(lammps_dir, relaxed_lammps_dir, lammps_cfg=lammps_cfg,
                                                         silent=True)
    elastic = lmp_elastic_calculator(lammps_dir, lammps_cfg=lammps_cfg, silent=True)
    out_cif = lammps_data_to_cif(['g'], lammps_dir, relaxed_lammps_dir,
                                 savedir=relaxed_cif_dir)
    structure_names = ['g']
    initial_energies = [initial_energy[name] for name in structure_names]
    final_energies = [final_energy[name] for name in structure_names]
    elastic_vectors = [elastic[name] for name in structure_names]
    prop_lmp = [el[3] for el in elastic_vectors]
    return prop_lmp, initial_energies, final_energies


def expand_dataset(dataset, out_directory, lammps_cfg, property_name='elastic_vector', n_steps=20, n_samples=20,
                   n_examples=100):
    # setup
    starting_time = time.time()
    cifs = []
    props = []
    datapoint_indices = []
    new_datapoints = []
    result_record = {}
    # randomly choose the points to optimize
    processed_points = np.arange(len(dataset.cached_data))
    rs = np.random.RandomState(41)
    rs.shuffle(processed_points)
    processed_indices = processed_points[:n_examples]
    for i, ind in enumerate(processed_indices):
        print(time.time() - starting_time)
        patience = 0
        data = dataset.cached_data[ind]
        new_datapoint = {'material_id': data['mp_id'] + 'L',
                         'formation_energy_per_atom': data['ealstic_vector'],
                         'ealstic_vector': data['ealstic_vector'],
                         'sym': 'fcc' if data['phase'] == 0 else 1,
                         'pretty_formula': data['mp_id'].split('_')[0],
                         'cif': data['cif']}
        add_to_dataset = False
        print(f'index: {ind}, i: {i}, phase: {new_datapoint["sym"]}')
        cif_str = data['cif']
        starting_prop = data['ealstic_vector']
        best_prop = starting_prop
        result_record[i] = {'best_cifs': [],
                            'best_props': [],
                            'starting_cif': cif_str,
                            'starting_prop': starting_prop,
                            'source_datapoints': [],
                            'added_to_dataset': []}

        cif_file = StringIO(cif_str)
        atoms = read(cif_file, format="cif")
        best_cif = cif_str
        struct_dir = os.path.join(out_directory, str(i))
        os.makedirs(struct_dir, exist_ok=True)
        ground_prop, ground_energy_init, ground_energy_final = simulate_ground_structure(atoms, struct_dir, lammps_cfg)
        print(f'Ground structure prop: {ground_prop}, initial energy: {ground_energy_init} final energy: {ground_energy_final}')
        for j in range(n_steps):
            print(f'step: {j}, starting prop: {starting_prop}', f'best prop so far: {best_prop}, formula: {new_datapoint["material_id"]}, {new_datapoint["pretty_formula"]}')
            best_step_cif = best_cif
            best_step_prop = 0
            step_dir = os.path.join(struct_dir, str(j))
            os.makedirs(step_dir, exist_ok=True)
            cif_dir = os.path.join(step_dir, 'raw_cif')
            os.makedirs(cif_dir, exist_ok=True)
            lammps_dir = os.path.join(step_dir, 'raw_lammps')
            os.makedirs(lammps_dir, exist_ok=True)
            relaxed_cif_dir = os.path.join(step_dir, 'relaxed_cif')
            os.makedirs(relaxed_cif_dir, exist_ok=True)
            relaxed_lammps_dir = os.path.join(step_dir, 'relaxed_lammps')

            cur_arr = np.array(atoms.numbers)
            samples = [pair for pair in combinations(range(cur_arr.shape[0]), 2) if
                       atoms.numbers[pair[0]] != atoms.numbers[pair[1]]]
            random.shuffle(samples)
            samples = samples[:n_samples]

            for k, (l, m) in enumerate(samples):
                #print(f'k: {k}, i, j: {l, m}')
                arr = cur_arr.copy()
                arr[l], arr[m] = arr[m], arr[l]

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
            out_cif = lammps_data_to_cif([f'{s}' for s in range(n_samples)], lammps_dir, relaxed_lammps_dir,
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
            best_step_idx = None
            filtered_step_data = filter_step_data(step_data, conditions)
            for l in range(len(filtered_step_data['elastic_vectors'])):
                if filtered_step_data['elastic_vectors'][l] > best_step_prop:
                    best_step_prop = filtered_step_data['elastic_vectors'][l]
                    best_step_idx = filtered_step_data['index'][l]
                    print(f'step: {j}, best_step_prop: {best_step_prop}')

            update_the_datapoint = best_step_prop > best_prop and best_step_idx is not None
            if update_the_datapoint:
                add_to_dataset = True  # if any improvement was made during any iteration, we add to the dataset
                patience = 0
                best_cif = out_cif[best_step_idx]
                atoms = read(os.path.join(relaxed_cif_dir, str(best_step_idx) + '.cif'), format="cif")
                best_prop = best_step_prop
                new_datapoint['formation_energy_per_atom'] = final_energies[best_step_idx]
                new_datapoint['ealstic_vector'] = best_prop
                new_datapoint['cif'] = best_cif

            else:
                patience += 1
                if patience > 1000:
                    break

            result_record[i]['best_cifs'].append(best_cif)
            result_record[i]['best_props'].append(best_prop)
        result_record[i]['source_datapoint'] = i
        result_record[i]['added_to_dataset'] = add_to_dataset

        assert (best_prop > starting_prop) == add_to_dataset, 'Error in the implementation - best_prop > starting prop'\
            ' should imply adding to the dataset'
        # because best_prop starts as == starting_prop and the point is added to dataset if it improves at least once

        if add_to_dataset:
            cifs.append(best_cif)
            props.append(best_prop)
            datapoint_indices.append(ind)
            new_datapoints.append(new_datapoint)

    mean_prop = []
    mean_prop.append(np.mean([result_record[k]['starting_prop'] for k in np.arange(n_examples)]))
    for s in range(n_steps):
        mean_prop.append(np.mean([result_record[k]['best_props'][s] for k in np.arange(n_examples)]))
    result_record['mean_prop'] = mean_prop
    result_record['datapoint_indices'] = datapoint_indices
    df = pd.DataFrame.from_records(new_datapoints)
    df.to_csv(os.path.join(out_directory, 'dataset.csv'), index=True)
    torch.save(result_record, os.path.join(out_directory, 'optimization_record'))


if __name__ == '__main__':
    out_file = torch.load(
        'C:\\Users\\GrzegorzKaszuba\\PycharmProjects\\cdvae-main\\hydra\\singlerun\\2023-07-15\\cdvae_new_elastic\\retrain_exp2_max0p9\\3\\lammps_results')
    experiment_path = os.path.join(os.getcwd(), 'localsearch_experiment')
    os.makedirs(experiment_path, exist_ok=True)
    input_cif = out_file['cif_lmp'][35]

    with open('subprocess_calls.yaml', 'r') as file:
        lmp_path = yaml.safe_load(file).get('lammps_call')
        if type(lmp_path) == str:
            lmp_path = [lmp_path]
    lammps_cfg = {
        'lammps_path': lmp_path,
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
            relaxed_lammps_dir = os.path.join(step_dir, 'relaxed_lammps')
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
            atoms = read(os.path.join(relaxed_cif_dir, str(best_cif) + '.data.cif'), format="cif")
            print(f'init: {i}, step: {j}, best_step: {best_cif}, best_prop: {best_prop}')
