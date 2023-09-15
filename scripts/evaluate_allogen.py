import time
import argparse
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace

from eval_utils import load_model, load_model_full, tensors_to_structures, load_model_data
from visualization_utils import save_scatter_plot, cif_names_list, extract_atom_counts, plot_atom_ratios_mpltern
from amir_lammps import lammps_pot, lammps_in, convert_cif_to_lammps, lammps_data_to_cif, lmp_energy_calculator, \
    lmp_elastic_calculator

import os
import re
from torch.utils.tensorboard import SummaryWriter
from pymatgen.io.cif import CifWriter
import shutil
import yaml

import math
from statistics import mean

from evaluate import reconstructon, generation, optimization
from cdvae.pl_data.dataset import AdHocCrystDataset
from composition_rank import DataTriangle

import numpy as np
import subprocess
import tempfile
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset
from torch_geometric.data import DataLoader
import copy

from pymatgen.io.cif import CifParser
from pymatgen.io.lammps.data import LammpsData
from pymatgen.core.periodic_table import Element

from typing import List
import tempfile
from ase.io import read

from conditions import Condition, ZLoss, filter_step_data
from mutations import Transposition, expand_dataset


def log_step_data(writer, filtered_step_data, step):
    for i in range(len(filtered_step_data['elastic_vectors'])):
        writer.add_scalar(f'property fc/case {filtered_step_data["index"][i]}',
                          filtered_step_data['fc_properties'][i], step)
        writer.add_scalar(f'property lammps/case {filtered_step_data["index"][i]}',
                          filtered_step_data['elastic_vectors'][i], step)
        writer.add_scalar(f'property fc-lammps error/case {filtered_step_data["index"][i]}',
                          filtered_step_data['fc_errors'][i], step)
        writer.add_scalar(f'final energy/case {filtered_step_data["index"][i]}',
                          filtered_step_data['final_energies'][i], step)
        writer.add_scalar(f'initial energy/case {filtered_step_data["index"][i]}',
                          filtered_step_data['initial_energies'][i], step)


    if len(filtered_step_data['fc_properties']) > 0:
        writer.add_scalar(f'mean_property/fc', mean(filtered_step_data['fc_properties']), step)
        writer.add_scalar(f'mean_property/lammps', mean(filtered_step_data['elastic_vectors']), step)
        writer.add_scalar(f'mean_property/fc-lammps error (abs)', mean(filtered_step_data['fc_errors']),
                          step)
        writer.add_scalar(f'mean_property/final energy', mean(filtered_step_data['final_energies']), step)
        writer.add_scalar(f'mean_property/initial energy', mean(filtered_step_data['initial_energies']), step)


def update_best_by_composition(best_by_composition, filtered_step_data):
    for i in range(len(filtered_step_data['elastic_vectors'])):
        if filtered_step_data['summary_formulas'][i] in best_by_composition.keys():
            if filtered_step_data['elastic_vectors'][i] > best_by_composition[
                filtered_step_data['summary_formulas'][i]]:
                best_by_composition[filtered_step_data['summary_formulas'][i]] = filtered_step_data['elastic_vectors'][
                    i]
        else:
            best_by_composition[filtered_step_data['summary_formulas'][i]] = filtered_step_data['elastic_vectors'][i]

def get_trainer(cfg, dirpath, epochs=50):
    trainer_args_dict = vars(cfg.train.pl_trainer).copy()

    # Remove keys starting with underscore
    keys_to_remove = [key for key in trainer_args_dict if key.startswith('_')]
    for key in keys_to_remove:
        trainer_args_dict.pop(key)

    # Modify the max_epochs and default_root_dir parameters
    trainer_args_dict['max_epochs'] = epochs
    trainer_args_dict['default_root_dir'] = dirpath

    # Now, we can pass all other parameters from cfg. For example:
    trainer_args_dict['logger'] = True
    trainer_args_dict['callbacks'] = None

    trainer_args_dict['deterministic'] = cfg.train.deterministic
    trainer_args_dict['check_val_every_n_epoch'] = cfg.logging.val_check_interval
    trainer_args_dict['progress_bar_refresh_rate'] = cfg.logging.progress_bar_refresh_rate
    trainer_args_dict['resume_from_checkpoint'] = None
    if torch.cuda.is_available():
        trainer_args_dict['gpus'] = -1
    # Pass the modified dict to the Trainer initialization
    trainer = pl.Trainer(**trainer_args_dict)
    trainer.checkpoint_callback.monitor = 'val_loss'
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            callback.dirpath = dirpath
    return trainer


def initialize_exp_cfg(atomic_symbols: List[str]) -> dict:
    atomic_numbers = [Element(sym).Z for sym in atomic_symbols]
    atom_indices = [z-1 for z in atomic_numbers]
    return {'atomic_symbols': atomic_symbols,
            'atomic_numbers': atomic_numbers,
            'atom_indices': atom_indices}


def read_cif_from_strings(cif_strings):
    structures = []
    for cif_string in cif_strings:
        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w") as temp_file:
            temp_file.write(cif_string)
            temp_file.flush()  # make sure data is written to disk
            structure = read(temp_file.name)
            structures.append(structure)
    return structures

def merge_datasets_cryst(dataset1, dataset2):
    # Merge the two lists
    new_dataset = copy.deepcopy(dataset1)
    new_dataset.cached_data = copy.deepcopy(new_dataset.cached_data + dataset2.cached_data)
    return new_dataset



def dictionary_cat(d, dim=0):
    for k, v in d.items():
        if type(v) == list:
            if type(v[0]) == torch.Tensor:
                d[k] = torch.cat(v, dim=dim)


def optimization_by_batch(model, ld_kwargs, exp_cfg, batch,
                          num_starting_points=100, num_gradient_steps=5000,
                          lr=1e-3, num_saved_crys=10, extra_returns=False, extra_breakpoints=(),
                          maximize=False, z_losses=()):
    if batch is not None:
        batch = batch.to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True

    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    if num_saved_crys > 1:
        interval = num_gradient_steps // (num_saved_crys - 1)
    else:
        interval = -1
    if extra_returns:
        z_list = []
        properties_list = []
        breakpoints = []
        cbf_storage = [None]
        cbf_list = []
        fc_comp = []

        def cbf_hook_fn(module, input, output):
            cbf_storage[0] = module.current_cbf

        cbf_hook = model.decoder.gemnet.register_forward_hook(cbf_hook_fn)

    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        if maximize:
            loss = -model.fc_property(z).mean()
        else:
            loss = model.fc_property(z).mean()
        for z_loss in z_losses:
            loss = loss + z_loss(z)
        loss.backward()
        opt.step()
        #if extra_returns:
        #    recent_z = z.detach().cpu()
        #    recent_property = model.inverse_transform(model.fc_property(z)).detach().cpu()

        if (i % interval == 0 and interval != -1) or i == (num_gradient_steps - 1) or i in extra_breakpoints:

            crystals = model.langevin_dynamics(z, ld_kwargs)
            if extra_returns:
                recent_z = z.detach().cpu()
                recent_property = model.scaler.inverse_transform(model.fc_property(z)).detach().cpu()
                breakpoints.append(i)
                z_list.append(recent_z)
                properties_list.append(recent_property)
                fc_comp.append(F.softmax(model.fc_composition(z).detach()).cpu()[:, exp_cfg['atom_indices']])

                cbf_list.append(cbf_storage[0])
            all_crystals.append(crystals)
    if extra_returns:
        z_list = torch.stack(z_list, dim=0)
        properties_list = torch.stack(properties_list, dim=0)
        cbf_list = torch.stack(cbf_list, dim=0)
        fc_comp = torch.stack(fc_comp, dim=0)
        # breakpoints = torch.tensor(breakpoints)
        return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
                ['frac_coords', 'atom_types', 'num_atoms', 'lengths',
                 'angles']}, z_list, properties_list, breakpoints, cbf_list, fc_comp

    return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}


def opt_chunk_generator(data, chunk_size):
    # Create a deep copy of data and move tensors to CPU
    data = {key: value.to('cpu').clone() for key, value in data.items()}

    # Start indices for the chunks
    start_indices = torch.arange(0, data['num_atoms'].numel(), chunk_size)
    # Ensure that the last chunk includes all remaining graphs
    if start_indices[-1] != data['num_atoms'].numel():
        start_indices = torch.cat([start_indices, torch.tensor([data['num_atoms'].numel()])])

    # Compute the end indices for the node attribute chunks
    end_indices_atoms = torch.cat([torch.tensor([0]), torch.cumsum(data['num_atoms'].flatten(), dim=0)])

    for start, end in zip(start_indices[:-1], start_indices[1:]):
        chunk = {}
        chunk['lengths'] = data['lengths'][:, start:end]
        chunk['angles'] = data['angles'][:, start:end]
        chunk['num_atoms'] = data['num_atoms'][:, start:end]

        # Calculate start and end indices for the node attributes
        start_atoms = end_indices_atoms[start].item()
        end_atoms = end_indices_atoms[end].item()
        chunk['frac_coords'] = data['frac_coords'][:, start_atoms:end_atoms]
        chunk['atom_types'] = data['atom_types'][:, start_atoms:end_atoms]

        yield chunk


def test_prop_fc(model, ld_kwargs, data_loader,
                 num_starting_points=100, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=10):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True

    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    interval = num_gradient_steps // (num_saved_crys - 1)
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        loss = model.fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps - 1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    start_time = time.time()
    prop_model = None

    if args.prop_model_path:
        prop_model_path = Path(args.prop_model_path)
        prop_model, _, prop_cfg = load_model(prop_model_path)
        prop_model.to('cuda')
        print('prop_model_parameters', count_parameters(prop_model))

    with open('subprocess_calls.yaml', 'r') as file:
        lmp_path = yaml.safe_load(file).get('lammps_call')

    lammps_cfg = {
        'lammps_path': lmp_path,
        'pot_file': lammps_pot,
        'input_template': lammps_in,
        'pot': 'eam/alloy'}

    exp_cfg = initialize_exp_cfg(['Cr', 'Fe', 'Ni'])

    model_path = Path(args.model_path)


    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)


    writer_path = os.path.join(model_path, 'logs')

    # write dummy figure to tensorboard

    if 'localsearch_dataset' in args.tasks:
        # ------------------- Dataset retrieval ---------------------
        loaders, cfg = load_model_full(model_path)
        train = loaders[0].dataset
        val = loaders[1].dataset
        test = loaders[2].dataset

        # ---------------------- Path setup -------------------------
        path_out = os.path.join(model_path, f'localsearch_train_set_{args.label}')
        new_dataset_path = os.path.join(path_out, 'new_data')
        new_train_path, new_val_path, new_test_path = [os.path.join(new_dataset_path, dset) for dset in ['train', 'val', 'test']]
        #combined_dataset_path = os.path.join(path_out, 'combined_data')
        #combined_train_path, combined_val_path, combined_test_path = [os.path.join(combined_dataset_path, dset) for dset in ['train', 'val', 'test']]

        # ------------------- Localsearch with LAMMPS ---------------
        # localsearch params
        n_steps = 20 # this is how many steps the algorithm will do
        n_samples = 20 # this is how many transpositions the algorithm will check before picking the best one
        # note: it's not necessary that the highest n_samples translates to the best result - greed doesn't always pay

        datasets = [train, val, test]
        directories = [new_train_path, new_val_path, new_test_path]
        dataset_names = ['train', 'val', 'test']

        for dataset, out_directory, dataset_name in zip(datasets, directories, dataset_names):
            os.makedirs(out_directory, exist_ok=True)
            expand_dataset(dataset, out_directory, lammps_cfg, property_name='ealstic_vector', n_steps=n_steps, n_samples=n_samples, n_examples=math.ceil(len(dataset)*0.1))
            shutil.copy(os.path.join(out_directory, 'dataset.csv'), os.path.join(new_dataset_path, dataset_name+'.csv'))




    if 'opt_retrain' in args.tasks:
        model, loaders, cfg = load_model_full(model_path)
        model.to('cuda')
        best_by_composition = {}
        path_out = os.path.join(model_path, f'retrain_{args.label}')
        writer = SummaryWriter(path_out)
        num_steps = 10

        (niggli, primitive, graph_method, preprocess_workers, lattice_scale_method) = (
            cfg.data.datamodule.datasets.train.niggli,
            cfg.data.datamodule.datasets.train.primitive,
            cfg.data.datamodule.datasets.train.graph_method,
            cfg.data.datamodule.datasets.train.preprocess_workers,
            cfg.data.datamodule.datasets.train.lattice_scale_method
        )

        cur_train_loader = copy.deepcopy(loaders[0])
        cur_val_loader = copy.deepcopy(loaders[1])
        cur_structure_loader = copy.deepcopy(loaders[2])
        os.makedirs(path_out, exist_ok=True)



        # --------------------------- trainer setup -----------------------------
        # Convert Namespace to dict
        trainer = get_trainer(cfg, os.path.join(path_out, 'initial_training'))


        for step in range(num_steps):
            # ------------------ Cycle setup ------------------
            n_structures_retrain = 0
            stepdir = os.path.join(path_out, str(step))
            cif_dir = os.path.join(stepdir, 'cif_raw')
            lammpsdata_dir = os.path.join(stepdir, 'lammps_raw')
            relaxed_cif_dir = os.path.join(stepdir, 'cif_relaxed')
            relaxed_lammpsdata_dir = os.path.join(stepdir, 'lammps_relaxed')
            chkdir = os.path.join(stepdir, 'checkpoints')
            pt_dir = os.path.join(stepdir, 'pt')
            for directory in [stepdir, cif_dir, lammpsdata_dir, relaxed_cif_dir, relaxed_lammpsdata_dir, chkdir, pt_dir]:
                os.makedirs(directory, exist_ok=True)


            exp_cfg = initialize_exp_cfg(['Cr', 'Fe', 'Ni'])


            train_properties = loaders[0].dataset.get_properties()
            train_compositions = loaders[0].dataset.get_compositions(exp_cfg['atomic_numbers'])
            for prop, comp in zip(train_properties, train_compositions):
                if comp not in best_by_composition.keys():
                    best_by_composition[comp] = prop
                else:
                    if prop > best_by_composition[comp]:
                        best_by_composition[comp] = prop


            cr_triangle = DataTriangle()
            cr_triangle.triangulate_data_dict(best_by_composition)
            cr_triangle.plot(savedir=path_out, label='_initial')

            def get_improvement_properties(train_compositions, train_properties, data_triangle):
                improvement_properties = []
                for comp, prop in zip(train_compositions, train_properties):
                    improvement_properties.append((prop-data_triangle.get_value(comp)).item())
                return improvement_properties

            cr_triangle.triangulate_data_dict(best_by_composition)
            cr_triangle.plot(savedir=path_out, label='_initial')
            new_props = get_improvement_properties(train_compositions, train_properties, cr_triangle)
            loaders[0].dataset.relabel(new_labels=new_props, model=model)

            model.calibrate()
            model.is_calibrating = True
            trainer.fit(model, train_dataloader=loaders[1], val_dataloaders=loaders[2])
            model.is_calibrating = False
            model, _, _ = load_model(model_path)
            # Load the best checkpoint
            best_model_path = trainer.checkpoint_callback.best_model_path
            #if best_model_path:

            model.load_state_dict(torch.load(best_model_path)['state_dict'])
            model.to('cuda')

            #timer['retrain'], cur_time = timer['retrain'] + time.time() - cur_time, time.time()

            output_dict = {'eval_setting': args,
                           'frac_coords': [],
                           'num_atoms': [],
                           'atom_types': [],
                           'lengths': [],
                           'angles': [],
                           'ld_kwargs': ld_kwargs,
                           'z': [],
                           'cif': [],
                           'fc_properties': [],
                           'fc_comp': [],
                           'cbf': []}

            # ------------------ Optimization step ------------------------------
            all_fc_properties = []
            all_fc_comp = []
            starting_data = []
            for n, batch in enumerate(cur_structure_loader):
                if n > 1:
                    continue
                if step == 0:
                    starting_data.append('take element comp and label from batch')
                opt_out, z, fc_properties, optimization_breakpoints, cbf, fc_comp =\
                    optimization_by_batch(model, ld_kwargs, exp_cfg, batch,
                                          num_starting_points=100, num_gradient_steps=500, lr=1e-3, num_saved_crys=1,
                                          extra_returns=True, maximize=True)

                n_generated_structures = batch.num_atoms.cpu().shape[0]
                chonker = opt_chunk_generator(opt_out, n_generated_structures)

                batch_output = {'frac_coords': [],
                                'num_atoms': [],
                                'atom_types': [],
                                'lengths': [],
                                'angles': [],
                                'z': z,
                                'fc_properties': fc_properties,
                                'fc_comp': fc_comp,
                                'cif': [],
                                'cbf': cbf}

                all_fc_properties.append(fc_properties)
                all_fc_comp.append(fc_comp)
                for chunk, bp in zip(chonker, optimization_breakpoints):
                    step_output = {
                        'frac_coords': chunk['frac_coords'],
                        'num_atoms': chunk['num_atoms'],
                        'atom_types': chunk['atom_types'],
                        'lengths': chunk['lengths'],
                        'angles': chunk['angles'],
                        'cif': [],
                    }

                    # ----------------------- Writing structures --------------------------------

                    structure_objects = tensors_to_structures(chunk['lengths'][0], chunk['angles'][0],
                                                              chunk['frac_coords'][0],
                                                              chunk['atom_types'][0], chunk['num_atoms'][0])

                    for j, structure in enumerate(structure_objects):
                        # Write structure to CIF file
                        structure.to(
                            filename=os.path.join(cif_dir, f'generated{j + n_structures_retrain}_step{bp+step*(bp+1)}.cif'),
                            fmt='cif')

                        cif_writer = CifWriter(structure)
                        cif_string = cif_writer.__str__()
                        step_output['cif'].append(cif_string)

                    for k, v in step_output.items():
                        batch_output[k].append(step_output[k])

                for k, v in batch_output.items():
                    output_dict[k].append(batch_output[k])
                n_structures_retrain += n_generated_structures
                # --------------------- Batch end ----------------------

            torch.save(output_dict, os.path.join(pt_dir, 'data.pt'))

            # ---------------------- CIF ---------------------
            cif_data = []
            for batch_cif in output_dict['cif']:
                for cif_str in batch_cif[0]:
                    cif_data.append(cif_str)
            # ------------------- LAMMPS --------------------

            convert_cif_to_lammps(cif_dir, lammpsdata_dir)
            initial_energies, final_energies = lmp_energy_calculator(lammpsdata_dir, relaxed_lammpsdata_dir, lammps_cfg, silent=True)
            elastic_vectors = lmp_elastic_calculator(lammpsdata_dir, lammps_cfg, silent=True)

            # sort examples alphanumerically (default alphabetically)
            structure_names = [n.split('.')[0] for n in
                               sorted(os.listdir(relaxed_lammpsdata_dir), key=lambda s: int(re.search('\d+', s).group()))]


            cif_lmp = lammps_data_to_cif([s_name for s_name in structure_names], lammpsdata_dir,
                                         relaxed_lammpsdata_dir, savedir=relaxed_cif_dir)


            #sort outputs by alphanumeric order
            initial_energies = [initial_energies[name] for name in structure_names]
            final_energies = [final_energies[name] for name in structure_names]
            elastic_vectors = [elastic_vectors[name] for name in structure_names]

            prop_lmp = [el[3] for el in elastic_vectors]
            summary_formulas = [extract_atom_counts(os.path.join(relaxed_cif_dir, s + '.cif'),
                                                    exp_cfg['atomic_symbols']) for s in structure_names]
            lammps_results = {'initial_energies': torch.tensor(initial_energies),
                              'final_energies': torch.tensor(final_energies),
                              'prop': torch.tensor(prop_lmp),
                              'cif_lmp': cif_lmp,
                              'summary_formulas': summary_formulas}
            torch.save(lammps_results, os.path.join(pt_dir, 'lammps_results.pt'))
            # ------------- Step logging ----------------
            step_fc_properties = []
            for fc in all_fc_properties:
                step_fc_properties = step_fc_properties + fc.reshape(-1).tolist()
            step_fc_comp = []
            for fc in all_fc_comp:
                step_fc_comp = step_fc_comp + fc.reshape(-1, 3).tolist()
            fc_errors = [abs(step_fc_properties[i]-prop_lmp[i]) for i in range(len(step_fc_properties))]
            step_data = {'fc_properties': step_fc_properties,
                         'fc_comp': step_fc_comp,
                         'initial_energies': initial_energies,
                         'final_energies': final_energies,
                         'structure_names': structure_names,
                         'summary_formulas': summary_formulas,
                         'elastic_vectors': prop_lmp,
                         'index': list(range(len(prop_lmp))),
                         'fc_errors': fc_errors}

            conditions = [Condition('final_energies', -250, -200),
                          Condition('elastic_vectors', 0, 500),
                          Condition('summary_formulas', 0, 0.9)]

            filtered_step_data = filter_step_data(step_data, conditions)
            log_step_data(writer, filtered_step_data, step)
            update_best_by_composition(best_by_composition, filtered_step_data)
            plot_atom_ratios_mpltern(filtered_step_data['summary_formulas'],
                                     property=filtered_step_data['elastic_vectors'],
                                     save_label=os.path.join(path_out, f'tri_step {step}'))


            # ------------------------ Recalibration ------------------------

            trainer = get_trainer(cfg, os.path.join(path_out, 'initial_training'))
            cr_triangle = DataTriangle()
            cr_triangle.triangulate_data_dict(best_by_composition)
            cr_triangle.plot(savedir=path_out, label=str(step))
            new_props = get_improvement_properties(train_compositions, train_properties, cr_triangle)
            print('improvement_properties:', new_props)
            loaders[0].dataset.relabel(new_labels=new_props, model=model)



            model.calibrate()
            model.is_calibrating = True
            trainer.fit(model, train_dataloader=cur_train_loader, val_dataloaders=cur_val_loader)
            model.is_calibrating = False # todo in order to recalibrate, dataset must be reinitialized (perhaps even datamodule -> new scaler)
                                            # todo the property module should perhaps be replaced thoroughly - completely new scaler and data distribution


            # Load the best checkpoint
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path:
                model.load_state_dict(torch.load(best_model_path)['state_dict'])

            torch.save(step_data, os.path.join(pt_dir, 'full_batch.pt'))
            torch.save(filtered_step_data, os.path.join(pt_dir, 'filtered_batch.pt'))
            # fc_properties, initial_energies, finala_energies, stepdir, structure_names

            # --------------- New datasets --------------------
            new_dataset = AdHocCrystDataset('identity_test_dataset', cif_lmp, prop_lmp, niggli, primitive,
                                            graph_method, preprocess_workers, lattice_scale_method,
                                            prop_name='ealstic_vector',
                                            scaler=loaders[0].dataset.scaler,
                                            lattice_scaler=loaders[0].dataset.lattice_scaler)

            new_train, new_val, new_structures = [copy.deepcopy(new_dataset) for i in
                                                  range(3)]  # this is so that the model fits closely to new examples
            # the accuracy growth is visible in val, and the new examples are further optimize with that fitting finished


            # --------------- Merged loaders ------------------

            cur_train_loader, cur_val_loader, cur_structure_loader = (
                DataLoader(merge_datasets_cryst(cur_train_loader.dataset, new_train),
                           batch_size=cur_train_loader.batch_size, shuffle=True,
                           num_workers=loaders[0].num_workers),
                copy.deepcopy(loaders[1]),
                DataLoader(new_structures, batch_size=cur_train_loader.batch_size, shuffle=False,
                           num_workers=loaders[0].num_workers)
            )

            # --------------- Retraining ----------------------
            #model.unfreeze()
            #trainer.fit(model, train_dataloader=cur_train_loader, val_dataloaders=cur_val_loader)

            # Load the best checkpoint
            #best_model_path = trainer.checkpoint_callback.best_model_path
           #model.load_state_dict(torch.load(best_model_path))

        final_comps = []
        final_props = []
        for k, v in best_by_composition.items():
            final_comps.append(k)
            final_props.append(v)
        plot_atom_ratios_mpltern(final_comps,
                                 final_props,
                                 save_label=os.path.join(path_out, f'tri final'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--prop_model_path', default='')
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=30, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=10, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')

    args = parser.parse_args()
    print('Tasks:', args.tasks)
    print('Path to model:', args.model_path)
    main(args)
