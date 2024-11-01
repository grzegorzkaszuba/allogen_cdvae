import time
import argparse
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace

from eval_utils import load_model, load_model_full, tensors_to_structures, load_model_data
from visualization_utils import save_scatter_plot, cif_names_list, extract_atom_counts
from visualization_utils import plot_atom_ratios_mpltern
from amir_lammps import lammps_pot, lammps_in, convert_cif_to_lammps, lammps_data_to_cif, lmp_energy_calculator, \
    lmp_elastic_calculator

import os
import re
from torch.utils.tensorboard import SummaryWriter
from pymatgen.io.cif import CifWriter
import shutil
import yaml
from pathlib import Path

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


def double_stack(nested_list2, dim1=None, dim2=None):
    if dim1 is None:
        unpacked1 = [torch.stack(nested_list2[i]) for i in range(len(nested_list2))]
    else:
        unpacked1 = [torch.cat(nested_list2[i], dim=dim1) for i in range(len(nested_list2))]
    if dim2 is None:
        unpacked2 = torch.stack(unpacked1)
    else:
        unpacked2 = torch.cat(unpacked1, dim=dim2)
    return unpacked2


def stack_output_batches(out_dict):
    new_dict = {}
    new_dict['eval_setting'] = out_dict['eval_setting']
    new_dict['frac_coords'] = double_stack(out_dict['frac_coords'], dim1=0, dim2=1)
    new_dict['num_atoms'] = double_stack(out_dict['num_atoms'], dim1=0, dim2=1)
    new_dict['atom_types'] = double_stack(out_dict['atom_types'], dim1=0, dim2=1)
    new_dict['lengths'] = double_stack(out_dict['lengths'], dim1=0, dim2=1)
    new_dict['angles'] = double_stack(out_dict['angles'], dim1=0, dim2=1)
    new_dict['ld_kwargs'] = out_dict['ld_kwargs']
    new_dict['z'] = torch.cat(out_dict['z'], dim=1)
    new_dict['cif'] = out_dict['cif']
    new_dict['fc_properties'] = torch.cat(out_dict['fc_properties'], dim=1)
    new_dict['fc_comp'] = torch.cat(out_dict['fc_comp'], dim=1)

def merge_datasets_cryst(dataset1, dataset2):
    # Merge the two lists
    new_dataset = copy.deepcopy(dataset1)
    new_dataset.cached_data = copy.deepcopy(new_dataset.cached_data + dataset2.cached_data)
    return new_dataset
def initialize_exp_cfg(atomic_symbols: List[str]) -> dict:
    atomic_numbers = [Element(sym).Z for sym in atomic_symbols]
    atom_indices = [z-1 for z in atomic_numbers]
    return {'atomic_symbols': atomic_symbols,
            'atomic_numbers': atomic_numbers,
            'atom_indices': atom_indices}

def optimization_by_batch(model, ld_kwargs, exp_cfg, batch,
                          num_starting_points=1000, num_gradient_steps=5000,
                          lr=1e-3, num_saved_crys=10, extra_returns=False, extra_breakpoints=(),
                          maximize=False, z_losses=()):
    if batch is not None:
        batch = batch.to(model.device)
        _, _, z = model.encode(batch)
        z = z.detach().clone()
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


class StructureOptimizer:
    CONDITIONS = [Condition('final_energies', -250, -200),
                  Condition('elastic_vectors', 0, 500),
                  Condition('summary_formulas', 0, 0.9)]
    def __init__(self, cfg, lammps_cfg, ld_kwargs, args, path_out, loaders, num_steps=10, elements=('Cr', 'Fe', 'Ni')):
        self.cfg = cfg
        self.lammps_cfg = lammps_cfg
        self.best_by_composition = {}
        self.path_out = path_out
        self.ld_kwargs = ld_kwargs
        self.writer = SummaryWriter(path_out)
        self.num_steps = num_steps
        self.exp_cfg = initialize_exp_cfg(elements)
        self.args = args


        self.current_step = 0

        self.cur_train_loader = copy.deepcopy(loaders[0])
        self.cur_val_loader = copy.deepcopy(loaders[1])
        self.cur_structure_loader = copy.deepcopy(loaders[2])
        os.makedirs(path_out, exist_ok=True)
        self.capture_cfg(cfg)
        # to add outside
        self.capture_best_from_loader(loaders[2])
        self.get_data_triangle(self.path_out)


    def step(self, model, loader):
        # Cycle Setup
        existing_structures = 0
        stepdir = os.path.join(self.path_out, str(self.current_step))
        cif_dir = os.path.join(stepdir, 'cif_raw')
        chkdir = os.path.join(stepdir, 'checkpoints')
        pt_dir = os.path.join(stepdir, 'pt')
        for directory in [stepdir, cif_dir, chkdir, pt_dir]:
            os.makedirs(directory, exist_ok=True)

        output_dict = {'eval_setting': self.args,
                       'frac_coords': [],
                       'num_atoms': [],
                       'atom_types': [],
                       'lengths': [],
                       'angles': [],
                       'ld_kwargs': self.ld_kwargs,
                       'z': [],
                       'cif': [],
                       'fc_properties': [],
                       'fc_comp': [],
                       'cbf': []}

        all_fc_properties = []
        all_fc_comp = []
        starting_data = []
        for n, batch in enumerate(loader):
            batch_output, n_generated_structures, breakpoints = self.batch_step(n, batch, cif_dir, model, existing_structures)
            all_fc_properties.append(batch_output['fc_properties'])
            all_fc_comp.append(batch_output['fc_comp'])
            for k, v in batch_output.items():
                output_dict[k].append(batch_output[k])
            existing_structures += n_generated_structures

        #compact_dict = stack_output_batches(output_dict)

        torch.save(output_dict, os.path.join(pt_dir, 'data.pt'))
        #torch.save(compact_dict, os.path.join(pt_dir, 'data2.pt'))

        #output_dict = stack_output_batches(output_dict)

        self.calculate_lammps(stepdir, all_fc_properties, all_fc_comp)


    def calculate_lammps(self, stepdir, all_fc_properties, all_fc_comp):
        # directory setup

        cif_dir = os.path.join(stepdir, 'cif_raw')
        lammpsdata_dir = os.path.join(stepdir, 'lammps_raw')
        relaxed_cif_dir = os.path.join(stepdir, 'cif_relaxed')
        relaxed_lammpsdata_dir = os.path.join(stepdir, 'lammps_relaxed')
        pt_dir = os.path.join(stepdir, 'pt')

        for directory in [lammpsdata_dir, relaxed_cif_dir, relaxed_lammpsdata_dir]:
            os.makedirs(directory, exist_ok=True)
        # ------------------- LAMMPS --------------------

        convert_cif_to_lammps(cif_dir, lammpsdata_dir)
        initial_energies, final_energies = lmp_energy_calculator(lammpsdata_dir, relaxed_lammpsdata_dir, self.lammps_cfg,
                                                                 silent=True)
        elastic_vectors = lmp_elastic_calculator(lammpsdata_dir, self.lammps_cfg, silent=True)

        # sort examples alphanumerically (default alphabetically)
        structure_names = [n.split('.')[0] for n in
                           sorted(os.listdir(relaxed_lammpsdata_dir), key=lambda s: int(re.search('\d+', s).group()))]

        cif_lmp = lammps_data_to_cif([s_name for s_name in structure_names], lammpsdata_dir,
                                     relaxed_lammpsdata_dir, savedir=relaxed_cif_dir)

        # sort outputs by alphanumeric order
        initial_energies = [initial_energies[name] for name in structure_names]
        final_energies = [final_energies[name] for name in structure_names]
        elastic_vectors = [elastic_vectors[name] for name in structure_names]

        prop_lmp = [el[3] for el in elastic_vectors]
        summary_formulas = [extract_atom_counts(os.path.join(relaxed_cif_dir, s + '.cif'),
                                                self.exp_cfg['atomic_symbols']) for s in structure_names]
        lammps_results = {'initial_energies': torch.tensor(initial_energies),
                          'final_energies': torch.tensor(final_energies),
                          'prop': torch.tensor(prop_lmp),
                          'cif_lmp': cif_lmp,
                          'summary_formulas': summary_formulas}
        torch.save(lammps_results, os.path.join(pt_dir, 'lammps_results.pt'))
        # ------------- Step logging ----------------
        step_fc_properties = torch.cat(all_fc_properties, dim=1)[:, :, 0].T.reshape(-1).tolist()



        fc_errors = [abs(step_fc_properties[i] - prop_lmp[i]) for i in range(len(step_fc_properties))]
        step_data = {'fc_properties': step_fc_properties,
                     'initial_energies': initial_energies,
                     'final_energies': final_energies,
                     'structure_names': structure_names,
                     'summary_formulas': summary_formulas,
                     'elastic_vectors': prop_lmp,
                     'index': list(range(len(prop_lmp))),
                     'fc_errors': fc_errors}

        torch.save(step_data, os.path.join(pt_dir, 'full_batch.pt'))

        filtered_step_data = filter_step_data(step_data, self.CONDITIONS)
        torch.save(filtered_step_data, os.path.join(pt_dir, 'filtered_batch.pt'))
        self.capture_best_from_step_data(filtered_step_data)

        plot_atom_ratios_mpltern(filtered_step_data['summary_formulas'],
                                 property=filtered_step_data['elastic_vectors'],
                                 save_label=os.path.join(self.path_out, f'tri_step {self.current_step}'))


        log_step_data(self.writer, filtered_step_data, self.current_step)

        self.current_step += 1
        #recalibration reimplemented

    def batch_step(self, n, batch, cif_dir, model, existing_structures):
        #if n > 1:
            #continue
        #if step == 0:
            #starting_data.append('take element comp and label from batch')
        step = self.current_step
        opt_out, z, fc_properties, optimization_breakpoints, cbf, fc_comp = \
            optimization_by_batch(model, self.ld_kwargs, self.exp_cfg, batch,
                                  num_starting_points=1000, num_gradient_steps=5000, lr=1e-3, num_saved_crys=1,
                                  extra_returns=True, maximize=True, extra_breakpoints=(200, 500, 1000, 1500, 2500, 3500, 5000))

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


        for chunk, bp in zip(chonker, optimization_breakpoints):
            step_output = {
                'frac_coords': chunk['frac_coords'],
                'num_atoms': chunk['num_atoms'],
                'atom_types': chunk['atom_types'],
                'lengths': chunk['lengths'],
                'angles': chunk['angles'],
                'cif': [],
            }

            structure_objects = tensors_to_structures(chunk['lengths'][0], chunk['angles'][0],
                                                      chunk['frac_coords'][0],
                                                      chunk['atom_types'][0], chunk['num_atoms'][0])

            for j, structure in enumerate(structure_objects):
                # Write structure to CIF file
                structure.to(
                    filename=os.path.join(cif_dir,
                                          f'generated{j + existing_structures}_step{bp+step * (bp +1)}.cif'),
                    fmt='cif')

                cif_writer = CifWriter(structure)
                cif_string = cif_writer.__str__()
                step_output['cif'].append(cif_string)

            for k, v in step_output.items():
                batch_output[k].append(step_output[k])

        #torch.save(output_dict, os.path.join(pt_dir, 'data.pt'))
        return batch_output, n_generated_structures, optimization_breakpoints





    def create_trainer(self, savepath, epochs=200):
        cfg = self.cfg
        trainer_args_dict = vars(cfg.train.pl_trainer).copy()

        # Remove keys starting with underscore
        keys_to_remove = [key for key in trainer_args_dict if key.startswith('_')]
        for key in keys_to_remove:
            trainer_args_dict.pop(key)

        # Modify the max_epochs and default_root_dir parameters
        trainer_args_dict['max_epochs'] = epochs
        trainer_args_dict['default_root_dir'] = savepath

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
                callback.dirpath = savepath
        return trainer

    def capture_best_from_loader(self, loader, reset_best=False):
        # captures best properties from the loader
        if reset_best:
            self.best_by_composition = {}
        train_properties = loader.dataset.get_properties()
        train_compositions = loader.dataset.get_compositions(self.exp_cfg['atomic_numbers'])
        for prop, comp in zip(train_properties, train_compositions):
            if comp not in self.best_by_composition.keys():
                self.best_by_composition[comp] = prop
            else:
                if prop > self.best_by_composition[comp]:
                    self.best_by_composition[comp] = prop

    def capture_best_from_step_data(self, step_data, reset_best=False):
        if reset_best:
            self.best_by_composition = {}
        for i in range(len(step_data['elastic_vectors'])):
            if step_data['summary_formulas'][i] in self.best_by_composition.keys():
                if step_data['elastic_vectors'][i] > self.best_by_composition[step_data['summary_formulas'][i]]:
                    self.best_by_composition[step_data['summary_formulas'][i]] = \
                    step_data['elastic_vectors'][i]
            else:
                self.best_by_composition[step_data['summary_formulas'][i]] = step_data['elastic_vectors'][
                    i]

    def get_data_triangle(self, path=None, plot=True, label='_initial', capture=True):
        # creates a "DataTriangle" for visualization and interpolation of best compositions
        if path is None:
            path = self.path_out
        cr_triangle = DataTriangle()
        cr_triangle.triangulate_data_dict(self.best_by_composition)
        if capture:
            self.data_triangle = cr_triangle
        if plot:
            cr_triangle.plot(savedir=path, label=label)
        return cr_triangle

    def relabel_dataset(self, loader, model):
        new_loader = copy.deepcopy(loader) # gk the copy might slow down the loader!
        cr_triangle = self.get_data_triangle()
        new_props = self.get_improvement_properties(new_loader)
        new_loader.dataset.relabel(new_props, model=model)
        return new_loader


    def calibrate_model(self, model, trainer, train, val, load_best=True):
        model.calibrate()
        model.is_calibrating = True
        trainer.fit(model, train_dataloader=train, val_dataloaders=val)
        if not load_best:
            return
        else:
            a = 5
            ckpts = list(Path(trainer.default_root_dir).glob('*.ckpt'))
            ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            state_dict = torch.load(ckpt)['state_dict']
            model.load_state_dict(state_dict)
    def get_improvement_properties(self, loader, data_triangle=None):
        # takes a loader and computes the labels as improvement from the best compositions in DataTriangle
        if data_triangle is None:
            if hasattr(self, 'data_triangle'):
                data_triangle = self.data_triangle
            else:
                raise AttributeError('get_improvement_properties with data_triangle=None requires the optimizer to have its own data triangle')
        improvement_properties = []
        train_properties = loader.dataset.get_properties()
        train_compositions = loader.dataset.get_compositions(self.exp_cfg['atomic_numbers'])
        for comp, prop in zip(train_compositions, train_properties):
            improvement_properties.append((prop - data_triangle.get_value(comp)).item())
        return improvement_properties

    def capture_cfg(self, cfg):
        (self.niggli, self.primitive, self.graph_method, self.preprocess_workers, self.lattice_scale_method) = (
            cfg.data.datamodule.datasets.train.niggli,
            cfg.data.datamodule.datasets.train.primitive,
            cfg.data.datamodule.datasets.train.graph_method,
            cfg.data.datamodule.datasets.train.preprocess_workers,
            cfg.data.datamodule.datasets.train.lattice_scale_method
        )

    def make_tensor_dataset(self, cif_lmp, prop_lmp, scaler_donor_loader):
        new_dataset = AdHocCrystDataset('identity_test_dataset', cif_lmp, prop_lmp, self.niggli, self.primitive,
                                        self.graph_method, self.preprocess_workers, self.lattice_scale_method,
                                        prop_name='ealstic_vector',
                                        scaler=scaler_donor_loader.dataset.scaler,
                                        lattice_scaler=scaler_donor_loader.dataset.lattice_scaler)

    def merge_loaders(self, loader_old, new_dataset):
        return DataLoader(merge_datasets_cryst(loader_old.dataset, new_dataset),
                       batch_size=loader_old.batch_size, shuffle=True,
                       num_workers=loader_old.num_workers)

