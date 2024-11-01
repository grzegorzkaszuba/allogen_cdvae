import time
import argparse
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace

from eval_utils import load_model, load_model_full, load_tensor_data, prop_model_eval, get_crystals_list, \
    get_cryst_loader, tensors_to_structures
from visualization_utils import save_scatter_plot, cif_names_list, extract_atom_counts, plot_atom_ratios_mpltern
from amir_lammps import lammps_pot, lammps_in, convert_cif_to_lammps, lammps_data_to_cif, lmp_energy_calculator, \
    lmp_elastic_calculator

import os
import re
from torch.utils.tensorboard import SummaryWriter
from pymatgen.io.cif import CifWriter

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
from mutations import Transposition, Transmutation



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


def run_lammps_simulation(cif_str, lammps_path):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as temp_cif:
        temp_cif.write(cif_str.encode())

    # Parse the CIF file with pymatgen
    parser = CifParser(temp_cif.name)
    structure = parser.get_structures()[0]

    # Create a LAMMPS data file
    lammps_data = LammpsData.from_structure(structure)
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as temp_dat:
        lammps_data.write_file(temp_dat.name)

    # Create a LAMMPS input script
    lammps_input = """
    units metal
    atom_style charge
    boundary p p p
    read_data {}
    pair_style lj/cut 2.5
    pair_coeff * * 1.0 1.0
    neighbor 0.3 bin
    neigh_modify delay 0 every 20 check no
    fix 1 all nve
    run 1000
    """.format(temp_dat.name)
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False, mode="w") as temp_in:
        temp_in.write(lammps_input)

    # Run lammps with the input script
    cmd = f"{lammps_path} < {temp_in.name}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    # Remove temp files
    os.remove(temp_cif.name)
    os.remove(temp_dat.name)
    os.remove(temp_in.name)

    # Return output and error as a dictionary
    return {"stdout": stdout.decode(), "stderr": stderr.decode()}


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

    model_path = Path(args.model_path)

    model, loaders, cfg = load_model_full(model_path)
    model.to('cuda')
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')

    writer_path = os.path.join(model_path, 'logs')

    # write dummy figure to tensorboard

    if 'enc_prop' in args.tasks:
        print('Evaluate model on the encoding -> property prediction pipeline')
        writer = SummaryWriter(os.path.join(writer_path, 'enc_prop'))
        for n, dset in enumerate(['train', 'val', 'test']):
            properties = []
            property_labels = []
            for batch in loaders[n]:
                _, _, z = model.encode(batch.to(torch.device('cuda')))
                property = model.fc_property(z).detach()

                properties.append(property)
                property_labels.append(batch.y.to(torch.device('cuda')))
            properties = torch.cat(properties, dim=0)
            property_labels = torch.cat(property_labels, dim=0)
            loss_pred = torch.nn.functional.mse_loss(properties, property_labels)
            save_scatter_plot(properties.detach().cpu(), property_labels.detach().cpu(), writer,
                              'enc_prop_' + dset + ' mse: ' + str(loss_pred))
        writer.close()

    if 'enc_dec_prop' in args.tasks:
        print(
            'Evaluate the combination of CDVAE and outside prediction module on the task of encoding -> decoding -> property prediction')
        recon_data_path = os.path.join(model_path, 'eval_recon.pt')
        recon_data = torch.load(recon_data_path)
        recon_crystals_list = get_crystals_list(recon_data['frac_coords'][0],
                                                recon_data['atom_types'][0],
                                                recon_data['lengths'][0],
                                                recon_data['angles'][0],
                                                recon_data['num_atoms'][0])
        predictions = prop_model_eval('perovskite', recon_crystals_list)
        properties_dec = torch.tensor(predictions).reshape(-1, 1)

        writer = SummaryWriter(os.path.join(writer_path, 'enc_dec_prop'))
        for n, dset in enumerate(['test']):
            properties = []
            property_labels = []
            for i, batch in enumerate(loaders[2]):
                if torch.cuda.is_available():
                    batch.cuda()
                _, _, z = model.encode(batch)

                properties.append(prop_model(batch).detach().cpu())
                property_labels.append(batch.y.cpu())
            properties = torch.cat(properties, dim=0)
            property_labels = torch.cat(property_labels, dim=0)
            # properties_dec = torch.cat(properties_dec, dim=0)
            loss_pred = torch.nn.functional.mse_loss(properties, property_labels)
            loss_pred_dec = torch.nn.functional.mse_loss(properties_dec, property_labels)
            save_scatter_plot(properties.numpy(), property_labels.detach().numpy(), writer,
                              dset + ' set gemprop mse ' + str(loss_pred.item()), plot_title='Loss from data')
            save_scatter_plot(properties_dec.numpy(), property_labels.detach().numpy(), writer,
                              dset + ' set recon + gemprop_mse: ' + str(loss_pred_dec.item()),
                              plot_title='Loss from reconstructed data')
        writer.close()

    if 'enc_dec_traj' in args.tasks:
        print(
            'Evaluate CDVAE and outside prediction module on the task of encoding -> decoding -> property prediction on different steps of structure optimization')
        recon_data_path = os.path.join(model_path, 'eval_recon_traj2b_20se.pt')
        recon_data = torch.load(recon_data_path)
        assert 'all_atom_types_stack' in recon_data, 'Error: the provided data do not contain Langevin dynamics trajectories!'
        writer = SummaryWriter(os.path.join(writer_path, 'enc_dec_traj_20_steps_per_sigma'))
        for step in range(recon_data['all_atom_types_stack'].shape[1]):
            print(f'preparing crystal structures from step {step}')
            recon_crystals_list = get_crystals_list(recon_data['all_frac_coords_stack'][0, step],
                                                    recon_data['all_atom_types_stack'][0, step],
                                                    recon_data['lengths'][0],
                                                    recon_data['angles'][0],
                                                    recon_data['num_atoms'][0])
            predictions = prop_model_eval('perovskite', recon_crystals_list)
            properties_dec = torch.tensor(predictions).reshape(-1, 1)

            if step == 0:
                dset = 'test'
                properties = []
                property_labels = []
                for i, batch in enumerate(loaders[2]):
                    if i >= 2:
                        continue
                    if torch.cuda.is_available():
                        batch.cuda()
                    _, _, z = model.encode(batch)

                    properties.append(prop_model(batch).detach().cpu())
                    property_labels.append(batch.y.cpu())
                properties = torch.cat(properties, dim=0)
                property_labels = torch.cat(property_labels, dim=0)
                loss_pred = torch.nn.functional.mse_loss(properties, property_labels)
                writer.add_text('Description',
                                f'Loss trajectory on {dset} set, mse on original structures {loss_pred:.5f}')
            loss_pred_dec = torch.nn.functional.mse_loss(properties_dec, property_labels)

            writer.add_scalar('Loss trajectory', loss_pred_dec.item(), step)
        writer.close()

    if 'gnn_prop' in args.tasks:
        print('Evaluate the accuracy of property prediction on a raw crystal structure')
        writer = SummaryWriter(os.path.join(writer_path, 'gnn_prop'))
        for n, dset in enumerate(['train', 'val', 'test']):
            properties = []
            property_labels = []
            for batch in loaders[n]:
                if torch.cuda.is_available():
                    batch.cuda()
                properties.append(prop_model(batch).detach())
                property_labels.append(batch.y)
            properties = torch.cat(properties, dim=0)
            property_labels = torch.cat(property_labels, dim=0)
            loss_pred = torch.nn.functional.mse_loss(properties, property_labels)
            save_scatter_plot(properties.cpu().numpy(), property_labels.cpu().numpy(), writer,
                              'gemprop_' + dset + 'mse: ' + str(loss_pred.item()),
                              plot_title='GNN loss on original structure')
        writer.close()

    if 'get_embeddings' in args.tasks:
        save_path = os.path.join(model_path, 'train_z')
        print('Create the latent embeddings of the training set')
        embeddings = []
        for i, batch in enumerate(loaders[0]):
            if torch.cuda.is_available():
                batch.cuda()
            _, _, z = model.encode(batch)
            embeddings.append(z.detach().cpu())
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, save_path)

    if 'retrain_val' in args.tasks:
        # writer = SummaryWriter(os.path.join(writer_path, 'retrain'))
        # compute losses on validation set
        val_losses = []
        crystals = []
        labels = []
        for batch in loaders[1]:
            crystals += get_crystals_list(batch.frac_coords, batch.atom_types, batch.lengths, batch.angles,
                                          batch.num_atoms)
            labels.append(batch.y)
            if torch.cuda.is_available():
                batch.cuda()
            prop = prop_model(batch)
            loss = (prop - batch.y) ** 2
            val_losses.append(loss.detach().cpu())
        labels = torch.cat(labels, dim=0).reshape(-1, 1)
        val_losses = torch.cat(val_losses, dim=0).reshape(-1)
        worst_predictions = val_losses.topk(math.ceil(val_losses.shape[0] * 0.2)).indices
        worst_mask = torch.zeros(len(crystals), dtype=torch.bool)
        worst_mask[worst_predictions] = True

        retrain_crystal_list = []
        new_val_crystal_list = []
        print(f'Partitioning data - the model will be retrained on {worst_predictions.shape[0]} hardest examples')
        for i, crystal in enumerate(tqdm(crystals)):
            crystal['y'] = labels[i]
            if worst_mask[i]:
                retrain_crystal_list.append(crystal)
            else:
                new_val_crystal_list.append(crystal)

        print('Creating retraining set and new validation set from val')
        retrain_loader = get_cryst_loader(retrain_crystal_list, cfg, prop_model.scaler)
        new_val_loader = get_cryst_loader(new_val_crystal_list, cfg, prop_model.scaler)

        trainer = pl.Trainer(
            default_root_dir=os.path.join(writer_path, 'retraining'),
            logger=True,
            callbacks=None,
            deterministic=cfg.train.deterministic,
            check_val_every_n_epoch=cfg.logging.val_check_interval,
            progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
            resume_from_checkpoint=None,
            **cfg.train.pl_trainer,
        )

        trainer.fit(model=prop_model, train_dataloader=retrain_loader, val_dataloaders=loaders[0])
        trainer.test(model=prop_model, test_dataloaders=loaders[1])

    if 'bayesian_gen' in args.tasks:
        print('Randomly generate new structures, predict their properties and assign uncertainty to those predictions')

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
            model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'ld_kwargs': ld_kwargs
        }, model_path / gen_out_name)
    '''
    if 'bayesian_val' in args.tasks:
        print('Predict the properties on the validation set and assign uncertainty of those predictions\n',
              'Plot the uncertainties against the actual errors of the predictions')

        train_property_labels = []
        train_encodings = []
        for i, batch in enumerate(loaders[0]):
            train_property_labels.append(batch.y.numpy())
            if torch.cuda.is_available():
                batch.cuda()
            _, _, z = model.encode(batch)
            train_encodings.append(z.detach().cpu().numpy())
        train_encodings = np.concatenate(train_encodings, axis=0)
        train_property_labels = np.concatenate(train_property_labels, axis=0)
        bayesian_opt = get_gaussian_regressor(train_encodings, train_property_labels)


        property_predictions = []
        squared_errors = []
        z_scores_list = []
        p_values_list = []
        property_labels = []
        mu_list = []
        var_list = []
        for i, batch in enumerate(loaders[1]):
            property_labels.append(batch.y)
            if torch.cuda.is_available():
                batch.cuda()
            properties = prop_model(batch)
            mu, var, z = model.encode(batch)
            mu_list.append(mu.detach().cpu())
            var_list.append(mu.detach().cpu())
            z_score, p_value = get_uncertainty(properties.detach().cpu().numpy(), z.detach().cpu().numpy(), bayesian_opt)
            property_predictions.append(properties.detach().cpu())
            z_scores_list.append(z_score)
            p_values_list.append(p_value)
            squared_errors.append(((properties-batch.y)**2).detach().cpu())
        property_predictions = torch.cat(property_predictions, dim=0)
        squared_errors = torch.cat(squared_errors, dim=0)
        property_labels = torch.cat(property_labels, dim=0)
        z_scores = np.concatenate(z_scores_list, axis=0)
        p_values = np.concatenate(p_values_list, axis=0)

        mu_list = torch.cat(mu_list, dim=0)
        print('shape1', mu_list.shape)
        var_list = torch.cat(var_list, dim=0)
        mean_var = (mu_list.abs()/mu_list.abs().mean(dim=0, keepdim=True)).mean(dim=1, keepdim=False)

        z_scores = torch.tensor(z_scores)
        p_values = torch.tensor(p_values)

        plt.scatter(mean_var, squared_errors)
        plt.show()
        plt.hist(mean_var.reshape(-1).numpy(), bins=20)
        plt.show()
        plt.hist(mean_var, bins=40)
        plt.show()
        plt.hist(mean_var, bins=10)
        plt.show()

        plt.scatter(p_values, squared_errors)
        plt.show()

        uncertainty_asc = torch.argsort(p_values, dim=0, descending=True).reshape(-1)
        output = {'property_predictions': property_predictions,
                  'property_labels': property_labels,
                  'z_scores': z_scores,
                  'p_values': p_values,
                  'uncertainty_asc': uncertainty_asc,
                  'mse': squared_errors
        }

        ordered_uncertainty_score = torch.linspace(0, 1, p_values.shape[0])
        ordered_mse = squared_errors[uncertainty_asc]
        plt.scatter(ordered_uncertainty_score, ordered_mse)
        plt.show()
    '''

    if 'gen_cif' in args.tasks:
        print('Generate structures from randomly created embeddings, create cif files and compute their properties')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
            model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen_cif.pt'
        else:
            gen_out_name = f'eval_gen_cif_{args.label}.pt'

        gen_crystals_list = get_crystals_list(frac_coords[0],
                                              atom_types[0],
                                              lengths[0],
                                              angles[0],
                                              num_atoms[0])
        retrain_loader = get_cryst_loader(gen_crystals_list, prop_cfg, prop_model.scaler, batch_size=args.batch_size)

        predictions = []
        for b in retrain_loader:
            if torch.cuda.is_available():
                b.cuda()
            pred = prop_model(b)
            properties = prop_model.scaler.inverse_transform(pred)
            predictions.append(properties.detach().cpu())
        predictions = torch.cat(predictions, dim=0)

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time,
            'ld_kwargs': ld_kwargs,
            'predictions': predictions
        }, model_path / gen_out_name)

        cif_path = os.path.join(model_path, gen_out_name[:-3])

        os.makedirs(cif_path, exist_ok=True)

        structure_objects = tensors_to_structures(lengths[0], angles[0], frac_coords[0], atom_types[0], num_atoms[0])
        for i, structure in enumerate(structure_objects):
            # Write structure to CIF file
            structure.to(filename=os.path.join(cif_path, f'generated{i}.cif'), fmt='cif')

    if 'opt_cif' in args.tasks:
        print('Run opt_cif multiple times')
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
                       'cbf': []}

        task_path = os.path.join(model_path, f'opt_cif_{args.label}')
        os.makedirs(task_path, exist_ok=True)
        total_generated_structures = 0
        n_batches = 3
        initial_struct_loader = loaders[1]
        extra_breakpoints = []
        for n, batch in enumerate(initial_struct_loader):
            if n == n_batches and n_batches > 0:
                break
            opt_out, z, fc_properties, optimization_breakpoints, cbf, _ = optimization_by_batch(model, ld_kwargs, batch,
                                                                                             num_starting_points=100,
                                                                                             num_gradient_steps=5000,
                                                                                             lr=1e-3, num_saved_crys=3,
                                                                                             extra_returns=True,
                                                                                             maximize=True,
                                                                                             extra_breakpoints=extra_breakpoints)


            n_generated_structures = batch.num_atoms.cpu().shape[0]
            chonker = opt_chunk_generator(opt_out, n_generated_structures)

            batch_output = {'frac_coords': [],
                            'num_atoms': [],
                            'atom_types': [],
                            'lengths': [],
                            'angles': [],
                            'z': z,
                            'fc_properties': fc_properties,
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
                        filename=os.path.join(task_path, f'generated{j + total_generated_structures}_step{bp}.cif'),
                        fmt='cif')

                    cif_writer = CifWriter(structure)
                    cif_string = cif_writer.__str__()
                    step_output['cif'].append(cif_string)

                for k, v in step_output.items():
                    batch_output[k].append(step_output[k])

            # dictionary_cat(batch_output, dim=0)
            for k, v in batch_output.items():
                output_dict[k].append(batch_output[k])
            total_generated_structures += n_generated_structures

        # dictionary_cat(output_dict, dim=1)
        output_dict['breakpoints'] = optimization_breakpoints
        torch.save(output_dict, os.path.join(task_path, 'data.pt'))


    if 'opt_retrain' in args.tasks:
        timer = {'setup': 0,
                 'optimize': 0,
                 'retrain': 0,
                 'lammps': 0}
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

        trainer = get_trainer(cfg, os.path.join(path_out, 'initial_training'))
        timer['setup'], cur_time = timer['setup'] + time.time() - start_time, time.time()


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


            lammps_cfg = {
                'lammps_path': '"C:\\Users\\GrzegorzKaszuba\\AppData\Local\\LAMMPS 64-bit 15Jun2023\\Bin\\lammps-shell.exe"',
                'pot_file': lammps_pot,
                'input_template': lammps_in,
                'pot': 'eam/alloy'}

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
            #loaders[0].dataset.relabel(new_labels=new_props, model=model)

            model.calibrate()
            model.is_calibrating = True
            trainer.fit(model, train_dataloader=loaders[0], val_dataloaders=loaders[2])
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
            timer['optimize'], cur_time = timer['optimize'] + time.time() - cur_time, time.time()

            # ---------------------- CIF ---------------------
            cif_data = []
            for batch_cif in output_dict['cif']:
                for cif_str in batch_cif[0]:
                    cif_data.append(cif_str)
            timer['optimize'], cur_time = timer['optimize'] + time.time() - cur_time, time.time()
            # ------------------- LAMMPS --------------------

            convert_cif_to_lammps(cif_dir, lammpsdata_dir)
            initial_energies, final_energies = lmp_energy_calculator(lammpsdata_dir, relaxed_lammpsdata_dir, lammps_cfg)
            elastic_vectors = lmp_elastic_calculator(lammpsdata_dir, lammps_cfg)

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
            timer['lammps'], cur_time = timer['lammps'] + time.time() - cur_time, time.time()
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
                if filtered_step_data['summary_formulas'][i] in best_by_composition.keys():
                    if filtered_step_data['elastic_vectors'][i] > best_by_composition[filtered_step_data['summary_formulas'][i]]:
                        best_by_composition[filtered_step_data['summary_formulas'][i]] = filtered_step_data['elastic_vectors'][i]
                else:
                    best_by_composition[filtered_step_data['summary_formulas'][i]] = filtered_step_data['elastic_vectors'][i]

            if len(filtered_step_data['fc_properties']) > 0:
                writer.add_scalar(f'mean_property/fc', mean(filtered_step_data['fc_properties']), step)
                writer.add_scalar(f'mean_property/lammps', mean(filtered_step_data['elastic_vectors']), step)
                writer.add_scalar(f'mean_property/fc-lammps error (abs)', mean(filtered_step_data['fc_errors']),
                                  step)
                writer.add_scalar(f'mean_property/final energy', mean(filtered_step_data['final_energies']), step)
                writer.add_scalar(f'mean_property/initial energy', mean(filtered_step_data['initial_energies']), step)

                plot_atom_ratios_mpltern(filtered_step_data['summary_formulas'],
                                         property=filtered_step_data['elastic_vectors'],
                                         save_label=os.path.join(path_out, f'tri_step {step}'))


                # ------------------------ Recalibration ------------------------
                trainer = get_trainer(cfg, os.path.join(path_out, 'initial_training'))
                cr_triangle = DataTriangle()
                cr_triangle.triangulate_data_dict(best_by_composition)
                cr_triangle.plot(savedir=path_out, label=str(step))
                new_props = get_improvement_properties(train_compositions, train_properties, cr_triangle)
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
                timer['retrain'], cur_time = timer['retrain'] + time.time() - cur_time, time.time()

            torch.save(step_data, os.path.join(pt_dir, 'full_batch.pt'))
            torch.save(filtered_step_data, os.path.join(pt_dir, 'filtered_batch.pt'))
            # fc_properties, initial_energies, final_energies, stepdir, structure_names

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
            #timer['retrain'], cur_time = timer['retrain'] + time.time() - cur_time, time.time()

        final_comps = []
        final_props = []
        for k, v in best_by_composition.items():
            final_comps.append(k)
            final_props.append(v)
        plot_atom_ratios_mpltern(final_comps,
                                 final_props,
                                 save_label=os.path.join(path_out, f'tri final'))
        print(timer)



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
