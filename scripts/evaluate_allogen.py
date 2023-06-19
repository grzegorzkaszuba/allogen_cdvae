import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model, load_model_full, load_tensor_data, prop_model_eval, get_crystals_list

import tensorboard
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib as mpl
import matplotlib.pyplot as plt




def save_scatter_plot(pred, label, writer, name, plot_title=None):
    if plot_title is None:
        plot_title = name

    plt.scatter(pred, label, s=1)

    # Set the limits of x and y to be the same
    min_val = min(min(pred), min(label))
    max_val = max(max(pred), max(label))

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # Add the line x = y
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Set the aspect of the plot to be equal, so the scale is the same on x and y
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('Prediction')
    plt.ylabel('Label')

    plt.title(plot_title)
    writer.add_figure(name, plt.gcf(), 0)

    plt.show()


def reconstructon(loader, model, ld_kwargs, num_evals,
                  force_num_atoms=False, force_atom_types=False, down_sample_traj_step=1):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if idx <=1:
            if torch.cuda.is_available():
                batch.cuda()
            print(f'batch {idx} in {len(loader)}')
            batch_all_frac_coords = []
            batch_all_atom_types = []
            batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
            batch_lengths, batch_angles = [], []

            # only sample one z, multiple evals for stoichaticity in langevin dynamics
            _, _, z = model.encode(batch)

            for eval_idx in range(num_evals):
                gt_num_atoms = batch.num_atoms if force_num_atoms else None
                gt_atom_types = batch.atom_types if force_atom_types else None
                outputs = model.langevin_dynamics(
                    z, ld_kwargs, gt_num_atoms, gt_atom_types)

                # collect sampled crystals in this batch.
                batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
                batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
                batch_atom_types.append(outputs['atom_types'].detach().cpu())
                batch_lengths.append(outputs['lengths'].detach().cpu())
                batch_angles.append(outputs['angles'].detach().cpu())
                if ld_kwargs.save_traj:
                    batch_all_frac_coords.append(
                        outputs['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                    batch_all_atom_types.append(
                        outputs['all_atom_types'][::down_sample_traj_step].detach().cpu())
            # collect sampled crystals for this z.
            frac_coords.append(torch.stack(batch_frac_coords, dim=0))
            num_atoms.append(torch.stack(batch_num_atoms, dim=0))
            atom_types.append(torch.stack(batch_atom_types, dim=0))
            lengths.append(torch.stack(batch_lengths, dim=0))
            angles.append(torch.stack(batch_angles, dim=0))
            if ld_kwargs.save_traj:
                all_frac_coords_stack.append(
                    torch.stack(batch_all_frac_coords, dim=0))
                all_atom_types_stack.append(
                    torch.stack(batch_all_atom_types, dim=0))
            # Save the ground truth structure
            input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack, input_data_batch)


def generation(model, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)


def optimization(model, ld_kwargs, data_loader,
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
    interval = num_gradient_steps // (num_saved_crys-1)
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        loss = model.fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps-1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}



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
    interval = num_gradient_steps // (num_saved_crys-1)
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        loss = model.fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps-1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    prop_model = None

    if args.prop_model_path:
        prop_model_path = Path(args.prop_model_path)
        prop_model, _, prop_cfg = load_model(prop_model_path)
        prop_model.to('cuda')
    print('prop_model_parameters', count_parameters(prop_model))



    model_path = Path(args.model_path)

    model, loaders, cfg = load_model_full(model_path)
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
            save_scatter_plot(properties.detach().cpu(), property_labels.detach().cpu(), writer, 'enc_prop_'+dset+' mse: '+str(loss_pred))
        writer.close()

    if 'enc_dec_prop' in args.tasks:
        print('Evaluate the combination of CDVAE and outside prediction module on the task of encoding -> decoding -> property prediction')
        recon_data_path = os.path.join(model_path, 'eval_recon.pt')
        recon_data = torch.load(recon_data_path)
        recon_crystals_list = get_crystals_list(recon_data['frac_coords'][0],
                                                recon_data['atom_types'][0],
                                                recon_data['lengths'][0],
                                                recon_data['angles'][0],
                                                recon_data['num_atoms'][0])
        predictions = prop_model_eval('perovskite', recon_crystals_list)
        properties_dec = torch.tensor(predictions).reshape(-1,1)

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
            #properties_dec = torch.cat(properties_dec, dim=0)
            loss_pred = torch.nn.functional.mse_loss(properties, property_labels)
            loss_pred_dec = torch.nn.functional.mse_loss(properties_dec, property_labels)
            save_scatter_plot(properties.numpy(), property_labels.detach().numpy(), writer, dset+ ' set gemprop mse '+str(loss_pred.item()), plot_title='Loss from data')
            save_scatter_plot(properties_dec.numpy(), property_labels.detach().numpy(), writer, dset+ ' set recon + gemprop_mse: '+str(loss_pred_dec.item()), plot_title='Loss from reconstructed data')
        writer.close()

    if 'enc_dec_traj' in args.tasks:
        print('Evaluate CDVAE and outside prediction module on the task of encoding -> decoding -> property prediction on different steps of structure optimization')
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
                writer.add_text('Description', f'Loss trajectory on {dset} set, mse on original structures {loss_pred:.5f}')
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
            save_scatter_plot(properties.cpu().numpy(), property_labels.cpu().numpy(), writer, 'gemprop_'+dset+'mse: '+str(loss_pred.item()), plot_title='GNN loss on original structure')
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

    if 'recon_prop' in args.tasks:
        print('Evaluate the outside prediction module on reconstructed cases')


    if 'gen_prop' in args.tasks:
        print('Generate new materials and compare the results of property prediction with and without decoding step')
        # for ground truth, DFT would be needed

    if 'opt_prop' in args.tasks:
        print('Optimize the structures with respect to the property module and check the correlation with the dec-prop pipeline')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--prop_model_path', default='')
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=1, type=int)
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
