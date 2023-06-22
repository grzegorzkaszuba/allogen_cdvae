import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model, load_model_full, load_tensor_data, prop_model_eval, get_crystals_list, get_cryst_loader
from visualization_utils import save_scatter_plot

import tensorboard
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib as mpl
import matplotlib.pyplot as plt

import math
import pytorch_lightning as pl

from evaluate import reconstructon, generation, optimization

from gaussian_process import get_gaussian_regressor, get_uncertainty

import numpy as np


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

    if 'retrain_val' in args.tasks:
        #writer = SummaryWriter(os.path.join(writer_path, 'retrain'))
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
            loss = (prop-batch.y)**2
            val_losses.append(loss.detach().cpu())
        labels = torch.cat(labels, dim=0).reshape(-1, 1)
        val_losses = torch.cat(val_losses, dim=0).reshape(-1)
        worst_predictions = val_losses.topk(math.ceil(val_losses.shape[0]*0.2)).indices
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

    if 'bayesian_val' in args.tasks:
        print('Predict the properties on the validation set and assign uncertainty of those predictions\n',
              'Plot the uncertainties against the actual errors of the predictions')
        print('safety1')

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

        print('safety2')




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
