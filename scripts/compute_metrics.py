from collections import Counter
import argparse
import os
import json
import tempfile
import subprocess
from scipy.optimize import linear_sum_assignment

import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
from collections import Counter

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
from pymatgen.io.cif import CifWriter

from gvect_utils import cif_to_json, modify_gvec, panna_cfg, gvector, gvect_distance, template_gdist

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

import pandas as pd
#CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
#CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


def slice_structure(struct: Structure, s: slice) -> Structure:
    """Return a new Structure that's a subset of the original based on the provided slice."""

    lattice = struct.lattice
    species_list = [site.species for site in struct[s]]
    coords_list = [site.coords for site in struct[s]]


def most_common_element_percentage(atomic_numbers) -> float:
    # Calculate the total number of atoms.
    total_atoms = atomic_numbers.shape[0]
    _, counts = np.unique(atomic_numbers, return_counts=True)
    # Find the count of the most common atomic number.
    most_common_count = np.max(counts)

    # Compute the percentage.
    most_common_ratio = most_common_count / total_atoms

    return most_common_ratio


import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def save_metrics(metric_dictionary, difficulties, path, label=None, colour=None):
    """
    Store metrics in a given directory and plot scores against difficulty.

    Parameters:
    - metric_dictionary (dict): Dictionary with metrics as numpy ndarrays.
    - difficulties (np.ndarray): Numpy array representing difficulty of examples.
    - path (str): Directory path to store the plots and mean metrics.
    - label (str, optional): Label for the model/dataset to include in plot titles.

    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Compute means of metrics and save as a dictionary
    mean_metrics = {key: np.mean(value) for key, value in metric_dictionary.items()}

    # Save the mean metrics as a PyTorch file
    torch.save(mean_metrics, os.path.join(path, 'mean_metrics.pt'))
    torch.save(metric_dictionary, os.path.join(path, 'metrics.pt'))

    # For each metric, create scatter plots of score vs difficulty
    for metric_name, scores in metric_dictionary.items():
        plt.figure(figsize=(10, 6))

        # Plotting
        if colour is None:
            plt.scatter(difficulties, scores, alpha=0.6)
        else:
            colour_map = {0: 'blue', 1: 'red'}
            plot_colours = [colour_map[c] for c in colour]
            plt.scatter(difficulties, scores, alpha=0.6, c=plot_colours)
        plt.xlabel('Most common element content')
        plt.ylabel(metric_name)

        # Set title based on presence of label
        if label:
            plt.title(f"{metric_name} for {label}")
        else:
            plt.title(f"{metric_name}y")

        plt.grid(True)

        # Save the plot
        plt.savefig(os.path.join(path, f"{metric_name}.png"))
        plt.close()


# Example usage:
# metric_dict = {"accuracy": np.random.rand(100), "loss": np.random.rand(100)}
# difficulties = np.random.rand(100)
# store_metrics_and_plot(metric_dict, difficulties, "./results", label="Model1")



class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        pass
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        #self.comp_fp = CompFP.featurize(comp)
        #try:
        #site_fps = [CrystalNNFP.featurize(
        #        self.structure, i) for i in range(len(self.structure))]
        #except Exception:
        #    # counts crystal as invalid if fingerprint cannot be constructed.
        #    self.valid = False
        #    self.comp_fp = None
        #    self.struct_fp = None
        #    return
        #self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_gdist(self):
        def process_one(pred, gt, is_valid, use_lammps=False):
            if not is_valid:
                return None, None
        #try:

            gdist = gvect_distance(pred.structure, gt.structure, panna_cfg)
            #gdist = None
            gdist_a = gvect_distance(pred.structure, gt.structure, panna_cfg, anonymous=True)
            #gdist_fcc = template_gdist(pred.structure, 'fcc', panna_cfg)
            #gdist_bcc = template_gdist(pred.structure, 'bcc', panna_cfg)

            #fcc_score = 1-(gdist_fcc/(gdist_bcc+gdist_fcc))
            return gdist, gdist_a
        #except Exception:
        #    return None
        validity = [c.valid for c in self.preds]
        self.validity = validity
        gdists = []
        gdists_a = []
        for i in tqdm(range(len(self.preds))):
            gdist, gdist_a = process_one(
                self.preds[i], self.gts[i], validity[i])
            if gdist is not None:
                gdists.append(gdist)
            if gdist_a is not None:
                gdists_a.append(gdist_a)
                print(f'\n\nmean gdist_a: {np.mean(gdists_a)} \n\n')
        gdists = np.array(gdists)
        gdists_a = np.array(gdists_a)
        return {'gdist': gdists,
                'anonymous_gdist': gdists_a}


    def get_metrics(self):
        return self.get_gdist()


class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_prop_wdist())
        print(metrics)
        metrics.update(self.get_coverage())
        return metrics


class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])
        valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        valid_x, valid_y = valid_indices.nonzero()
        props = np.ones([self.step_opt, self.num_opt]) * np.inf
        valid_crys = [c for c in self.crys if c.valid]
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in valid_crys])
            percentiles = Percentiles[self.eval_model_name]
            props[valid_x, valid_y] = pred_props
            best_props = props.min(axis=0)
            sr_5 = (best_props <= percentiles[0]).mean()
            sr_10 = (best_props <= percentiles[1]).mean()
            sr_15 = (best_props <= percentiles[2]).mean()
        return {'SR5': sr_5, 'SR10': sr_10, 'SR15': sr_15}

    def get_metrics(self):
        return self.get_success_rate()


def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data['frac_coords'][batch_idx],
        data['atom_types'][batch_idx],
        data['lengths'][batch_idx],
        data['angles'][batch_idx],
        data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def main(args):
    panna_path = ''
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    if 'recon' in args.tasks:
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        recon_metric_out = os.path.join(recon_file_path.split('.')[0], 'recon_metrics')
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path)
        pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
        gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)
        mcep = np.array([most_common_element_percentage(pr.atom_types) for pr in pred_crys])
        rec_evaluator = RecEval(pred_crys, gt_crys)
        recon_metrics = rec_evaluator.get_metrics()
        all_metrics.update(recon_metrics)
        save_metrics(recon_metrics, mcep[rec_evaluator.validity], recon_metric_out)
        df = pd.read_csv(os.path.join(cfg.data.root_path, 'test.csv'))

        if 'sym' in df.columns:
            sym = list(df['sym'])
            for i in range(len(sym)):
                if sym[i] == 'fcc':
                    sym[i] = 0
                else:
                    sym[i] = 1
            if 'preprocess_limit' in cfg.data.datamodule.datasets.test[0].keys():
                sym = sym[:cfg.data.datamodule.datasets.test[0].preprocess_limit]
            recon_metric_out_col = os.path.join(recon_file_path.split('.')[0], 'recon_metrics_col')
            save_metrics(recon_metrics, mcep[rec_evaluator.validity], recon_metric_out_col, colour=sym)

    if 'gen' in args.tasks:
        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
        if 'recon' not in args.tasks:
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    if 'opt' in args.tasks:
        opt_file_path = get_file_paths(args.root_path, 'opt', args.label)
        crys_array_list, _ = get_crystal_array_list(opt_file_path)
        opt_crys = p_map(lambda x: Crystal(x), crys_array_list)

        opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
        opt_metrics = opt_evaluator.get_metrics()
        all_metrics.update(opt_metrics)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    args = parser.parse_args()
    main(args)
