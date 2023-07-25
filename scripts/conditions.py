import torch


class Condition:
    def __init__(self, key, minval, maxval):
        self.key = key
        self.min = minval
        self.max = maxval

    def test(self, data_dict, i):
        tested_value = data_dict[self.key][i]
        if self.key == 'summary_formulas':
            n_atoms = sum(tested_value)
            return all([el / n_atoms >= self.min and el / n_atoms <= self.max for el in tested_value])
        else:
            return tested_value >= self.min and tested_value <= self.max


class ZLoss:
    RELEVANT_ELEMENTS = torch.tensor([23, 25, 27])
    def __init__(self, type='comp', minval=0, maxval=1, magnitude=1, weight=1):
        self.type = type
        self.min = minval
        self.max = maxval
        self.magnitude = magnitude
        self.weight = weight

    def __call__(self, z, model):
        if self.type == 'comp':
            compositions = model.fc_composition(z)
            rel_compositions = compositions[:, self.RELEVANT_ELEMENTS]
            return torch.where(rel_compositions > self.max, rel_compositions, 0) - torch.where(rel_compositions < self.min, rel_compositions, 0) * self.weight
        if self.type == 'best_by_comp':
            compositions = model.fc_composition(z)
            rel_compositions = compositions[:, self.RELEVANT_ELEMENTS]
            return - self.model.composition_rank(rel_compositions) * self.weight




def filter_step_data(step_data, conditions):
    filtered = {}
    for k in step_data.keys():
        filtered[k] = []
    for i in range(len(step_data['elastic_vectors'])):
        if all(c.test(step_data, i) for c in conditions):
            for k, v in step_data.items():
                filtered[k].append(step_data[k][i])
    return filtered