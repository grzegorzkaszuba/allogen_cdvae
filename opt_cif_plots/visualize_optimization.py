import torch
import tensorboard
import json
import os
import re

def sorted_alphanum(strings):
    def alpha_numeric_key(string):
        return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', string)]

    return sorted(strings, key=alpha_numeric_key)

def sorted_values(dictionary):
    sorted_keys = sorted_alphanum(list(dictionary.keys()))
    return [dictionary[key] for key in sorted_keys]

from torch.utils.tensorboard import SummaryWriter
opt_cif_path = 'C:\\Users\\GrzegorzKaszuba\\PycharmProjects\\cdvae-main\\opt_cif_plots'
data_path = os.path.join(opt_cif_path, 'data')

writer = SummaryWriter(os.path.join(opt_cif_path, 'plots'))

data = []
for filename in sorted_alphanum(os.listdir(data_path)):
    if '.json' in filename:
        with open(os.path.join(data_path, filename), 'r') as f:
            data.append([sv[3] for sv in sorted_values(json.load(f))])

data = torch.tensor(data)
mean_data = data.mean(dim=1)
timepoints = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]

for i in range(data.shape[1]):
    variable_values = data[:, i]  # Select the values for the i-th variable

    # Plot the values for the i-th variable at each timepoint
    for t, value in zip(timepoints, variable_values):
        writer.add_scalar(f'Generate_elastic_vector_{i}', value, t)

# Calculate the mean variable

# Plot the values for the mean variable at each timepoint
for t, ind in zip(timepoints, range(mean_data.shape[0])):
    writer.add_scalar('Mean_elastic_vector', mean_data[ind], t)

writer.close()


print('end')



