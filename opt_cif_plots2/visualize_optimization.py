if __name__ == '__main__':
    import numpy as np
    import torch
    import tensorboard
    import json
    import os
    import re
    from scripts.eval_utils import load_model, get_cryst_loader
    from pathlib import Path
    from cdvae.pl_data.dataset import AdHocCrystDataset

    from torch_geometric.data import DataLoader
    from cdvae.pl_data.datamodule import worker_init_fn
    import matplotlib.pyplot as plt

    def numeric_key(string):
        match = re.search('\d+', string)
        return int(match.group()) if match else 0

    def sorted_alphanum(strings):
        def numeric_key(string):
            match = re.search('\d+', string)
            return int(match.group()) if match else 0

        return sorted(strings, key=numeric_key)

    def sorted_values(dictionary):
        sorted_keys = sorted_alphanum(list(dictionary.keys()))
        return [dictionary[key] for key in sorted_keys]

    def join_step_dicts(dict_list):
        # Initialize an output dictionary and a maximum example number
        out_dict = {}
        max_example_num = 0
        index_shift = 0
        # Iterate over the list of dictionaries
        for d in dict_list:
            for key, value in d.items():
                example_num, step = key.split('_')

                # If we're in the first dictionary, add items directly to out_dict
                if d == dict_list[0]:
                    out_dict[key] = value
                    max_example_num = max(max_example_num, int(example_num[9:]))
                else:
                    # In subsequent dictionaries, increment the example number
                    new_example_num = "generated" + str(int(example_num[9:]) + index_shift)
                    max_example_num = max(max_example_num, int(new_example_num[9:]))
                    new_key = new_example_num + '_' + step
                    out_dict[new_key] = value
            index_shift = max_example_num+1
        return out_dict


    def step_dict_to_tensor(d):
        # split keys and sort by 'exampleN' then 'stepM'
        sorted_keys = sorted(d.keys(), key=lambda x: (numeric_key(x.split('_')[0]), numeric_key(x.split('_')[1])))

        # group keys by example and step to create a 2D list
        sorted_examples = []
        current_example = sorted_keys[0].split('_')[0]
        current_example_steps = []

        for key in sorted_keys:
            example, step = key.split('_')
            if example == current_example:
                current_example_steps.append(d[key][3])  # get 4th attribute
            else:
                sorted_examples.append(current_example_steps)
                current_example = example
                current_example_steps = [d[key][3]]  # start a new list for the new example

        # don't forget to add the last example's steps
        sorted_examples.append(current_example_steps)

        # convert the 2D list to a 2D torch tensor
        tensor = torch.tensor(sorted_examples)

        return tensor


    def step_dict_to_tensor_z(d):
        # split keys and sort by 'exampleN' then 'stepM'
        sorted_keys = sorted(d.keys(), key=lambda x: (numeric_key(x.split('_')[0]), numeric_key(x.split('_')[1])))

        # group keys by example and step to create a 2D list
        sorted_examples = []
        current_example = sorted_keys[0].split('_')[0]
        current_example_steps = []

        for key in sorted_keys:
            example, step = key.split('_')
            if example == current_example:
                current_example_steps.append(list(d[key]))  # get 4th attribute
            else:
                sorted_examples.append(current_example_steps)
                current_example = example
                current_example_steps = [list(d[key])]  # start a new list for the new example

        # don't forget to add the last example's steps
        sorted_examples.append(current_example_steps)

        # convert the 2D list to a 2D torch tensor
        tensor = torch.tensor(sorted_examples)

        return tensor

    def capture_timesteps(d):
        # Split keys into 'stepM' and get the unique values
        timesteps = set(key.split('_')[1] for key in d.keys())

        # Sort the timesteps using your sorted_alphanum function
        sorted_timesteps = sorted_alphanum(list(timesteps))

        return sorted_timesteps


    from torch.utils.tensorboard import SummaryWriter
    opt_cif_path = 'C:\\Users\\GrzegorzKaszuba\\PycharmProjects\\cdvae-main\\opt_cif_plots2'
    data_path = os.path.join(opt_cif_path, 'data')

    writer = SummaryWriter(os.path.join(opt_cif_path, 'plots', 'develop'))

    data_dicts = []
    for filename in sorted_alphanum(os.listdir(data_path)):
        if '.json' in filename:
            with open(os.path.join(data_path, filename), 'r') as f:
                data_dicts.append(json.load(f))


    trial_data = step_dict_to_tensor(data_dicts[0])
    combined_data = join_step_dicts(data_dicts)
    timesteps = capture_timesteps(data_dicts[0])
    timesteps = [int(timestep[4:]) for timestep in timesteps]

    data = step_dict_to_tensor(combined_data).T
    mean_data = data.mean(dim=1)
    for i in range(data.shape[1]):
        variable_values = data[:, i]  # Select the values for the i-th variable

        # Plot the values for the i-th variable at each timepoint
        for t, value in zip(timesteps, variable_values):
            writer.add_scalar(f'MD/Elastic_vector_{i}', value, t)


    # Plot the values for the mean variable at each timepoint
    for t, ind in zip(timesteps, range(mean_data.shape[0])):
        writer.add_scalar('MD/Mean_elastic_vector', mean_data[ind], t)

    pt_data_list = [torch.load(os.path.join(data_path, directory, 'data.pt')) for directory in os.listdir(data_path) if directory[:5]=='multi']
    z_init = torch.cat([d['z'] for d in pt_data_list], dim=1)
    fc_properties = torch.cat([d['fc_properties'] for d in pt_data_list], dim=1)

    model_path = 'C:\\Users\\GrzegorzKaszuba\\PycharmProjects\\cdvae-main\\hydra\\singlerun\\2023-06-25\\cdvae_prop_elastic'
    model, _, cfg = load_model(Path(model_path))
    if torch.cuda.is_available():
        model.cuda()
    fc_properties = model.scaler.inverse_transform(fc_properties)
    fc_properties = fc_properties[:, :, 0].cpu()
    fc_error = torch.abs(fc_properties - data)

    mean_property = fc_properties.mean(dim=1)
    mean_error = fc_error.mean(dim=1)

    for i in range(data.shape[1]):
        variable_values = fc_properties[:, i]  # Select the values for the i-th variable

        # Plot the values for the i-th variable at each timepoint
        for t, value in zip(timesteps, variable_values):
            writer.add_scalar(f'FC/Elastic_vector_{i}', value, t)


    # Plot the values for the mean variable at each timepoint
    for t, ind in zip(timesteps, range(mean_data.shape[0])):
        writer.add_scalar('FC/Mean_elastic_vector', mean_property[ind], t)


    for i in range(data.shape[1]):
        variable_values = fc_error[:, i]  # Select the values for the i-th variable

        # Plot the values for the i-th variable at each timepoint
        for t, value in zip(timesteps, variable_values):
            writer.add_scalar(f'FC_error/Elastic_vector_{i}', value, t)


    # Plot the values for the mean variable at each timepoint
    for t, ind in zip(timesteps, range(mean_data.shape[0])):
        writer.add_scalar('FC_error/Mean_elastic_vector', mean_error[ind], t)

    ld_kwargs = pt_data_list[0]['ld_kwargs']


    def get_cif_dict(path):
        d = {}
        for filename in os.listdir(path):
            if filename[-4:] == '.cif':
                with open(os.path.join(path, filename), 'r') as f:
                    d[filename] = f.read()
        return d

    cif_data_dicts = [get_cif_dict(os.path.join(data_path, directory)) for directory in os.listdir(data_path) if directory[:5]=='multi']
    combined_cif_dict = join_step_dicts(cif_data_dicts)
    cif_keys = []
    cif_data = []

    for k, v in combined_cif_dict.items():
        cif_keys.append(k)
        cif_data.append(v)

    (niggli, primitive, graph_method, preprocess_workers, lattice_scale_method) = (
        cfg.data.datamodule.datasets.train.niggli,
        cfg.data.datamodule.datasets.train.primitive,
        cfg.data.datamodule.datasets.train.graph_method,
        cfg.data.datamodule.datasets.train.preprocess_workers,
        cfg.data.datamodule.datasets.train.lattice_scale_method
    )

    dataset = AdHocCrystDataset('identity_test_dataset', cif_data, None, niggli, primitive,
                     graph_method, preprocess_workers, lattice_scale_method)


    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=16,
        num_workers=0,
        worker_init_fn=worker_init_fn) #scaler could be added like in get_cryst_loader, but should not be necessary

    z_list = []
    for batch in loader:
        if torch.cuda.is_available():
            batch.cuda()
        _, _, z = model.encode(batch.cuda())
        z_list.append(z.detach().cpu().numpy())

    z_list = np.concatenate(z_list, axis=0)
    z_dict = {}
    for n, k in enumerate(cif_keys):
        z_dict[k] = z_list[n]

    z_tensor = step_dict_to_tensor_z(z_dict)
    #torch.save(z_tensor, 'z_identity')
    z_tensor = torch.transpose(z_tensor, 0, 1)

    z_error = torch.sum(torch.abs(z_tensor - z_init), dim=2)

    mean_z_error = z_error.mean(dim=1)
    for i in range(data.shape[1]):
        variable_values = z_error[:, i]  # Select the values for the i-th variable

        # Plot the values for the i-th variable at each timepoint
        for t, value in zip(timesteps, variable_values):
            writer.add_scalar(f'z_identity_error/Error_{i}', value, t)


    # Plot the values for the mean variable at each timepoint
    for t, ind in zip(timesteps, range(mean_data.shape[0])):
        writer.add_scalar('z_identity_error/Mean_error', mean_error[ind], t)


    x_vals = []
    y_vals = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x_vals.append(torch.log(z_error[i, j]).item())
            y_vals.append(torch.log(fc_error[i, j]).item())

    plt.scatter(x_vals, y_vals, s=2)
    plt.xlabel('log(z_error)')
    plt.ylabel('log(md_error)')
    plt.grid(True)

    # Convert plot to tensorboardX format and add it to tensorboard
    writer.add_figure('log z error to fc error', plt.gcf())


    x_vals = []
    y_vals = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x_vals.append(torch.clamp(torch.log(z_error[i, j]), min=torch.tensor(0.), max=torch.tensor(5.)).item())
            y_vals.append(torch.clamp(torch.log(fc_error[i, j]), min=torch.tensor(0.), max=torch.tensor(8.)).item())

    plt.scatter(x_vals, y_vals, s=2)
    plt.xlabel('log(z_error)')
    plt.ylabel('log(md_error)')
    plt.grid(True)

    # Convert plot to tensorboardX format and add it to tensorboard
    writer.add_figure('bounded log z error to fc error', plt.gcf())


    writer.close()






