from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from ase.visualize import view

import matplotlib as mpl
import matplotlib.pyplot as plt

def cif_names_list(timesteps, datapoints):
    def cif_names_gen(timestep, datapoints):
        for i in range(datapoints):
            yield f'generated{i}_step{timestep}'
    return [list(cif_names_gen(timestep, datapoints)) for timestep in range(timesteps)]


def visualize_structure(lattice_lengths, lattice_angles, atom_numbers, atom_coords):
    # Convert atomic numbers to symbols
    atom_types = [Element.from_Z(Z).symbol for Z in atom_numbers]

    # Create a Lattice from lengths and angles
    lattice = Lattice.from_parameters(*lattice_lengths, *lattice_angles)

    # Create a structure
    structure = Structure(lattice, atom_types, atom_coords)

    # Convert to ase Atoms object
    structure_ase = AseAtomsAdaptor.get_atoms(structure)

    # Print out unique atom types as a legend
    unique_atom_types = set(atom_types)
    print("Atom types in the structure:")
    for atom_type in unique_atom_types:
        print(atom_type)

    # Visualize the structure
    view(structure_ase)



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


import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

# Create a tensor of floats with shape 10, 128
data = torch.rand(10, 128)

# Create a list of 10 names
names = ['name_' + str(i) for i in range(10)]

# Initialize SummaryWriter
writer = SummaryWriter()

# Iterate over tensors and names
for i, (tensor, name) in enumerate(zip(data, names)):
    # Reshape the tensor to 8x16
    tensor_reshaped = tensor.view(8, 16)

    # Normalize tensor to range [0, 1] for correct image display
    tensor_reshaped = (tensor_reshaped - tensor_reshaped.min()) / (tensor_reshaped.max() - tensor_reshaped.min())

    # Create a colormap
    plt.figure(figsize=(5,5))
    im = plt.imshow(tensor_reshaped.numpy(), cmap='jet')
    plt.axis('off')

    # Add a colorbar to the figure
    plt.colorbar(im, orientation='horizontal')

    # Save plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Load buffer as an image
    img = Image.open(buffer)

    # Convert image to tensor
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)

    # Add to tensorboard
    writer.add_image(name, img_tensor, i)

    plt.close()

# Close the writer
writer.close()





if __name__ == '__main__':

    lattice_lengths = [5.468, 5.468, 5.468]  # lengths of lattice vectors for Si
    lattice_angles = [90, 90, 90]  # angles of lattice vectors
    atom_types = [6]*8  # atomic symbols for Si

    # Silicon positions in the diamond crystal structure
    atom_coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                   [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]

    visualize_structure(lattice_lengths, lattice_angles, atom_types, atom_coords)
