from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from ase.visualize import view

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from io import BytesIO

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import re

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

def extract_atom_counts(cif_file, elements):
    try:
        with open(cif_file, 'r') as file:
            for line in file:
                if line.startswith('_chemical_formula_sum'):
                    # Extract element counts from the line
                    formula = line.split('\'')[1]  # Get the string between quotes

                    counts = []
                    for elem in elements:
                        search = re.search(fr'{elem}(\d*)', formula)
                        if search:
                            count = int(search.group(1)) if search.group(1) != '' else 1
                        else:
                            count = 0
                        counts.append(count)
            return tuple(counts)
    except:
        emergency_out = [10 for i in range(len(elements))]
        emergency_out[0] = -1
        return tuple(emergency_out)



"""
def plot_atom_ratios_mpltern(atom_counts, property=None, writer=None, global_step=None, save_label=''):
    # Calculate atom ratios
    atom_ratios = [[count / sum(counts) for count in counts] for counts in atom_counts]

    # Separate the ratios into separate lists for each element
    cr_ratios = [ratios[0] for ratios in atom_ratios]
    fe_ratios = [ratios[1] for ratios in atom_ratios]
    ni_ratios = [ratios[2] for ratios in atom_ratios]

    # Create ternary plot
    fig, tax = plt.subplots(subplot_kw={'projection': 'ternary'})

    # Set labels and title
    tax.set_tlabel("Cr %")
    tax.set_llabel("Fe %")
    tax.set_rlabel("Ni %")

    # Plot points
    sc = tax.scatter(cr_ratios, fe_ratios, ni_ratios, marker='o', c=property, cmap='viridis')

    # Add colorbar if property is defined
    if property is not None:
        plt.subplots_adjust(right=0.8)  # Make space for colorbar
        cax = plt.axes([0.85, 0.1, 0.03, 0.8])  # Specify the position and size of colorbar
        fig.colorbar(sc, cax=cax, label='Property')

    # Add grid
    tax.grid(color='lightblue')

    # If writer is defined, log the plot to TensorBoard, otherwise show the plot
    if writer is not None:
        # Convert the plot to a PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        # Convert the PIL Image to a numpy array
        img_arr = np.array(img)

        # Log the image
        writer.add_image('Ternary Plot', img, global_step=global_step)

        # Close the figure to free memory
        plt.close(fig)
    else:
        # Show plot
        plt.savefig(save_label + '.png')
        plt.close(fig)
"""

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
