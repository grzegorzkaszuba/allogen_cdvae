from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from ase.visualize import view

import matplotlib as mpl
import matplotlib.pyplot as plt

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



if __name__ == '__main__':

    lattice_lengths = [5.468, 5.468, 5.468]  # lengths of lattice vectors for Si
    lattice_angles = [90, 90, 90]  # angles of lattice vectors
    atom_types = [6]*8  # atomic symbols for Si

    # Silicon positions in the diamond crystal structure
    atom_coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                   [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]

    visualize_structure(lattice_lengths, lattice_angles, atom_types, atom_coords)
