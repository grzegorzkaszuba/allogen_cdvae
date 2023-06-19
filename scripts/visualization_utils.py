from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from ase.visualize import view


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



if __name__ == '__main__':

    lattice_lengths = [5.468, 5.468, 5.468]  # lengths of lattice vectors for Si
    lattice_angles = [90, 90, 90]  # angles of lattice vectors
    atom_types = [6]*8  # atomic symbols for Si

    # Silicon positions in the diamond crystal structure
    atom_coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                   [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]

    visualize_structure(lattice_lengths, lattice_angles, atom_types, atom_coords)
