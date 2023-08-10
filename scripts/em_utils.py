import numpy as np
from random import shuffle, randint
import torch
from pymatgen.analysis import local_env
from torch_geometric.nn.pool import radius
from pymatgen.core import Structure
from types import SimpleNamespace
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

def generate_interaction_matrix(labels_per_element: int, possible_elements: list, uniform=True):
    """
    Generate a random interaction matrix based on the number of labels per element and the possible elements.

    :param labels_per_element: Number of subclasses for each element (e.g. for 3, you'd have Fe_1, Fe_2, Fe_3)
    :param possible_elements: List of possible atomic types (e.g. ['Fe', 'Ni'])
    :return: A numpy array representing the interaction matrix
    """
    n_subclasses = labels_per_element * len(possible_elements)
    n_original_classes = len(possible_elements)

    # Generate random values
    if not uniform:
        matrix = np.random.rand(n_subclasses, n_original_classes)
    else:
        matrix = np.ones((n_subclasses, n_original_classes))

    # Normalize each row so they sum up to 1
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def random_structure(size=16, random_state=None):
    # Assuming a simple cubic lattice for simplicity
    # Adjust the lattice constant as needed
    lattice = [[3.5, 0, 0], [0, 3.5, 0], [0, 0, 3.5]]

    if random_state is None:
        n_Ni = randint(0, size)  # Random number of Ni atoms between 0 and 16
        n_Fe = size - n_Ni  # Compute number of Fe atoms

        species_list = ["Ni"] * n_Ni + ["Fe"] * n_Fe
        shuffle(species_list)  # Randomize order of Ni and Fe atoms

        coords = [np.random.rand(3) for _ in species_list]  # Random coordinates in the unit cell
    else:
        n_Ni = random_state.randint(0, size)  # Random number of Ni atoms between 0 and 16
        n_Fe = size - n_Ni  # Compute number of Fe atoms

        species_list = ["Ni"] * n_Ni + ["Fe"] * n_Fe
        random_state.shuffle(species_list)  # Randomize order of Ni and Fe atoms

        coords = [random_state.rand(3) for _ in species_list]  # Random coordinates in the unit cell

    return Structure(lattice, species_list, coords)


def compute_interactions(structure_list, rbf_function):
    # Define possible species pairs
    species_pairs = [("Ni", "Ni"), ("Fe", "Ni"), ("Fe", "Fe")]

    # Initialize result matrix with zeros
    interaction_matrix = {pair: 0 for pair in species_pairs}

    for structure in structure_list:
        # Convert coordinates to tensor format
        coords_tensor = torch.tensor(structure.cart_coords, dtype=torch.float)

        # Generate adjacency matrix with edge indices
        edge_indices_list = radius(coords_tensor, coords_tensor, r=3.5)
        source_nodes = edge_indices_list[0].tolist()
        target_nodes = edge_indices_list[1].tolist()
        #source_nodes = np.array(source_nodes)

        for i, j in zip(source_nodes, target_nodes):
            # Calculate the distance between atoms
            dist = np.linalg.norm(structure.cart_coords[i] - structure.cart_coords[j])

            # Get the species of the source and target node
            source_species = structure[i].species_string
            target_species = structure[j].species_string

            # Apply the RBF function
            interaction_value = rbf_function(dist)

            # Add the interaction value to the corresponding pair in the matrix
            pair = tuple(sorted([source_species, target_species]))
            interaction_matrix[pair] += interaction_value

    # Normalize the interaction matrix by number of structures
    for pair in species_pairs:
        interaction_matrix[pair] /= len(structure_list)

    # Convert to a symmetric matrix format
    matrix = np.zeros((2, 2))
    species_index = {"Ni": 0, "Fe": 1}
    for pair, value in interaction_matrix.items():
        i, j = species_index[pair[0]], species_index[pair[1]]
        matrix[i][j] = value
        matrix[j][i] = value

    return matrix

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def build_crystal_graph2(crystal, graph_method='crystalnn'):
    """
    """

    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CrystalNN)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages, edge_lengths = [], [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            src_coords = frac_coords[i] * lengths
            tgt_coords = frac_coords[j] * lengths + np.array(to_jimage) * lengths
            distance_vec = tgt_coords - src_coords
            edge_length = np.linalg.norm(distance_vec)

            edge_lengths.append(edge_length)
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_lengths.append(edge_length)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    edge_lengths = np.array(edge_lengths)
    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    #return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms
    return SimpleNamespace(frac_coords=frac_coords, atom_types=atom_types, lengths=lengths, angles=angles,
                           edge_indices=edge_indices.T, to_jimages=to_jimages, num_atoms=num_atoms, edge_lengths=edge_lengths)


def expand_atom_labels(cryst, labels_per_atom, possible_atoms):
    n_atoms = len(cryst.atom_types)
    n_elements = len(possible_atoms)

    # Create an empty array to store the expanded labels
    #expanded_labels = np.zeros((n_atoms, n_elements * labels_per_atom))
    expanded_labels = np.zeros((n_atoms, labels_per_atom))

    # Map from atomic number to index in possible_atoms
    atom_index_map = {Element(a).Z: n for n, a in enumerate(possible_atoms)}

    for i, atom_type in enumerate(cryst.atom_types):
        # Find the index for this atom type
        idx = atom_index_map.get(atom_type, None)
        if idx is not None:
            # Generate random probabilities and normalize
            probs = np.random.rand(labels_per_atom)
            probs /= probs.sum()

            # Assign the probabilities to the corresponding slice in expanded_labels
            #expanded_labels[i, idx * labels_per_atom: (idx + 1) * labels_per_atom] = probs
            expanded_labels[i] = probs
    cryst.expanded_labels = expanded_labels

    return cryst

# Example usage:
#labels_per_element = 3  # e.g. Fe_1, Fe_2, Fe_3 for each element
#possible_elements = ['Fe', 'Ni']
#interaction_matrix = generate_interaction_matrix(labels_per_element, possible_elements)
#print(interaction_matrix)


