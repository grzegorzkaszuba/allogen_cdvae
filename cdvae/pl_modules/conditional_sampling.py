import torch
from pymatgen.core.periodic_table import Element
from typing import List




def atom_marginalization(possible_atoms: List[str],
                         probability_distribution: torch.tensor,
                         element_count: torch.tensor) -> (torch.tensor, torch.tensor):
    # 1. Convert the possible_atoms list into atom indices
    atom_indices = [Element(atom).Z - 1 for atom in possible_atoms]  # Subtracting 1 for 0-based indexing
    atom_indices = torch.tensor(atom_indices, dtype=torch.long)

    # 2. Mask the tensors
    # We use advanced indexing to select only the relevant dimensions
    # >>>> Note from me! As we really don't use the unmasked probability distributions - it's just format of network output
    # >>>> I took the freedom to overwrite the names of those original variables for clarity and copy them
    probability_distribution = probability_distribution.index_select(2, atom_indices).clone()
    element_count = element_count.index_select(1, atom_indices).clone()

    probability_distribution = torch.nn.functional.normalize(probability_distribution, p=1, dim=2)
    element_count = element_count
    sampled_atoms = sample_atom_types(probability_distribution)
    element_state = element_status(sampled_atoms, element_count)
    discrepancies = element_discrepancies(sampled_atoms, element_count)
    while not torch.all(element_state >= 0):
        sampled_atoms, element_state, probability_distribution = resolve_overrepresented_atoms(
            probability_distribution, sampled_atoms, element_state, element_count)
        element_state = element_status(sampled_atoms, element_count)
        discrepancies = element_discrepancies(sampled_atoms, element_count)

    return sampled_atoms
    out = torch.zeros(dtype=sampled_atoms.dtype)
    out[atom_indices] += sampled_atoms
    return out


def element_status(sampled_atoms: torch.tensor, target_formula: torch.tensor) -> torch.tensor:
    """
    Determines the status of each atom type for structures in the batch based on the comparison
    between the sampled atoms and the target summary formula.

    Args:
    - sampled_atoms (torch.tensor): Tensor containing the sampled atom types.
                                    Shape: [batch_size, atoms_per_structure]
    - target_formula (torch.tensor): Tensor containing the target atom counts for each atom type.
                                     Shape: [batch_size, number_of_possible_atoms]

    Returns:
    - torch.tensor: Tensor containing the status for each atom type (-1: overrepresented,
                    0: underrepresented or exact, 1: resolved/frozen).
                    Shape: [batch_size, number_of_possible_atoms]
    """

    batch_size, number_of_possible_atoms = target_formula.shape

    # Counting the number of each atom type in the sampled configuration
    sampled_counts = torch.zeros(batch_size, number_of_possible_atoms, dtype=torch.long)
    for i in range(batch_size):
        sampled_counts[i] = torch.bincount(sampled_atoms[i], minlength=number_of_possible_atoms)

    # Determining the atom status
    element_status_tensor = torch.zeros_like(target_formula)

    overrepresented_mask = sampled_counts > target_formula
    element_status_tensor[overrepresented_mask] = -1

    # For now, we won't mark any atom as resolved/frozen (status=1).
    # This will be handled in the next functions as we progress through the iterations.

    return element_status_tensor


# Use the helper functions that we defined earlier:

def resolve_overrepresented_atoms(prob_distribution: torch.tensor, sampled_atoms: torch.tensor, overrepresentation_mask: torch.tensor, ground_truth_counts: torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor):
    for i in range(ground_truth_counts.size(0)):
        overrepresentation = (overrepresentation_mask[i] < 0).nonzero(as_tuple=True)[0]
        for atom_type in overrepresentation:
            num_to_replace = -overrepresentation_mask[i, atom_type].item()
            atom_indices = (sampled_atoms[i] == atom_type).nonzero(as_tuple=True)[0]
            to_replace_indices = torch.multinomial((1 - prob_distribution[i][atom_indices, atom_type]), num_to_replace, replacement=False)
            replacement_atoms = torch.multinomial(prob_distribution[i][atom_indices[to_replace_indices]].index_fill(1, atom_type, 0), num_to_replace)
            sampled_atoms[i, atom_indices[to_replace_indices]] = replacement_atoms
            prob_distribution[i, :, atom_type] = 0  # Updating the entire batch's atom type probability to 0.
    overrepresentation_mask = element_status(sampled_atoms, ground_truth_counts)
    return sampled_atoms, overrepresentation_mask, prob_distribution


# Helper functions:

def sample_atom_types(prob_distribution: torch.tensor) -> torch.tensor:
    batch_size, atoms_per_case, n_elements = prob_distribution.shape
    flat_probs = prob_distribution.view(-1, n_elements)  # Flatten to 2D
    flat_samples = torch.multinomial(flat_probs, 1).squeeze()
    return flat_samples.view(batch_size, atoms_per_case)

def element_discrepancies(sampled_atoms: torch.tensor, ground_truth_counts: torch.tensor) -> torch.tensor:
    sampled_counts = torch.zeros_like(ground_truth_counts)
    for i in range(ground_truth_counts.size(1)):
        sampled_counts[:, i] = (sampled_atoms == i).sum(dim=1)
    return ground_truth_counts - sampled_counts

if __name__ == '__main__':
# Testing:

    possible_atoms = ['H', 'O', 'C']

    # Convert the possible_atoms list into atom indices using pymatgen's Element
    atom_indices = [Element(atom).Z - 1 for atom in possible_atoms]

    # Creating a padded probability distribution
    max_index = max(atom_indices)
    prob_distributions_raw = torch.tensor([
        [[0.7, 0.2, 0.1], [0.8, 0.1, 0.1], [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.1, 0.8, 0.1]],
        [[0.98, 0.01, 0.01], [0.9, 0.02, 0.08], [0.98, 0.01, 0.01], [0.4, 0.3, 0.3], [0.4, 0.2, 0.4]]
    ])

    prob_distributions = torch.zeros(*prob_distributions_raw.shape[:-1], max_index + 1)
    for idx, atom_idx in enumerate(atom_indices):
        prob_distributions[..., atom_idx] = prob_distributions_raw[..., idx]


    def pad_tensor(tensor, index_list, total_elements=118):
        """
        Function to pad tensor with zeros, based on a given list of indices.
        """
        padded_tensor = torch.zeros(tensor.size(0), total_elements, dtype=tensor.dtype)
        for idx, original_idx in enumerate(index_list):
            padded_tensor[:, original_idx] = tensor[:, idx]
        return padded_tensor

    # In the test case:

    # Original and target atom counts for each structure
    element_counts = torch.tensor([
        [2, 1, 2],  # H2O2: 2 Hydrogens, 1 Oxygen, 2 Carbons
        [3, 2, 0]   # H3O2: 3 Hydrogens, 2 Oxygens, 0 Carbons
    ])

    element_counts = pad_tensor(element_counts, [0, 5, 7])


    sampled_results = atom_marginalization(possible_atoms, prob_distributions, element_counts)

    print("\nSampled Atoms:")
    print(sampled_results)