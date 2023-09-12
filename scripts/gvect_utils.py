import tempfile
import os
import json
from pymatgen.io.cif import CifWriter, CifParser
import numpy as np
import subprocess
from scipy.optimize import linear_sum_assignment
import math
import yaml


with open('subprocess_calls.yaml', 'r') as file:
    python_call = yaml.safe_load(file).get('python_call')

panna_cfg = {
    'gvec.ini': os.path.join(os.getcwd(), 'panna_config', 'gvec.ini'),
    'gvect_calculator': os.path.join(os.getcwd(), 'panna_config', 'panna', 'src', 'panna', 'gvect_calculator.py'),
    'gvect_tensor': os.path.join(os.getcwd(), 'panna_config', 'panna', 'gvect_already_computed.dat'),
    'python_call': python_call,
    'gvect_in_cif': os.path.join(os.getcwd(), 'panna_config', 'in_cif', 'struct.cif'),
    'gvect_in': os.path.join(os.getcwd(), 'panna_config', 'in', 'struct.example'),
    'gvect_out': os.path.join(os.getcwd(), 'panna_config', 'out', 'struct.bin'),
}

def save_structure_to_file(struct: 'Structure', path: str, save_as_cartesian=False):
    if not save_as_cartesian:
        writer = CifWriter(struct)
        writer.write_file(path)
    else:
        cartesian_coords = struct.cart_coords*struct.lattice.lengths
        cartesian_structure = Structure(struct.lattice, struct.species, cartesian_coords,
                                        coords_are_cartesian=True, site_properties=struct.site_properties)

        # Now, write this structure with Cartesian coordinates to a CIF file.
        writer = CifWriter(cartesian_structure)
        writer.write_file(path)

def gvect_distance(struct1, struct2, panna_cfg, anonymous=False):
    atomic_numbers, counts = np.unique(struct1.atomic_numbers, return_counts=True)
    ctrl_at, ctrl_ct = np.unique(struct2.atomic_numbers, return_counts=True)
    if not anonymous:
        assert np.all(atomic_numbers == ctrl_at) and np.all(counts == ctrl_ct),\
            'Gvect similarity can only be used if same composition is ensured!'
    gvects = []
    for struct in [struct1, struct2]:
        if anonymous:
            modified_structure = deepcopy(struct)

            # Get lattice and fractional coordinates from the original structure
            lattice = modified_structure.lattice
            fractional_coords = [site.frac_coords for site in modified_structure]

            # Reinitialize the Structure with Zn atoms while maintaining the original lattice and coordinates
            struct = Structure(lattice, ["Zn"] * len(modified_structure), fractional_coords)
        cif_path = panna_cfg['gvect_in_cif']
        ex_path = panna_cfg['gvect_in']
        save_structure_to_file(struct, cif_path)
        cif_to_json(cif_path, ex_path)
        out_dir = panna_cfg['gvect_out']
        modify_gvec(panna_cfg['gvec.ini'], cif_path, ex_path.split('struct.')[0], out_dir.split('struct.')[0])
        # create the corresponding config file
        # genrate corresponding gvectors (files with .bin extention)
        subprocess.call([f'{panna_cfg["python_call"]}', f'{panna_cfg["gvect_calculator"]}', '--config', f'{panna_cfg["gvec.ini"]}'])
        os.remove('gvect_already_computed.dat')

        gvect_tensor = gvector(panna_cfg['gvect_out'])
        gvects.append(gvect_tensor)
    dis_mat = []
    s1, s2 = gvects[0], gvects[1]
    for i in range(len(struct2)):
        dis_mat.append(np.linalg.norm(s1[i:i + 1] - s2, axis=1))
    min_dis = []


    a = np.copy(dis_mat)
    for j in range(len(a)):
        a[j].sort()
        min_dis.append(a[j][0])

    total_dis = np.sum(min_dis)
    mean_dis = np.mean(min_dis)

    cost_matrix = np.linalg.norm(s1[:, None, :] - s2[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    mean_dis = total_cost/len(row_ind)

    return mean_dis

def cif_to_json(cif_file, path):
    """
    gets a cif file as an input and converts the info
    into a json file, which will be used later for
    Parinnelo vector (gvector) calculation
    """
    # Read the CIF file using pymatgen's CIFParser
    cif_parser = CifParser(cif_file)
    structure = cif_parser.get_structures(primitive=False)[0]

    lattice_matrix = structure.lattice.matrix

    # Extract relevant information
    # Extract relevant information
    atoms_list = []
    for i, site in enumerate(structure):
        atom_id = i + 1
        atom_symbol = site.specie.symbol
        fract_x, fract_y, fract_z = site.frac_coords

        # Convert fractional coordinates to Cartesian coordinates
        cartesian_coords = fract_x * lattice_matrix[0] + fract_y * lattice_matrix[1] + fract_z * lattice_matrix[2]

        atoms_list.append([atom_id, atom_symbol, cartesian_coords.tolist(), [0, 0, 0]])  # Assume forces as [0, 0, 0]

    # Create the JSON dictionary
    json_dict = {
        "atoms": atoms_list,
        "atomic_position_unit": "cartesian",
        "lattice_vectors": lattice_matrix.tolist(),
        "energy": [0, "Ha"]
    }

    # Convert to JSON string
    json_str = json.dumps(json_dict, indent=4)

    # Write JSON to the output file with .example extension
    with open(path, 'w') as f:
        f.write(json_str)


def modify_gvec(file_path, cif_file, ex_directory=None, bin_directory=None):
    """
    Modifies the config file (gvec.ini) for each
    cif file.
    """
    # get the atom types
    cif_parser = CifParser(cif_file)
    structure = cif_parser.get_structures(primitive=False)[0]
    atom_types = sorted(set([site.specie.symbol for site in structure]))

    # Read the content of the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the line that starts with "species"
    for i, line in enumerate(lines):
        if line.strip().startswith("species"):
            # Modify the line with the new species list
            lines[i] = f"""species = {", ".join(atom_types)}\n"""
        if ex_directory is not None and line.strip().startswith("input_json_dir"):
            lines[i] = f"input_json_dir = {ex_directory}\n"
        if bin_directory is not None and line.strip().startswith("output_gvect_dir"):
            lines[i] = f"output_gvect_dir = {os.path.join(bin_directory)}\n"

    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)


def gvector(gvector):
    with open(gvector, "rb") as binary_file:
        bin_version = int.from_bytes(binary_file.read(4),
                                     byteorder='little',
                                    signed=False)
        if bin_version != 0:
            print("Version not supported!")
            exit(1)
        # converting to int to avoid handling little/big endian
        flags = int.from_bytes(binary_file.read(2),
                               byteorder='little',
                               signed=False)
        n_atoms = int.from_bytes(binary_file.read(4),
                                 byteorder='little',
                                 signed=False)
        g_size = int.from_bytes(binary_file.read(4),
                                byteorder='little',
                                signed=False)
        payload = binary_file.read()
        data = np.frombuffer(payload, dtype='<f4')
        en = data[0]
        gvect_size = n_atoms * g_size
        spec_tensor = np.reshape((data[1:1+n_atoms]).astype(np.int32),
                                 [1, n_atoms])
        gvect_tensor = np.reshape(data[1+n_atoms:1+n_atoms+gvect_size],
                    [n_atoms, g_size])
    return (gvect_tensor)


from copy import deepcopy
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice


def create_zinc_supercell(target_volume, structure_type="fcc", template_atom='Zn'):
    """
    Returns a Zn supercell (either FCC or BCC) with 54 atoms and approximately the target volume.
    """
    volume_per_atom = target_volume / 54

    if structure_type == "fcc":
        # Compute the lattice constant "a" for the desired ratio
        a = (2 * volume_per_atom / math.sqrt(2)) ** (1 / 3)

        # Construct the FCC lattice
        lattice = Lattice([[a, 0, 0], [0, a, 0], [0, 0, a * math.sqrt(2)]])
        frac_coords = [
            [0, 0, 0],
            [0, .5, .5]
        ]

        #frac_coords = [
        #    [0, 0, 0],
        #    [0.5, 0.5, 0],
        #    [0.5, 0, 0.5],
        #    [0, 0.5, 0.5]
        #]

        structure = Structure(lattice, [template_atom] * 2, frac_coords)
        structure.make_supercell([3, 3, 3])

    elif structure_type == "bcc":
        lattice_constant = (volume_per_atom * 2) ** (1 / 3)

        lattice = Lattice.cubic(lattice_constant)
        structure = Structure(lattice, [template_atom, template_atom], [[0, 0, 0], [0.5, 0.5, 0.5]])
        structure.make_supercell([3, 3, 3])

    else:
        raise ValueError("Invalid structure_type provided. Choose either 'fcc' or 'bcc'.")

    return structure


def template_gdist(compared_structure, structure_type, panna_config, lattice_constant=None):
    """
    Compares a modified structure (with all atoms replaced by Zn) to a perfect Zn template
    (either FCC or BCC) using my_similarity_function.
    """

    # If lattice_constant is provided, infer the target volume
    if lattice_constant:
        target_volume = lattice_constant ** 3 * (4 if structure_type == "fcc" else 2) * 54
    else:
        target_volume = compared_structure.volume

    zinc_template = create_zinc_supercell(target_volume, structure_type)
    # Modify the compared_structure: replace all atoms with Zn
    modified_structure = deepcopy(compared_structure)

    # Get lattice and fractional coordinates from the original structure
    lattice = modified_structure.lattice
    fractional_coords = [site.frac_coords for site in modified_structure]

    # Reinitialize the Structure with Zn atoms while maintaining the original lattice and coordinates
    modified_structure = Structure(lattice, ["Zn"] * len(modified_structure), fractional_coords)


    # Compare the structures using my_similarity_function
    result = gvect_distance(modified_structure, zinc_template, panna_config)

    return result

# Remember to import and integrate my_similarity_function for this to work.

