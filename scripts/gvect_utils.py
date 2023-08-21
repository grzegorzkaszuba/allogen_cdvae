import tempfile
import os
import json
from pymatgen.io.cif import CifWriter, CifParser
import numpy as np


panna_cfg = {
    'gvec.ini': os.path.join(os.getcwd(), 'panna_config', 'gvec.ini'),
    'gvect_calculator': os.path.join(os.getcwd(), 'panna_config', 'panna', 'src', 'panna', 'gvect_calculator.py'),
    'gvect_tensor': os.path.join(os.getcwd(), 'panna_config', 'panna', 'gvect_already_computed.dat'),
    'python_call': 'python',
    'gvect_in_cif': os.path.join(os.getcwd(), 'panna_config', 'in_cif', 'struct.cif'),
    'gvect_in': os.path.join(os.getcwd(), 'panna_config', 'in', 'struct.example'),
    'gvect_out': os.path.join(os.getcwd(), 'panna_config', 'out', 'struct.bin'),
}

def cif_to_json(cif_file, path):
    """
    gets a cif file as an input and converts the info
    into a json file, which will be used later for
    Parinnelo vector (gvector) calculation
    """
    # Read the CIF file using pymatgen's CIFParser
    cif_parser = CifParser(cif_file)
    structure = cif_parser.get_structures(primitive=False)[0]

    # Extract relevant information
    atoms_list = []
    for i, site in enumerate(structure):
        atom_id = i + 1
        atom_symbol = site.specie.symbol
        fract_x, fract_y, fract_z = site.frac_coords
        atoms_list.append([atom_id, atom_symbol, [fract_x, fract_y, fract_z], [0, 0, 0]])  # Assume forces as [0, 0, 0]

    # Create the JSON dictionary
    json_dict = {
        "atoms": atoms_list,
        "atomic_position_unit": "cartesian",
        "lattice_vectors": structure.lattice.matrix.tolist(),
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


def gvector (gvector):
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