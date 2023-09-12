import os  # used for directory and file handling
from pathlib import Path  # used for directory creation
import shutil  # used for file moving

import numpy as np  # used in update_lammps_data_file for handling arrays
from ase.io import read, write, lammpsdata  # used for reading CIF files and writing LAMMPS files

from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
import subprocess

import os
from pathlib import Path

lammps_pot = os.path.join('lammps_config', 'potentials')
lammps_in = os.path.join('lammps_config', 'lammps_prompt_template')
def update_lammps_data_file(lammps_file, atoms):
    # Retrieve the masses and atom types from the CIF file
    masses = atoms.get_masses()
    atom_types = atoms.get_atomic_numbers()

    # Read the content of the LAMMPS data file
    with open(lammps_file, 'r') as f:
        lines = f.readlines()

    # Find the line number where the atom section startsm
    atom_section_line = None
    for i, line in enumerate(lines):
        if line.strip() == "Atoms":
            atom_section_line = i
            break

    if atom_section_line is not None:
        # Insert mass information before the atom section
        mass_info = ["0 0 0 xy xz yz \n Masses \n\n"]
        unique_atom_types, indices = np.unique(atom_types, return_index=True)
        type = 1
        symbols = []
        for atom_type, index in zip(unique_atom_types, indices):
            atom_index = index + 1
            mass = masses[index]
            symbol = atoms.get_chemical_symbols()[index]
            mass_info.append(f"{type} {mass}  #{symbol}\n")
            type += 1
            symbols.append(symbol + " ")
        mass_info.append("\n")
        lines.insert(atom_section_line, "".join(mass_info))
        lines[0] = "".join(symbols)

    # Overwrite the LAMMPS data file with the updated content
    with open(lammps_file, 'w') as f:
        f.writelines(lines)

    return symbols


def convert_cif_to_lammps(source_dir, target_dir):
    # Define target directory
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Get a list of all CIF files in the source directory
    cif_files = [f for f in os.listdir(source_dir) if f.endswith('.cif')]

    # Convert each CIF file to LAMMPS data format
    for cif_file in cif_files:
        cif_path = os.path.join(source_dir, cif_file)
        data_path = os.path.join(target_dir, cif_file)

        # Load CIF file
        atoms = read(cif_path)

        # Save LAMMPS data file
        lammps_file = os.path.splitext(data_path)[0] + '.data'
        write(lammps_file, atoms, format='lammps-data')

        # Update the LAMMPS data file with masses
        update_lammps_data_file(lammps_file, atoms)

        #print(f"Converted {cif_file} to {lammps_file}")

    # Move the LAMMPS data files to the target directory
    file_names = os.listdir(source_dir)

    for file_name in file_names:
        if file_name.endswith('.data'):
            shutil.move(os.path.join(source_dir, file_name), target_dir)




def extract_initial_final_energy(filename):
    """Extracts the initial and final energy from a lammps log file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Initialize energy values as None
    initial_energy = None
    final_energy = None
    # Iterate over the lines in the log file
    for i, line in enumerate(lines):
        # If the line contains "Energy initial", the next line contains the energy value
        if "Energy initial" in line:
            energy_line = lines[i+1]
            energies = energy_line.split()
            initial_energy = float(energies[0])
            final_energy = float(energies[-1])

    if initial_energy is None or final_energy is None:
        raise ValueError(f"Could not find 'Energy initial' in {filename}")

    return initial_energy, final_energy



def modify_file(file_path, lines_to_search, replacement_lines):
    """Modifies the content of a file by searching and replacing lines."""
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace the lines in the file
    for line in lines_to_search:
        for i, file_line in enumerate(lines):
            if line in file_line:
                for replace_line in replacement_lines:
                    if replace_line[0] == line:
                        lines[i] = replace_line[1] + "\n"
    return lines
def lmp_energy_calculator(source_dir, target_dir, lammps_cfg, silent=False, pot_type='meam'):
    """minimises the structures and calculates the energy"""
    (pot, pot_file, lammps_command, lammps_inputs_dir) = (
        lammps_cfg['pot'],
        lammps_cfg['pot_file'],
        lammps_cfg['lammps_path'],
        lammps_cfg['input_template']
    )

    assert pot_type in ['meam', 'eam'], 'Wrong potential type: pot_type has to be either mean or eam'
    if pot_type == 'meam':
        pot = 'mean'
    elif pot_type == 'eam':
        pot = 'eam/alloy'

    initial_energies = {}
    final_energies = {}
    # Defining the paths for our directories
    relaxed_structures_dir = target_dir
    folder_path = source_dir

    # Creating the directories if they do not exist
    Path(relaxed_structures_dir).mkdir(parents=True, exist_ok=True)

    # list of available structures
    #file_names = os.listdir(folder_path)
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.data')]

    # modify read_data and run the minimisation
    input_file = os.path.join(lammps_inputs_dir, 'in.minimize')

    # Specify the lines to search for in the input file
    search_lines = ['pair_style', 'pair_coeff', 'read_data', 'wd']

    # minimise and save prop to file
    for name in file_names:
        with open(os.path.join(folder_path, name)) as file:
            lines = file.readlines()
            elms = lines[0]
            if pot_type == 'meam':
                if 'Fe' not in elms:
                    pot_path = os.path.join(os.getcwd(), pot_file, 'NiCr')
                elif 'Ni' not in elms:
                    pot_path = os.path.join(os.getcwd(), pot_file, 'FeCr')
                elif 'Cr' not in elms:
                    pot_path = os.path.join(os.getcwd(), pot_file, 'FeNi')
                else:
                    pot_path = os.path.join(os.getcwd(), pot_file, 'NiFeCr')
                pair_coeff_call = 'pair_coeff * * ' + os.path.join(pot_path, 'library.meam') + f' {elms} ' + os.path.join(pot_path, f'{pot_path}.meam') + f' {elms}'
            else:
                pot_path = os.path.join(os.getcwd(), pot_file, 'NiFeCr.eam.alloy')
                pair_coeff_call = 'pair_coeff * * ' + pot_path + f' {elms}'

        modification_lines = [
            ('pair_style', f'pair_style {pot}'),
            ('pair_coeff', pair_coeff_call),
            #('pair_coeff', 'pair_coeff * * ' + os.path.join(os.getcwd(), pot_file) + f' {elms}'),
            ("read_data", f"read_data {os.path.join(folder_path, name)}"),
            ('wd', f'write_data {os.path.join(relaxed_structures_dir, name)}')]

        lmp_task_dir = os.path.join(os.getcwd(), lammps_inputs_dir)
        modified_lines = modify_file(input_file, search_lines, modification_lines)
        new_input_file = os.path.join(lmp_task_dir, f'in.{name}_min')

        with open(new_input_file, 'w') as file:
            file.writelines(modified_lines)


        root_dir = os.getcwd()

        if not silent:
            print('printing lammps command')
            print(f'{lammps_command} -in {os.path.join(lmp_task_dir, f"in.{name}_min")}')

        # controlling the directories

        os.chdir(lmp_task_dir)
        assert 'RELAXATIONTASKHERE' in os.listdir()

        # run the simulation and get the energy
        if silent:
            subprocess.call(f'{lammps_command} -in {os.path.join(lmp_task_dir, f"in.{name}_min")}', stdout=open(os.devnull, 'wb'))
        else:
            os.system(f'{lammps_command} -in {os.path.join(lmp_task_dir, f"in.{name}_min")}')
        initial_energy, final_energy = extract_initial_final_energy('log.lammps')
        initial_energies[name.split('.')[0]] = initial_energy
        final_energies[name.split('.')[0]] = final_energy

        os.remove(new_input_file), os.remove('log.lammps')
        os.chdir(root_dir)

    #return prop, df
    return initial_energies, final_energies

def lmp_elastic_calculator(source_dir, lammps_cfg, silent=False, pot_type='meam'):
    """minimises the structures and calculates the elastic vector"""

    (pot, pot_file, lammps_command, lammps_inputs_dir) = (
        lammps_cfg['pot'],
        lammps_cfg['pot_file'],
        lammps_cfg['lammps_path'],
        lammps_cfg['input_template']
    )

    elastic_vectors = {}
    # Defining the paths for our directories
    folder_path = source_dir # todo this could be relaxed-structures


    # list of available structures
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.data')]

    # input files to be modified
    input_template_init = os.path.join(os.getcwd(), lammps_inputs_dir, 'elastic', 'init.mod')
    input_template_pot = os.path.join(os.getcwd(), lammps_inputs_dir, 'elastic', 'potential.mod')
    output_file = os.path.join(os.getcwd(), lammps_inputs_dir, 'elastic_output')

    # Specify the lines to search for in the input files
    search_lines_init = ['read_data']
    search_lines_pot = ['pair_style', 'pair_coeff']

    for name in file_names:
        with open(os.path.join(folder_path, name)) as file:
            lines = file.readlines()
            elms = lines[0]

        modification_init = [("read_data", f"read_data {os.path.join(folder_path, name)}")]
        modification_pot = [
            ('pair_style', f'pair_style {pot}'),
            ('pair_coeff', 'pair_coeff * * ' + os.path.join(os.getcwd(), pot_file) + f' {elms}')
        ]

        # modify the files
        modified_lines_init = modify_file(input_template_init, search_lines_init, modification_init)
        modified_lines_pot = modify_file(input_template_pot, search_lines_pot, modification_pot)


        with open(os.path.join(output_file, 'init.mod'), 'w') as file:
            file.writelines(modified_lines_init)
        with open(os.path.join(output_file, 'potential.mod'), 'w') as file:
            file.writelines(modified_lines_pot)

        # Move to prompt directory to capture all the auxiliary files
        root_cwd = os.getcwd()
        os.chdir(output_file)
        assert 'PREDICTIONTASKHERE' in os.listdir()
        # run the simulation
        if silent:
            subprocess.call(f"{lammps_command} -in {os.path.join(os.getcwd(), 'in.elastic')}", stdout=open(os.devnull, 'wb'))
        else:
            os.system(f"{lammps_command} -in {os.path.join(os.getcwd(), 'in.elastic')}")

        # extract elastic_vector from the log file
        elastic_vector = extract_elastic_vector("log.lammps")
        elastic_vectors[name.split('.')[0]] = elastic_vector

        # clean up
        os.remove(os.path.join(output_file, 'init.mod'))
        os.remove(os.path.join(output_file, 'potential.mod'))
        os.remove('log.lammps')
        os.remove("restart.equil")

        os.chdir(root_cwd)
        #print(f"{name} DONE")

    return elastic_vectors


def extract_elastic_vector(log_file):
    """Extract elastic vector from log file"""
    with open(log_file, "r") as file:
        lines = file.readlines()

    for line in reversed(lines):
        if "cdvae" in line:
            # assuming the elastic vector is in square brackets
            elastic_vector = eval(line.split("cdvae ")[1].strip())
            return elastic_vector

    raise Warning('The value of elastic vector not found in lammps logs')
    return None

def extract_element_types(f):
    return f.readlines()[0].strip().split(' ')

def lammps_data_to_cif(structure_names, raw_path, relaxed_path, savedir=None):
    cif_strings = []
    for name in structure_names:
        # Read LAMMPS data file
        atoms = lammpsdata.read_lammps_data(os.path.join(relaxed_path, name+'.data'), style='atomic')

        # Manually map LAMMPS atom types to element symbols

        with open(os.path.join(raw_path, name+'.data'), 'r') as f:
            element_symbols = extract_element_types(f)

        atom_types = atoms.get_array('type')
        atom_symbols = [element_symbols[atom_type - 1] for atom_type in atom_types]
        atoms.set_chemical_symbols(atom_symbols)

        # Convert ASE Atoms to PyMatGen Structure
        structure = AseAtomsAdaptor().get_structure(atoms)

        # Write CIF file using CifWriter
        cif_writer = CifWriter(structure)
        cif_strings.append(cif_writer.__str__())
        if savedir is not None:
            structure.to(filename=os.path.join(savedir, name+'.cif'), fmt='cif')
    return cif_strings
