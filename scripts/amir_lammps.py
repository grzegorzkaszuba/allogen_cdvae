import os  # used for directory and file handling
from pathlib import Path  # used for directory creation
import shutil  # used for file moving

import numpy as np  # used in update_lammps_data_file for handling arrays
from ase.io import read, write  # used for reading CIF files and writing LAMMPS files

lammps_pot = os.path.join('lammps_config', 'potentials', 'NiFeCr.eam.alloy')
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


def convert_cif_to_lammps(source_dir):
    # Define target directory
    target_dir = os.path.join(source_dir, 'lammps_data')
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


import os
import json
import subprocess
import pandas as pd
from pathlib import Path

def extract_energy_from_log(filename):
    """Extracts the energy and other parameters from a lammps log file and returns them in a DataFrame."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # The data dictionary will store our data
    data = {}

    # Iterate over the lines in the log file
    for line in lines:
        # Split the line into words
        words = line.split()

        # If the line contains "Step", "Energy", etc., it's a header line
        if "Step" in words or "Energy" in words or "Temp" in words:
            headers = words

            # Initialize an empty list for each header
            for header in headers:
                data[header] = []
        elif len(words) == len(headers):
            # If the line contains numeric data, add the data to our dictionary
            for i, word in enumerate(words):
                data[headers[i]].append(float(word))

    # Convert the dictionary to a pandas DataFrame and return it
    df = pd.DataFrame(data)
    return df


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
def lmp_energy_calculator(source_dir, pot, pot_name, lammps_command, lammps_inputs_dir):
    """minimises the structures and calculates the energy"""

    # Defining the paths for our directories
    relaxed_structures_dir = os.path.join(source_dir, "lammps_data", "relaxed-structures")
    finalDB_dir = os.path.join(source_dir, "lammps_data", "finalDB")
    folder_path = os.path.join(source_dir, "lammps_data")

    # Creating the directories if they do not exist
    Path(relaxed_structures_dir).mkdir(parents=True, exist_ok=True)
    Path(finalDB_dir).mkdir(parents=True, exist_ok=True)

    energy_json_file = os.path.join(finalDB_dir, "energy.json")

    if os.path.isfile(energy_json_file):
        # Load the existing data from the JSON file
        with open(energy_json_file, "r") as file:
            prop = json.load(file)
        os.remove(energy_json_file)
    else:
        prop = {}

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

        modification_lines = [
            ('pair_style', f'pair_style {pot}'),
            ('pair_coeff', 'pair_coeff * * ' + os.path.join(os.getcwd(), pot_name) + f' {elms}'),
            ("read_data", f"read_data {os.path.join(folder_path, name)}"),
            ('wd', f'write_data {os.path.join(relaxed_structures_dir, name)}')]

        modified_lines = modify_file(input_file, search_lines, modification_lines)
        new_input_file = os.path.join(lammps_inputs_dir, f'in.{name}_min')

        with open(new_input_file, 'w') as file:
            file.writelines(modified_lines)

        print('printing lammps command')
        print(f'{lammps_command} -in {os.path.join(os.getcwd(), lammps_inputs_dir, f"in.{name}_min")}')

        # run the simulation and get the energy
        os.system(f'{lammps_command} -in {os.path.join(os.getcwd(), lammps_inputs_dir, f"in.{name}_min")}')



        #df = extract_energy_from_log('log.lammps')
        #prop[name] = df.iloc[-1]['Energy']

    # save the data
    with open(energy_json_file, "w") as file:
        json.dump(prop, file)

    #return prop, df


def lmp_elastic_calculator(source_dir, pot, pot_name, lammps_command, lammps_inputs_dir):
    """minimises the structures and calculates the elastic vector"""

    # Defining the paths for our directories
    finalDB_dir = os.path.join(source_dir, "lammps_data", "finalDB")
    folder_path = os.path.join(source_dir, "lammps_data")

    # Creating the directories if they do not exist
    Path(finalDB_dir).mkdir(parents=True, exist_ok=True)

    elastic_json_file = os.path.join(finalDB_dir, "elastic.json")

    if os.path.isfile(elastic_json_file):
        # Load the existing data from the JSON file
        with open(elastic_json_file, "r") as file:
            prop = json.load(file)
        os.remove(elastic_json_file)
    else:
        prop = {}

    # list of available structures
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.data')]

    # input files to be modified
    input_file_init = os.path.join(lammps_inputs_dir, 'elastic', 'init.mod')
    input_file_pot = os.path.join(lammps_inputs_dir, 'elastic', 'potential.mod')
    output_file = os.path.join(lammps_inputs_dir, 'elastic_output')

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
            ('pair_coeff', 'pair_coeff * * ' + os.path.join(os.getcwd(), pot_name) + f' {elms}')
        ]

        # modify the files
        modified_lines_init = modify_file(input_file_init, search_lines_init, modification_init)
        modified_lines_pot = modify_file(input_file_pot, search_lines_pot, modification_pot)



        new_input_file_init = os.path.join(lammps_inputs_dir, 'init.mod')
        new_input_file_pot = os.path.join(lammps_inputs_dir, 'potential.mod')

        shutil.copy(input_file_init, os.path.join(output_file, 'init.mod'))
        shutil.copy(input_file_pot, os.path.join(output_file, 'potential.mod'))

        # Write the modified files back
        modify_file(os.path.join(output_file, 'init.mod'), search_lines_init, modification_init)
        # modify potential
        modify_file(os.path.join(output_file, 'potential.mod'), search_lines_pot, modification_pot)

        # Move to prompt directory to capture all the auxiliary files
        root_cwd = os.getcwd()
        os.chdir(output_file)
        # run the simulation
        os.system(f"{lammps_command} -in {os.path.join(os.getcwd(), 'in.elastic')}")

        # extract elastic_vector from the log file
        elastic_vector = extract_elastic_vector("log.lammps")
        os.chdir(root_cwd)

        # clean up
        os.remove(new_input_file_init)
        os.remove(new_input_file_pot)

        prop.update({f"{name.split('.')[0]}": elastic_vector})
        print(f"{name} DONE")

    os.remove("log.lammps")
    os.remove("restart.equil")

    with open(elastic_json_file, "a") as file:
        json.dump(prop, file)


def extract_elastic_vector(log_file):
    """Extract elastic vector from log file"""
    with open(log_file, "r") as file:
        lines = file.readlines()

    for line in reversed(lines):
        if "cdvae" in line:
            # assuming the elastic vector is in square brackets
            elastic_vector = eval(line.split("cdvae ")[1].strip())
            return elastic_vector


    return None  # in case the elastic vector was not found in the log file
