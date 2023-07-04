import os
import tempfile
import numpy as np
from ase.io import read, write
from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import lammpsdata
import json
import subprocess
import re

lammps = "lmp"

def convert_cif_to_lammps(directory):
    # Get a list of all CIF files in the directory
    cif_files = [f for f in os.listdir(directory) if f.endswith('.cif')]
    Path('LMP/generated/lammps-data').mkdir(parents=True, exist_ok=True)

    # Convert each CIF file to LAMMPS data format
    for cif_file in cif_files:
        cif_path = os.path.join(directory, cif_file)

        # Load CIF file
        atoms = read(cif_path)

        # Save LAMMPS data file
        lammps_file = os.path.splitext(cif_path)[0] + '.data'
        write(lammps_file, atoms, format='lammps-data')

        # Update the LAMMPS data file with masses
        update_lammps_data_file(lammps_file, atoms)

        print(f"Converted {cif_file} to {lammps_file}")
    os.system('mv LMP/generated/*.data LMP/generated/lammps-data')

