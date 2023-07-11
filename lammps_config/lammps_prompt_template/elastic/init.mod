# metal units, elastic constants in GPa
units           metal
variable cfac equal 1.0e-4
variable cunits string GPa
variable up equal 1.0e-6
variable atomjiggle equal 1.0e-5


# Define minimization parameters
variable etol equal 0.0
variable ftol equal 1.0e-10
variable maxiter equal 100
variable maxeval equal 1000
variable dmax equal 1.0e-2

boundary        p p p
read_data	Cr_relaxed.lmp

# Need to set mass to something, just to satisfy LAMMPS
mass 1 1.0e-20
