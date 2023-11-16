#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=100   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=512M   # memory per CPU core
#SBATCH --mail-user=dallinspencer@gmail.com   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
##SBATCH -C 'kepler'   # features syntax (use quotes): -C 'a&b&c&d'
#SBATCH --mail-type=FAIL

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#SBATCH --qos=physics --partition=physics2

module load python/3.7
module load gcc/8
module load openmpi/3.1 
module load python-mpi4py/3.0
export LD_LIBRARY_PATH=/apps/gcc/8.3.0/lib64

mpiexec -n 100 python -W ignore fft_multi.py AstFam_families

exit 0
