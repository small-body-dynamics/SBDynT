#!/bin/bash
# submit_array.sh

#SBATCH --ntasks=200
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=512M

# pad the task ID with leading zeros (to get 001, 002, etc.)
CASE_NUM=`printf $SLURM_ARRAY_TASK_ID`

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

##SBATCH --qos=physics --partition=physics2

module load python/3.7
module load gcc/8
module load openmpi/3.1
module load python-mpi4py/3.0

export LD_LIBRARY_PATH=/apps/gcc/8.3.0/lib64

mpiexec -n 200 python integrate_multi.py YHuang_freei
