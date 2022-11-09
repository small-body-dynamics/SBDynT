#!/bin/bash
# submit_array.sh

#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=256M
#SBATCH --array=0-99

# pad the task ID with leading zeros (to get 001, 002, etc.)
CASE_NUM=`printf $SLURM_ARRAY_TASK_ID`

module load python/3.7

python make_series.py $CASE_NUM
