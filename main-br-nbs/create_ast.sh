#!/bin/bash
# submit_array.sh

#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=256M
#SBATCH --array=0-1185

# pad the task ID with leading zeros (to get 001, 002, etc.)
CASE_NUM=`printf $SLURM_ARRAY_TASK_ID`

module load python/3.7

python make_ast_f.py $CASE_NUM

python make_ast_ser.py $CASE_NUM
