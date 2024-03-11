#!/bin/bash
# submit_array.sh

#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=512M
#SBATCH --array=0-106

# pad the task ID with leading zeros (to get 001, 002, etc.)
CASE_NUM=`printf $SLURM_ARRAY_TASK_ID`

module load python/3.7

python integrate_sh.py Test $CASE_NUM

mv /tmp/archive_$CASE_NUM.bin ../data/Test/$CASE_NUM/archive.bin
