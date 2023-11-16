#!/bin/bash
# submit_array.sh

#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=512M
#SBATCH --array=0-2000

# pad the task ID with leading zeros (to get 001, 002, etc.)
CASE_NUM=`printf $SLURM_ARRAY_TASK_ID`

module load python/3.7

python make_f.py AstFam_families $CASE_NUM 8planet

mv /tmp/archive_ias15_$CASE_NUM.bin Sims/AstFam_families/$CASE_NUM/archive_hires.bin

#python make_series.py $CASE_NUM

