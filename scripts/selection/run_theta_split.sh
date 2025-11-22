#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=Split_Theta
#SBATCH --array=0-19
#SBATCH --output=log-split-%A_%a.out

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc14-opt/setup.sh

# Number of bins
N_BINS=20
BIN_INDEX=$SLURM_ARRAY_TASK_ID

time python -u /home/users/l/lebrun/scripts/split_MC_cos_theta.py $N_BINS $BIN_INDEX
