#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=Split_1B
#SBATCH --output=log-split1bin-%j.out

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc14-opt/setup.sh

# Number of bins
N_BINS=20
time python -u /home/users/l/lebrun/scripts/split_MC_cos_theta.py $N_BINS $1
