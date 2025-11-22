#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=BinFriend
#SBATCH --output=log-BinFriend-%j.out

# Load ROOT environment (adapt if needed)
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc14-opt/setup.sh

# Run the scaling script for a given bin index
time python -u /home/users/l/lebrun/scripts/binFriend_MC_cos_theta.py
