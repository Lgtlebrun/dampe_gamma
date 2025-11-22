#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=FilterBin_$1
#SBATCH --output=log-scale-%j-bin$1.out

# Load ROOT environment (adapt if needed)
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Run the scaling script for a given bin index
time python /home/users/l/lebrun/scripts/filter_thetaBin.py $1
