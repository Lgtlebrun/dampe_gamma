#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=FilterSucc
#SBATCH --output=log-succFilter-%j.out

# Load ROOT environment (adapt if needed)
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Run the scaling script for a given bin index
time python -u /home/users/l/lebrun/scripts/selection/successiveFiltering_allBins_.py
