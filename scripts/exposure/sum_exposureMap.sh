#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --job-name=SumExposureMap
#SBATCH --output=log-SumExposure-%j.out

# Load ROOT environment (adapt if needed)
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

echo "Running sum of exposure maps for energy bin : $1"

# Run the exposure map computation script
time python -u /home/users/l/lebrun/scripts/exposure/sumExposure.py "$1" 
