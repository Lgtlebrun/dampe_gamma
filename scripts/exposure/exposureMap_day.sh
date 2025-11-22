#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=ExposureMap_${1}_${2}
#SBATCH --output=log-exposure-%j.out

# Load ROOT environment (adapt if needed)
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

echo "Running exposure map computation for date $1 and energy $2 GeV"

# Run the exposure map computation script
time python -u /home/users/l/lebrun/scripts/exposureMap.py "$1" "$2"
