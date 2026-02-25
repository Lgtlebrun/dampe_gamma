#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=FluxMap
#SBATCH --output=log-Flux-%j.out

# Load ROOT environment (adapt if needed)
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh
pip install healpy
pip install --no-deps astropy-healpix
echo "Running flux map"

# Run the exposure map computation script
time python -u /home/users/l/lebrun/scripts/globalFluxMap.py
