#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00              # Adjust time for processing full year
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                    # Increased memory for all data
#SBATCH --job-name=ExposureMapTarget
#SBATCH --output=log-expTarget-%j.out

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

YEAR="2023"
ENERGY=19 # Energy bin
TARGET="Vela"

echo "Running exposure map for all data in $YEAR at Energy bin ${ENERGY}"

time python -u /home/users/l/lebrun/scripts/exposure/exposureMap_target.py "$YEAR" "$ENERGY" "$TARGET"