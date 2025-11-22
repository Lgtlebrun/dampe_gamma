#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=ExposureMapTarget
#SBATCH --output=log-expTarget-%A_%a.out
#SBATCH --array=0-19  # 20 energy bins = 20 jobs indexed 0 to 19

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Parameters
TARGET="KM3-230213A"
START_DATE="2022-08-13"
END_DATE="2023-08-13"
NUM_ENERGIES=20

# Get energy bin from SLURM_ARRAY_TASK_ID
ENERGY=$(( SLURM_ARRAY_TASK_ID % NUM_ENERGIES ))

echo "Running exposure map from ${START_DATE} to ${END_DATE} at Energy bin ${ENERGY}, target ${TARGET}"

time python -u /home/users/l/lebrun/scripts/exposure/exposureMap_targetBounded.py "$START_DATE" "$END_DATE" "$ENERGY" "$TARGET"
