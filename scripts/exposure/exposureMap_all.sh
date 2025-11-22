#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=ExposureAll
#SBATCH --output=log-exposure-%A_%a.out
#SBATCH --array=0-$(($(wc -l < available_dates.txt)-1))

source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Read the line corresponding to this task ID
date=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" available_dates.txt)

# Energy parameter (adjust or make variable too)
energy=10

echo "[$SLURM_ARRAY_TASK_ID] Processing $date with $energy GeV"
time python -u /home/users/l/lebrun/scripts/exposureMap.py "$date" "$energy"
