#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=4:00:00               # Increase if needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=TimeObs
#SBATCH --output=log-timeobs-%x-%j.out

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Target parameters
target_l=216.06
target_b=-11.1349
target_name="KM3-230213A"

# Date range
start_date="2022-08-13"
end_date="2023-08-13"

echo "[INFO] Running timeObs.py for l=$target_l b=$target_b from $start_date to $end_date"

# Call Python script with start_date and end_date
time python -u /home/users/l/lebrun/scripts/timeObsBounded.py \
    "$target_l" \
    "$target_b" \
    "$start_date" \
    "$end_date" \
    "$target_name" 
