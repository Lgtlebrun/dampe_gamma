#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=ExposureMap
#SBATCH --output=log-%x-%A_%a.out
#SBATCH --array=0-365
#SBATCH --exclude=cpu091

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Check for required argument
if [[ -z "$1" ]]; then
    echo "Usage: sbatch exposure_map_job.sh <YEAR>"
    exit 1
fi

YEAR=$1

# Filter only dates for that year
mapfile -t date_list < <(grep "^$YEAR-" /home/users/l/lebrun/available_dates.txt)

# Check SLURM_ARRAY_TASK_ID doesn't exceed number of dates
if [[ $SLURM_ARRAY_TASK_ID -ge ${#date_list[@]} ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds number of $YEAR dates (${#date_list[@]})"
    exit 0
fi

# Get this job’s date
my_date=${date_list[$SLURM_ARRAY_TASK_ID]}

# Set energy
energy_bin=6

echo "[$SLURM_ARRAY_TASK_ID] Running exposure map for $my_date at energy bin ${energy_bin}"
time python -u /home/users/l/lebrun/scripts/exposure/exposureMap_GIE.py "$my_date" "$energy_bin"

