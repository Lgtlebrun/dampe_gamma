#!/bin/bash
#SBATCH --job-name=exposure_daily
#SBATCH --partition=shared-cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-365
#SBATCH --output=logs/%x_%A_%a.out

# All sky exposure map of one day, one task per day with data in the given year.
# Usage: sbatch slurm/08_exposure_daily.sh YEAR ENERGY_BIN
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mapfile -t dates < <(grep "^$1-" "${DAMPE_DATA_DIR}/available_dates.txt")
day="${dates[${SLURM_ARRAY_TASK_ID}]:-}"
[[ -z "${day}" ]] && { echo "No date for task ${SLURM_ARRAY_TASK_ID}"; exit 0; }

python -m dampe_gamma exposure-map --date "${day}" --energy-bin "$2"
