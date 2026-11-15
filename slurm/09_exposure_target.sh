#!/bin/bash
#SBATCH --job-name=exposure_target
#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out

# Exposure around a target, one task per GIEM energy bin.
# Usage: sbatch slurm/09_exposure_target.sh TARGET YEAR|ALL
#        sbatch slurm/09_exposure_target.sh TARGET START END      (dates as YYYY-MM-DD)
# Run one mode per target at a time: both write to the same daily folder (see REPORT.md).
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

if [[ $# -eq 2 ]]; then
    period=(--year "$2")
else
    period=(--start "$2" --end "$3")
fi
python -m dampe_gamma exposure-map --target "$1" "${period[@]}" --energy-bin "${SLURM_ARRAY_TASK_ID}"
