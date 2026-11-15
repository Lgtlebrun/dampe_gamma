#!/bin/bash
#SBATCH --job-name=observing_time
#SBATCH --partition=shared-cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-119
#SBATCH --output=logs/%x_%A_%a.out

# Monthly observing time histograms of one sky position, one task per month with data.
# Usage: sbatch slurm/07_observing_time.sh L_DEG B_DEG NAME     (e.g. 184.56 -5.78 Crab)
# The months are then summed in notebooks/exposure/sum_observing_time.ipynb.
# For a single date range instead:
#   python -m dampe_gamma observing-time L B --name NAME --start 2022-08-13 --end 2023-08-13
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"

mapfile -t months < <(cut -c1-7 "${DAMPE_DATA_DIR}/available_dates.txt" | sort -u)
month="${months[${SLURM_ARRAY_TASK_ID}]:-}"
[[ -z "${month}" ]] && { echo "No month for task ${SLURM_ARRAY_TASK_ID}"; exit 0; }

python -m dampe_gamma observing-time "$1" "$2" --name "$3" --month "${month}"
