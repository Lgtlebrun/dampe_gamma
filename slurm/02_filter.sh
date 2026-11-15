#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --partition=shared-cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out

# Usage: sbatch slurm/02_filter.sh [EXPONENT]   (one task per cos(theta) bin)
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
python -m dampe_gamma filter "${SLURM_ARRAY_TASK_ID}" --exponent "${1:-0}"
