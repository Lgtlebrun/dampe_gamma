#!/bin/bash
#SBATCH --job-name=split_and_weight
#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out

# Usage: sbatch slurm/01_split_and_weight.sh [EXPONENT]   (one task per cos(theta) bin)
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
N_BINS=20
python -m dampe_gamma split "${N_BINS}" "${SLURM_ARRAY_TASK_ID}" --exponent "${1:-0}"
