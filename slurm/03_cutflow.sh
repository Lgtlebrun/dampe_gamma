#!/bin/bash
#SBATCH --job-name=cutflow
#SBATCH --partition=shared-cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch slurm/03_cutflow.sh [EXPONENT]
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
python -m dampe_gamma cutflow --exponent "${1:-0}"
