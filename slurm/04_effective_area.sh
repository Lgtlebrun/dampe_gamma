#!/bin/bash
#SBATCH --job-name=effective_area
#SBATCH --partition=shared-cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch slurm/04_effective_area.sh [log|giem]
# The exposure maps expect the GIEM energy binning.
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
python -m dampe_gamma effective-area --energy-binning "${1:-log}"
