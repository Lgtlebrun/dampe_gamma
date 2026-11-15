#!/bin/bash
#SBATCH --job-name=counts_map
#SBATCH --partition=shared-cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# All sky counts map of the flight photons.
# Usage: sbatch slurm/11_counts_map.sh [ENERGY_BIN]
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
python -m dampe_gamma counts-map --energy-bin "${1:-6}"
