#!/bin/bash
#SBATCH --job-name=energy_dispersion
#SBATCH --partition=shared-cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# Usage: sbatch slurm/06_energy_dispersion.sh
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
python -m dampe_gamma energy-dispersion
