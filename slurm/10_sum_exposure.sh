#!/bin/bash
#SBATCH --job-name=sum_exposure
#SBATCH --partition=shared-cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

# Sum the daily exposure maps of one energy bin, all sky or around a target.
# Usage: sbatch slurm/10_sum_exposure.sh ENERGY_BIN [TARGET]
source "${SLURM_SUBMIT_DIR}/slurm/env.sh"
python -m dampe_gamma sum-exposure --energy-bin "$1" ${2:+--target "$2"}
