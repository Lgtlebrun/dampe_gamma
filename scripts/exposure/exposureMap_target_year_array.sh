#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=6:00:00              # Adjust time as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                     # Enough memory for single energy bin
#SBATCH --job-name=ExposureMapTarget
#SBATCH --output=log-expTarget-%A_%a.out
#SBATCH --array=0-179               # 1 year * 20 energy bins = 20 jobs indexed 0 to 19

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Parameters
TARGET="Crab"
YEAR=2016
NUM_ENERGIES=20   # energy bins 0 to 19

# Calculate year and energy bin from SLURM_ARRAY_TASK_ID
ENERGY=$(( SLURM_ARRAY_TASK_ID % NUM_ENERGIES ))        # modulo

echo "Running exposure map for year ${YEAR} at Energy bin ${ENERGY}, target ${TARGET}"

time python -u /home/users/l/lebrun/scripts/exposure/exposureMap_target.py "$YEAR" "$ENERGY" "$TARGET"
