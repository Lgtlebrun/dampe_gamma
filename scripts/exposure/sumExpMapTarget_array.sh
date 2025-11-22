#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=6:00:00              # Adjust time as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                     # Enough memory for single energy bin
#SBATCH --job-name=SumExpMapTarget
#SBATCH --output=log-SumTarget-%A_%a.out
#SBATCH --array=0-19                # 20 energy bins 

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Parameters
TARGET="Geminga"
NUM_ENERGIES=20   # energy bins 0 to 19

# Calculate year and energy bin from SLURM_ARRAY_TASK_ID
ENERGY=$(( SLURM_ARRAY_TASK_ID % NUM_ENERGIES ))        # modulo

echo "Running all-time exposure map at Energy bin ${ENERGY}, target ${TARGET}"

time python -u /home/users/l/lebrun/scripts/exposure/sumExpMapTarget.py "$ENERGY" "$TARGET"
