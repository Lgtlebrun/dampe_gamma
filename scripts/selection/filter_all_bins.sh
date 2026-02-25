#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=Filter_Theta
#SBATCH --array=0-19
#SBATCH --output=log-filter-%A_%a.out

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc14-opt/setup.sh

# Number of bins
BIN_INDEX=$SLURM_ARRAY_TASK_ID

# Parse exponent if provided; default to 0 if not
if [ -z "$1" ]; then
    EXPONENT=0
else
    EXPONENT=$1
fi

echo "Filtering bin index: $BIN_INDEX with exponent: $EXPONENT"

time python -u /home/users/l/lebrun/scripts/selection/filter_thetaBin.py $BIN_INDEX -e $EXPONENT