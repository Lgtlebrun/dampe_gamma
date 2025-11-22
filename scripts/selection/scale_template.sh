#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=scale_thetaBin
#SBATCH --output=log-scaleBin-%j.out

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Parse exponent if provided; default to 0 if not
if [ -z "$2" ]; then
    EXPONENT=0
else
    EXPONENT=$2
fi

# Run the scaling script for a given bin index
time python /home/users/l/lebrun/scripts/scale_thetaBin.py $1 -e $EXPONENT
