#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=ExposureMapArray
#SBATCH --output=log-exposure-%A_%a.out
#SBATCH --array=0-29  # 30 days in April

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Generate date list for April 2021
start_date="2021-04-01"
date_list=()
for i in {0..29}; do
    date_list+=($(date -d "$start_date +$i days" +%Y-%m-%d))
done

# Get this job's date
my_date=${date_list[$SLURM_ARRAY_TASK_ID]}

# Set energy (customize as needed)
energy=10

echo "[$SLURM_ARRAY_TASK_ID] Running exposure map for $my_date at ${energy} GeV"
time python -u /home/users/l/lebrun/scripts/exposureMap.py "$my_date" "$energy"
