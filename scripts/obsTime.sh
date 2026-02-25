#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=TimeObs
#SBATCH --output=log-timeobs-%x-%A_%a.out
#SBATCH --array=0-120

# Load environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

# Pulsar l0 and b0 values
target_l=184.56
target_b=-5.78
target_name="Crab"


# Compute unique year-month combinations
mapfile -t month_list < <(cut -d '-' -f 1,2 /home/users/l/lebrun/available_dates.txt | sort -u)

if [[ $SLURM_ARRAY_TASK_ID -ge ${#month_list[@]} ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds number of unique months (${#month_list[@]})"
    exit 1
fi

# Get YYYY-MM and extract year/month
ym=${month_list[$SLURM_ARRAY_TASK_ID]}
year=$(cut -d '-' -f 1 <<< "$ym")
month=$(cut -d '-' -f 2 <<< "$ym")

echo "[$SLURM_ARRAY_TASK_ID] Running timeObs.py for l=$target_l b=$target_b on $year-$month (from $date_str)"
time python -u /home/users/l/lebrun/scripts/timeObs.py "$target_l" "$target_b" "$year" "$month" "$target_name"
