#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=effArea
#SBATCH --output=log-effArea-%j.out

# Load DmpSoftware

#source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

#source /cvmfs/dampe.cern.ch/centos7/etc/setup.sh 
#dampe_init > /dev/null
echo "Running effArea script on $(hostname) with SLURM_JOB_ID=$SLURM_JOB_ID"
time python -u /home/users/l/lebrun/scripts/effArea.py 
