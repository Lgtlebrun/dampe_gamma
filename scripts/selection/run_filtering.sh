#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=MC_Filtering
#SBATCH --output=log-filter-%j.out

# Load DmpSoftware

#source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

#source /cvmfs/dampe.cern.ch/centos7/etc/setup.sh 
#dampe_init > /dev/null

time python /home/users/l/lebrun/scripts/filter_MC_Jenni.py "$@"
