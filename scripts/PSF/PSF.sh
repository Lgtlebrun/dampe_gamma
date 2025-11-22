#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=PSF
#SBATCH --output=log-PSF-%j.out

# Load DmpSoftware

#source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc14-opt/setup.sh

#source /cvmfs/dampe.cern.ch/centos7/etc/setup.sh 
#dampe_init > /dev/null

time python -u /home/users/l/lebrun/scripts/PSF/PSF_twoStages.py 
