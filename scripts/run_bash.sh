#!/bin/bash
#SBATCH --partition=private-dpnc-cpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --mem=15G
#SBATCH --job-name=AddVarGammaSel_Data
#SBATCH --output=log-%j.out

# Load DmpSoftware

#source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh

#source /cvmfs/dampe.cern.ch/centos7/etc/setup.sh 
#dampe_init > /dev/null

# Argument - name of a ntuple
time python GammaSel_Data_forParzival.py ${1}
