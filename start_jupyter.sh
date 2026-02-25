#!/bin/bash

# Define variables
PORT=8888                    # Port for Jupyter (adjust if needed)

#Once on the interactive node, load environment and activate the 'cern' alias
echo "Interactive node assigned. Setting up environment..."
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh
pip install healpy

#Start Jupyter Notebook with no token
echo "Starting Jupyter Notebook on $HOST..."
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' &
# Notify user that Jupyter is running and accessible
echo "Jupyter Notebook is running at: http://localhost:$PORT/"
echo "You can now access it through VS Code or a browser."

