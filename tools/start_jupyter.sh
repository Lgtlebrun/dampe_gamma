#!/bin/bash
# Start a Jupyter server on a compute node, reachable through an SSH tunnel or VS Code.
# Usage: tools/start_jupyter.sh [VENV_DIR]
# Without argument the LCG view is used; with a virtual environment (for example one
# holding the PINT fork used by the light curve notebooks) that environment is activated.
set -euo pipefail

PORT="${PORT:-8888}"
if [[ $# -ge 1 ]]; then
    source "$1/bin/activate"
else
    source "$(dirname "$0")/../slurm/env.sh"
fi

echo "Starting Jupyter on $(hostname), port ${PORT}"
jupyter notebook --no-browser --port="${PORT}" --ip=0.0.0.0 \
    --NotebookApp.token='' --NotebookApp.password=''
