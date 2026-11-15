#!/bin/bash
# Environment shared by every job: ROOT and the scientific Python stack from an LCG view.
# Override LCG_VIEW to use another release, and DAMPE_* variables to relocate data (see README).

LCG_VIEW="${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc14-opt}"
source "${LCG_VIEW}/setup.sh"

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export DAMPE_DATA_DIR="${DAMPE_DATA_DIR:-${REPO_DIR}/data}"

mkdir -p "${REPO_DIR}/logs"
