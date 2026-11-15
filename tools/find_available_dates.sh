#!/bin/bash
# List the days with flight data between 2016 and 2024 into data/available_dates.txt.
# Usage: tools/find_available_dates.sh [HK_DIR]
set -euo pipefail

HK_DIR="${1:-${DAMPE_HK_DIR:-/dpnc/beegfs/dampe/prod/FM/skim/6.0.0/v2}}"
OUTPUT="$(cd "$(dirname "$0")/.." && pwd)/data/available_dates.txt"

find "${HK_DIR}" -name "*_data_photon.root" \
    | sed -E 's#.*/([0-9]{4})/([0-9]{2})/([0-9]{2})_data_photon.root#\1-\2-\3#' \
    | sort \
    | awk '$0 >= "2016-01-01" && $0 < "2025-01-01"' > "${OUTPUT}"

echo "$(wc -l < "${OUTPUT}") dates written to ${OUTPUT}"
