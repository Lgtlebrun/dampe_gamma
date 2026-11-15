#!/bin/bash
# Print the days between START and END that have no daily exposure map in DAILY_DIR.
# Usage: tools/check_missing_dates.sh DAILY_DIR [START] [END]
set -euo pipefail

DAILY_DIR="${1:?usage: $0 DAILY_DIR [START] [END]}"
START_DATE="${2:-2016-01-01}"
END_DATE="${3:-2024-12-31}"

expected=$(mktemp)
present=$(mktemp)
trap 'rm -f "${expected}" "${present}"' EXIT

day="${START_DATE}"
while [[ ! "${day}" > "${END_DATE}" ]]; do
    echo "${day}"
    day=$(date -I -d "${day} + 1 day")
done | sort > "${expected}"

find "${DAILY_DIR}" -maxdepth 1 -name "*.npz" -printf "%f\n" | sed 's/\.npz$//' | sort > "${present}"

echo "Missing dates between ${START_DATE} and ${END_DATE}:"
comm -23 "${expected}" "${present}"
