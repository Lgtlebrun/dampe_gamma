#!/bin/bash

# Folder and available_dates.txt path
FOLDER="/dpnc/beegfs/dampe/tryf/ForKimberly/exposureMaps_GIE/3.313GeV/daily"
AVAILABLE_DATES_FILE="$FOLDER/available_dates.txt"

# Date range
START_DATE="2016-01-01"
END_DATE="2024-12-31"

# Temporary file for expected dates
EXPECTED_DATES=$(mktemp)

# Generate expected dates
current_date="$START_DATE"
while [[ "$current_date" < "$END_DATE" || "$current_date" == "$END_DATE" ]]; do
    echo "$current_date"
    current_date=$(date -I -d "$current_date + 1 day")
done | sort > "$EXPECTED_DATES"

# Clean available_dates.txt: remove .npz, trim spaces, normalize line endings
CLEANED_DATES=$(mktemp)
cat "$AVAILABLE_DATES_FILE" \
    | sed 's/\.npz$//' \
    | tr -d '\r' \
    | awk '{$1=$1;print}' \
    | sort > "$CLEANED_DATES"

# Find missing dates
echo "Missing dates between $START_DATE and $END_DATE:"
comm -23 "$EXPECTED_DATES" "$CLEANED_DATES"

# Clean up
rm "$EXPECTED_DATES" "$CLEANED_DATES"
