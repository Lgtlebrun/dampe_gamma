#!/bin/bash
find /dpnc/beegfs/dampe/prod/FM/skim/6.0.0/v2 -name "*_data_photon.root" | \
    sed -E 's#.*/([0-9]{4})/([0-9]{2})/([0-9]{2})_data_photon.root#\1-\2-\3#' \
    | sort > available_dates.txt
awk '$0 >= "2016-01-01" && $0 < "2025-01-01"' available_dates.txt > tmp && mv tmp available_dates.txt

