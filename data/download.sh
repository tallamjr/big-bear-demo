#!/usr/bin/env bash
# This script downloads NYC taxi trip data files (yellow_tripdata)
# from 2009-01 to 2024-12 using wget.
# It is designed to be simple and robust – if a file already exists,
# the download for that file is skipped.

for year in {2009..2024}; do
    for month in {1..12}; do
        # Format month as two digits (e.g. 01, 02, ...)
        month_padded=$(printf "%02d" "$month")
        file="yellow_tripdata_${year}-${month_padded}.parquet"
        url="https://d37ci6vzurychx.cloudfront.net/trip-data/${file}"
        dir_prefix="yellow"

        # Check if the file already exists
        file_on_disk=$dir_prefix"/"$file
        echo $file_on_disk
        if [ -f "$file_on_disk" ]; then
            echo "$file already exists, skipping download."
        else
            echo "Downloading $file from $url..."
            wget --directory-prefix="$dir_prefix" "$url" || echo "Failed to download $file"
        fi
    done
done

