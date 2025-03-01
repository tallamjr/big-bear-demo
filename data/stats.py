import glob

import pyarrow.parquet as pq

# Path pattern for your Parquet files (adjust as needed)
file_pattern = "./nyc_yellow_taxi_parquet/*"

# Get all matching Parquet files
parquet_files = glob.glob(file_pattern)

# Compute the total uncompressed size
total_uncompressed_size = sum(
    sum(
        pq.ParquetFile(file).metadata.row_group(i).total_byte_size
        for i in range(pq.ParquetFile(file).metadata.num_row_groups)
    )
    for file in parquet_files
)

print(
    f"Total uncompressed size: {total_uncompressed_size / (1024 * 1024 * 1024):.2f} GiB"
)
