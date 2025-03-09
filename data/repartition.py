import gc
from datetime import datetime

import polars as pl

# Process years 2010 to 2018 (inclusive)
for year in range(2012, 2019):
    df = pl.scan_parquet("./nyc_yellow_taxi_parquet/part-*")
    print(f"Filtering: {year}...")

    df = df.filter(
        pl.col("tpepPickupDateTime").is_between(
            datetime(year, 1, 1), datetime(year, 12, 31)
        )
    ).with_columns(
        [
            pl.col("tpepPickupDateTime").dt.year().alias("year"),
            pl.col("tpepPickupDateTime").dt.month().alias("month"),
        ]
    )

    df.collect().write_parquet(file="hive", partition_by=["year", "month"])
    del df
    gc.collect()
    print(f"Written Partition: {year}...")
