from pprint import pprint as print

import polars as pl
from polars.datatypes import *
from polars.io.partition import PartitionByKey

schema = {
    "VendorID": Int64,
    "tpep_pickup_datetime": Datetime(time_unit="ns", time_zone=None),
    "tpep_dropoff_datetime": Datetime(time_unit="ns", time_zone=None),
    "passenger_count": Int64,
    "trip_distance": Float64,
    "RatecodeID": Int64,
    "store_and_fwd_flag": String,
    "PULocationID": Int64,
    "DOLocationID": Int64,
    "payment_type": Int64,
    "fare_amount": Float64,
    "extra": Float64,
    "mta_tax": Float64,
    "tip_amount": Float64,
    "tolls_amount": Float64,
    "improvement_surcharge": Float64,
    "total_amount": Float64,
    "congestion_surcharge": Float64,
    "airport_fee": Float64,
}


for year in range(2018, 2019):
    print(f"Filtering: {year}...")
    df = pl.scan_parquet(f"./yellow/yellow_tripdata_{year}-0[3-6]*.parquet*")
    df = df.with_columns(
        [
            pl.col("tpep_pickup_datetime").dt.year().alias("year"),
            pl.col("tpep_pickup_datetime").dt.month().alias("month"),
            # Explicitly cast columns per your schema:
            pl.col("VendorID").cast(schema["VendorID"], strict=False),
            pl.col("passenger_count").cast(schema["passenger_count"], strict=False),
            pl.col("trip_distance").cast(schema["trip_distance"], strict=False),
            pl.col("RatecodeID").cast(schema["RatecodeID"], strict=False),
            # ... add additional casts as needed for each column ...
            pl.col("congestion_surcharge")
            .fill_null(0)
            .cast(schema["congestion_surcharge"], strict=False),
            pl.col("airport_fee")
            .fill_null(0)
            .cast(schema["airport_fee"], strict=False),
        ]
    )
    df.lazy().sink_parquet(
        PartitionByKey(
            "./yellow.hive/{key[0].name}={key[0].value}/{key[1].name}={key[1].value}/0000.parquet",
            by=[pl.col("year"), pl.col("month")],
            include_key=False,
        ),
        mkdir=True,
        engine="streaming",
    )
    print(f"Hive partition written to disk for: {year}...")

df = pl.scan_parquet("yellow.hive/**/*.parquet", hive_partitioning=True)
count = df.select(pl.len().count()).collect(engine="streaming")
print(count)
