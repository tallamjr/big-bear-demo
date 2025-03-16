import polars as pl
from polars.io.partition import PartitionByKey

df = pl.scan_parquet("./yellow/*", allow_missing_columns=True)

df = (
    df.with_columns(
        [
            pl.col("Trip_Pickup_DateTime").str.strptime(
                pl.Datetime, format="%Y-%m-%d %H:%M:%S"
            ),
            pl.col("Trip_Dropoff_DateTime").str.strptime(
                pl.Datetime, format="%Y-%m-%d %H:%M:%S"
            ),
        ]
    )
    .with_columns(
        [
            pl.col("Trip_Pickup_DateTime").dt.year().alias("year"),
            pl.col("Trip_Pickup_DateTime").dt.month().alias("month"),
        ]
    )
    .sink_parquet(
        PartitionByKey(
            "./yellow.hive/{key[0].name}={key[0].value}/{key[1].name}={key[1].value}/0000.parquet",
            by=[pl.col("year"), pl.col("month")],
            include_key=False,
        ),
        mkdir=True,
        engine="streaming",
    )
)
