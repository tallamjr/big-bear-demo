from datetime import datetime
from pprint import pprint

import polars as pl

# Lazy load the Parquet file (does NOT load into memory)
df = pl.scan_parquet("../data/nyc_yellow_taxi_parquet/*")
#
# Count total rows without loading the full dataset
row_count = df.select(pl.len()).collect()

print(row_count)

pprint(df.collect_schema())
# Schema([('vendorID', String),
#         ('tpepPickupDateTime', Datetime(time_unit='ns', time_zone=None)),
#         ('tpepDropoffDateTime', Datetime(time_unit='ns', time_zone=None)),
#         ('passengerCount', Int32),
#         ('tripDistance', Float64),
#         ('puLocationId', String),
#         ('doLocationId', String),
#         ('startLon', Float64),
#         ('startLat', Float64),
#         ('endLon', Float64),
#         ('endLat', Float64),
#         ('rateCodeId', Int32),
#         ('storeAndFwdFlag', String),
#         ('paymentType', String),
#         ('fareAmount', Float64),
#         ('extra', Float64),
#         ('mtaTax', Float64),
#         ('improvementSurcharge', String),
#         ('tipAmount', Float64),
#         ('tollsAmount', Float64),
#         ('totalAmount', Float64)])

print(df.limit(10).collect(streaming=True))

# Group by pickup location and count occurrences
popular_pickups = (
    df.filter(pl.col("puLocationId").is_not_null())
    .group_by("puLocationId")
    .agg(pl.len().alias("num_trips"))
    .sort("num_trips", descending=True)
    .limit(10)
    .collect(streaming=True)
)

print(popular_pickups)

# Filter for 2016 only and compute daily total fares
daily_revenue = (
    df.filter(
        pl.col("tpepPickupDateTime").is_between(
            datetime(2016, 1, 1), datetime(2016, 12, 31)
        )
    )
    .with_columns(pl.col("tpepPickupDateTime").dt.date().alias("date"))
    .group_by("date")
    .agg(pl.sum("fareAmount").alias("total_fare"))
    .sort("date")
    .collect()
)

print(daily_revenue)

# Filter for trips longer than 50 miles, sorted by distance
longest_trips = (
    df.filter(pl.col("tripDistance") > 50)
    .select(["tripDistance", "fareAmount"])
    .sort("tripDistance", descending=True)
    .limit(10)
    .collect(streaming=True)
)

print(longest_trips)

# Define a bounding box for NYC (approximate)
min_lat, max_lat = 40.5, 40.9
min_lon, max_lon = -74.25, -73.70

result = (
    df
    # Filter out rows with null puLocationId and restrict to 2016
    # .filter(pl.col("puLocationId").is_not_null())
    .filter(
        pl.col("tpepPickupDateTime").is_between(
            datetime(2010, 1, 1), datetime(2018, 1, 1)
        )
    )
    # Filter trips by the NYC bounding box (based on startLat and startLon)
    .filter(
        (pl.col("startLat") >= min_lat)
        & (pl.col("startLat") <= max_lat)
        & (pl.col("startLon") >= min_lon)
        & (pl.col("startLon") <= max_lon)
    )
    # # Compute trip duration in minutes (convert nanoseconds to minutes)
    .with_columns(
        (
            (pl.col("tpepDropoffDateTime") - pl.col("tpepPickupDateTime")).cast(
                pl.Int64
            )
            / 1e9
            / 60
        ).alias("trip_duration")
    )
    # # Calculate average speed in mph: (tripDistance miles) / (duration in hours)
    .with_columns(
        (pl.col("tripDistance") * 60 / pl.col("trip_duration")).alias("avg_speed")
    )
    # Additional filtering on computed metrics:
    #   - Ensure positive trip duration,
    #   - Keep trips longer than 0.5 miles,
    #   - Fare amount below 150
    .filter((pl.col("trip_duration") > 0) & (pl.col("fareAmount") < 150))
    # Extract the date from the pickup datetime
    .with_columns(pl.col("tpepPickupDateTime").dt.date().alias("date"))
    # # Group by date and paymentType and compute aggregates
    .group_by(["date", "paymentType"])
    .agg(
        [
            pl.len().alias("num_trips"),
            pl.mean("tripDistance").alias("avg_trip_distance"),
            pl.sum("fareAmount").alias("total_fare"),
            pl.mean("trip_duration").alias("avg_duration"),
            pl.mean("avg_speed").alias("avg_speed"),
            pl.mean("tipAmount").alias("avg_tip"),
        ]
    )
    .sort(["date", "paymentType"])
)

qplan = result.explain(format="plain", optimized=False)
print(f"NAIVE Q-PLAN:\n {qplan}")
print(qplan)

qplan = result.explain(format="plain", optimized=True)
print(f"OPTIMIZED Q-PLAN:\n {qplan}")

result.show_graph(show=False, output_path="query-plan.png")

print(result.collect(streaming=True))

# print(
#     df.filter(pl.col("puLocationId").is_not_null())
#     .filter(
#         pl.col("tpepPickupDateTime").is_between(
#             datetime(2010, 1, 1), datetime(2018, 1, 1)
#         )
#     )
#     .select(pl.len().alias("non_null_count"))
#     .collect(streaming=True)
# )
