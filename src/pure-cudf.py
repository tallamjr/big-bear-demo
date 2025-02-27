import glob
import resource
import subprocess

import cudf

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))


def is_nvidia_gpu_available():
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# Check for NVIDIA GPU availability
if is_nvidia_gpu_available():
    print("NVIDIA GPU detected, using cuDF for GPU acceleration.")
else:
    print("No NVIDIA GPU detected, cuDF will run on CPU (if supported).")

# Read all Parquet files into a cuDF DataFrame
file_pattern = "../data/nyc_yellow_taxi_parquet/*.parquet"
files = glob.glob(file_pattern)
if not files:
    raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

# Concatenate all files into a single DataFrame
dfs = [cudf.read_parquet(f) for f in files]
df = cudf.concat(dfs, ignore_index=True)
print("Data loaded successfully.")

# Count total rows
row_count = len(df)
print("Row count:", row_count)

# Print the DataFrame schema (data types)
print("DataFrame schema:")
print(df.dtypes)

# Display first 10 rows
print("First 10 rows:")
print(df.head(10))

# Group by pickup location and count occurrences
popular_pickups = (
    df[df["puLocationId"].notnull()]
    .groupby("puLocationId")
    .size()
    .reset_index(name="num_trips")
    .sort_values("num_trips", ascending=False)
    .head(10)
)
print("Top 10 popular pickup locations:")
print(popular_pickups)

# Filter for 2016 only and compute daily total fares
mask_2016 = (df["tpepPickupDateTime"] >= "2016-01-01") & (
    df["tpepPickupDateTime"] <= "2016-12-31"
)
df_2016 = df[mask_2016].copy()
df_2016["date"] = df_2016["tpepPickupDateTime"].dt.date  # Extract date
daily_revenue = (
    df_2016.groupby("date")
    .agg({"fareAmount": "sum"})
    .reset_index()
    .rename(columns={"fareAmount": "total_fare"})
    .sort_values("date")
)
print("Daily revenue for 2016:")
print(daily_revenue)

# Filter for trips longer than 50 miles, sorted by tripDistance
longest_trips = (
    df[df["tripDistance"] > 50][["tripDistance", "fareAmount"]]
    .sort_values("tripDistance", ascending=False)
    .head(10)
)
print("Longest trips (over 50 miles):")
print(longest_trips)

# Define a bounding box for NYC (approximate)
min_lat, max_lat = 40.5, 40.9
min_lon, max_lon = -74.25, -73.70

# Filter dataset for trips between 2010 and 2018 within the NYC bounding box
mask_date = (df["tpepPickupDateTime"] >= "2010-01-01") & (
    df["tpepPickupDateTime"] < "2018-01-01"
)
mask_bbox = (
    (df["startLat"] >= min_lat)
    & (df["startLat"] <= max_lat)
    & (df["startLon"] >= min_lon)
    & (df["startLon"] <= max_lon)
)
result = df[mask_date & mask_bbox].copy()

# Compute trip duration in minutes (using total_seconds)
result["trip_duration"] = (
    result["tpepDropoffDateTime"] - result["tpepPickupDateTime"]
).dt.total_seconds() / 60

# Calculate average speed in mph: (tripDistance miles) / (duration in hours)
result["avg_speed"] = result["tripDistance"] / (result["trip_duration"] / 60)

# Additional filtering: ensure positive trip duration and fare amount below 150
result = result[(result["trip_duration"] > 0) & (result["fareAmount"] < 150)].copy()

# Extract the date from the pickup datetime for aggregation
result["date"] = result["tpepPickupDateTime"].dt.date

# Group by date and paymentType and compute aggregates
agg_result = (
    result.groupby(["date", "paymentType"])
    .agg(
        {
            "tpepPickupDateTime": "count",  # Count trips
            "tripDistance": "mean",
            "fareAmount": "sum",
            "trip_duration": "mean",
            "avg_speed": "mean",
            "tipAmount": "mean",
        }
    )
    .reset_index()
    .rename(
        columns={
            "tpepPickupDateTime": "num_trips",
            "tripDistance": "avg_trip_distance",
            "fareAmount": "total_fare",
            "trip_duration": "avg_duration",
            "avg_speed": "avg_speed",
            "tipAmount": "avg_tip",
        }
    )
    .sort_values(by=["date", "paymentType"])
)
print("Aggregated metrics by date and payment type:")
print(agg_result)

# Note: cuDF does not provide query plan explanation or visualisation like Polars.
print("cuDF does not offer query plan explanations as Polars does.")
