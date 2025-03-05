import resource
import subprocess

import h3
import polars as pl
import polars_h3 as plh3

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


# Set up collection arguments based on GPU availability
collect_args = {}
if is_nvidia_gpu_available():
    collect_args["engine"] = "gpu"
    print("NVIDIA GPU detected, using GPU engine for collection.")
else:
    collect_args["streaming"] = True
    print("No NVIDIA GPU detected, using streaming mode for collection.")

# Lazy load the Parquet file (does NOT load into memory)
df = pl.scan_parquet("../data/nyc_yellow_taxi_parquet/*")

# Define a reference location (e.g. Times Square)
ref_lat = 40.7580
ref_lon = -73.9855
resolution = 9  # chosen resolution for H3 index
ring_size = 1  # immediate neighbours

# Compute the H3 index for the reference location using the standard h3 library
ref_h3_index = h3.latlng_to_cell(ref_lat, ref_lon, resolution)

# Obtain neighbouring H3 indices using h3's grid_ring function
neighbouring_indices = h3.grid_ring(ref_h3_index, ring_size)
neighbouring_indices_list = list(neighbouring_indices)

# Compute the H3 index for each row and filter for pickups in the neighbouring hexagons.
# Cast the resulting "h3_index" column to string to match the type in neighbouring_indices_list.
result_lazy = (
    df.with_columns(
        plh3.latlng_to_cell(
            pl.col("startLat"), pl.col("startLon"), resolution=resolution
        ).alias("h3_index")
    )
    .filter(pl.col("h3_index").cast(pl.Utf8).is_in(neighbouring_indices_list))
    .select(["startLat", "startLon", "tpepPickupDateTime", "totalAmount", "h3_index"])
)

# Execute the lazy query
result = result_lazy.collect()
print(result)
