import polars as pl

for year in range(2009, 2025):
    print(f"Checking {year}: \t\t")
    df = pl.scan_parquet(f"yellow/yellow_tripdata_{year}-*.parquet")
    print(df.collect_schema().len())
    print(df.collect_schema())
