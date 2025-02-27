# 🐻‍❄️ A Polar Bear Showcase

Not too long ago Data Scientist was dubbed the sexiest job title of the decade
and it seems like from the early 2010s there has been a nice DataFrame library
hit the scene every other year.

While "Data Science" and the now ubiqtuous "DataFrame" (think table of data,
maybe the dreaded excel spreadsheet comes to mind) might be in their infancy,
relational database research, i.e. that essentially deals with tabular data is
very mature.

In 2012 `pandas` shot to fame with a easy Pythonic way to manipulate and handle
tabular data. Database administrators (DBAs) and SQL query writing wizards could no longer gate keep their
secrets of being able to manage databases of data and tables and the power to
manipulate "large" datasets was now available to anyone who a bit of Python. I
am intentionally ignoring all statistics people who use R because well .. I am

The trouble was that these new kids on the block with their shiny data science
tools largely ignored the glorious database research that came before. And can
you blame them, they just wanted to get on with visualising how many pople in
2nd class survived the Titantic incident.

Over the years the bifurcation continued with DataFrame libraries working with
data that could be processed in memory but what happens when you have some
serious big data or worse yet and super complex join of two tables.

Well, what you would do is naively bring everything into memory and hope that your
large outer-product cross join can also fit in memory. But _it need-not be this
way John!_

Projects like Apache Spark were one of the few libraries that took database
research seriously and put a lot of effort into understanding how best to create
a Query Plan and Query optimiser.

<!-- TODO: A more detailed history lesson will follow but let's get on with the show. -->

To showcase **Polars' lazy execution, query planning, and optimisation** we will
look at the **50GB NYC Taxi dataset (~1.5 billion rows)** and present how to
efficiently query a larger than RAM dataset on a **32GB RAM laptop**.

For this we will:

- **Leverage lazy execution** (`pl.scan_parquet()`) instead of eager loading
  (`pl.read_parquet()`)
- **Push down predicates** to filter data **before** loading it into memory
- **Avoid unnecessary computations** through **query optimisations** like column
  projection
- **Use fast aggregations** to summarise large datasets

The code for this demo is all in `src/main.py` and as long as the
`requirements.txt` are installed for `CPU` then all should be fine. We will
compare results with running on GPU which would require one to run
`./install-gpu-deps.sh`

### **1. Count Rows Efficiently (Full Dataset Scan)**

Let's first look at counting rows, all 1.5 _billion_ of them! 👀

- Instead of materialising the dataset in memory, **Polars will scan metadata**
  to count rows efficiently.

```python
import polars as pl

# Lazy load the Parquet file (does NOT load into memory)
df = pl.scan_parquet("nyc_taxi_faker.parquet")

# Count total rows without loading the full dataset
row_count = df.select(pl.count()).collect()

print(row_count)
```

**Why is this fast?**

- **Query Optimisation:** Polars **pushes down the count aggregation** to avoid
  reading the entire dataset.
- **Metadata Scan:** Parquet stores **row counts in metadata**, allowing
  **Polars to retrieve them without a full scan**.

### **2. Find the Most Popular Pickup Locations**

- Using **groupby & aggregation** on the entire dataset but **only returning the top 10 results**.

```python
df = pl.scan_parquet("nyc_taxi_faker.parquet")

# Group by pickup location and count occurrences
popular_pickups = (
    df.group_by("PULocationID")
    .agg(pl.count().alias("num_trips"))
    .sort("num_trips", descending=True)
    .limit(10)  # Only return the top 10
    .collect()
)

print(popular_pickups)
```

**Why is this efficient?**

- **Predicate Pushdown:** Only computes group counts, **not loading unused columns**.
- **Optimised Aggregation:** Polars uses **multi-threading** for fast counting.

### **3. Compute Daily Total Revenue (But Only for 2016)**

- **Time-based filtering** ensures **only necessary rows are read**.

```python
from datetime import datetime

df = pl.scan_parquet("nyc_taxi_faker.parquet")

# Filter for 2023 only and compute daily total fares
daily_revenue = (
    df.filter(pl.col("tpep_pickup_datetime").is_between(datetime(2023, 1, 1), datetime(2023, 12, 31)))
    .group_by(pl.col("tpep_pickup_datetime").dt.date())  # Aggregate per day
    .agg(pl.sum("fare_amount").alias("total_fare"))
    .sort("tpep_pickup_datetime")  # Ensure results are ordered
    .collect()
)

print(daily_revenue)
```

**Why does this run well on 32GB RAM?**

- **Lazy Filtering:** The **date filter is pushed down**, meaning **only 2016
  trips are read**.
- **Column Pruning:** Polars **only loads `tpep_pickup_datetime` and
  `fare_amount`**, not the whole dataset.

### **4. Find the Longest Taxi Trips (Optimised Distance Query)**

- **Filtering + Sorting on large dataset**.

```python
df = pl.scan_parquet("nyc_taxi_faker.parquet")

# Filter for trips longer than 50 miles, sorted by distance
longest_trips = (
    df.filter(pl.col("trip_distance") > 50)
    .select(["trip_distance", "fare_amount", "PULocationID", "DOLocationID"])
    .sort("trip_distance", descending=True)
    .limit(10)
    .collect()
)

print(longest_trips)
```

**Why does this perform well?**

- **Predicate Pushdown:** Only trips with `trip_distance > 50` are processed.
- **Column Projection:** Only four columns are read instead of all.

### **Key Takeaways**

| Query Type                           | Polars Optimisation Used         |
| ------------------------------------ | -------------------------------- |
| **Count total rows**                 | Metadata scan, no full load      |
| **Top pickup locations**             | GroupBy pushdown, column pruning |
| **Daily revenue for 2023**           | Date filtering pushdown          |
| **Find longest trips**               | Filter + sort optimisation       |
| **Extract coordinates from GeoJSON** | Lazy evaluation of JSON          |

With these queries, you can **scan 1.5 billion rows efficiently** and **return
small, meaningful results** while keeping memory usage low!

So we talked a lot about query optimisation and planning but perhaps this is
still not appreciated.

Query optimisation and planning often involves doing "predicate pushdown" and
"projection pushdown". If some of you are like me you might be asking what on
earth is the difference -- well an awesome explanation can be found here.

So, in a nutshell, re-arrange the query so we touch as little data as necessary!

### **5. Query Plans in Detail**

`polars` has one of the best query planners and optimisers in the business and
we can get more of a feel for what is going on if we inspect the query with
`df.explain()`. Here we show a much more complex query that does many different
filtering operations and aggregations.

It leverages Polars’ lazy evaluation, predicate
pushdown, column projection, and streaming mode. This query assumes you’re
working with a 1.5 billion-row NYC Yellow Taxi dataset (using the provided
schema) and aims to demonstrate several advanced operations in one go.

#### What This Query Demonstrates

- **Predicate Pushdown & Column Projection:** The filters on
  **tpepPickupDateTime**, **startLat**, **startLon**, and **puLocationId** are
  pushed down to the data source. This means Polars only loads the row groups that
  could possibly match these conditions—vital when dealing with billions of rows.

- **Lazy Evaluation & Streaming:** By using `pl.scan_parquet(...,
streaming=True)` and deferring execution until the final `.collect()`, the query
  planner optimises the entire operation, ensuring minimal memory usage even on a
  laptop with 32 GB of RAM.

- **Complex Aggregation & Derived Metrics:** The query computes additional
  columns (trip duration, average speed) and applies further filtering on these
  derived values. Then, grouping by date and payment type demonstrates the power
  of Polars’ parallel aggregations over massive datasets.

```python
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

```

I encourage the reader to checkout the large query in detail, but the main thing
is to inspect to two different query plan graphs, the first being an
**un-optimised query plan** and then followed by the **optimised query plan**
version -- do you spot a difference?

**1. Un-optimised**

```bash
NAIVE Q-PLAN:
 SORT BY [col("date"), col("paymentType")]
  AGGREGATE
        [len().alias("num_trips"), col("tripDistance").mean().alias("avg_trip_distance"), col("fareAmount").sum().alias("total_fare"), col("trip_duration").mean().alias("avg_duration"), col("avg_speed").mean().alias("avg_speed"), col("tipAmount").mean().alias("avg_tip")] BY [col("date"), col("paymentType")] FROM
     WITH_COLUMNS:
     [col("tpepPickupDateTime").dt.date().alias("date")]
      FILTER [(col("trip_duration")) > (0.0)] FROM
        FILTER [(col("fareAmount")) < (150.0)] FROM
           WITH_COLUMNS:
           [[([(col("tripDistance")) * (60.0)]) / (col("trip_duration"))].alias("avg_speed")]
             WITH_COLUMNS:
             [[([([(col("tpepDropoffDateTime")) - (col("tpepPickupDateTime"))].strict_cast(Int64).cast(Unknown(Float))) / (dyn float: 1.0000e9)]) / (60.0)].alias("trip_duration")]
              FILTER [(col("startLat")) >= (40.5)] FROM
                FILTER [(col("startLat")) <= (40.9)] FROM
                  FILTER [(col("startLon")) >= (-74.25)] FROM
                    FILTER [(col("startLon")) <= (-73.7)] FROM
                      FILTER col("tpepPickupDateTime").strict_cast(Datetime(Microseconds, None)).is_between([2010-01-01 00:00:00, 2018-01-01 00:00:00]) FROM
                        Parquet SCAN [../data/nyc_yellow_taxi_parquet/part-00000-tid-8898858832658823408-a1de80bd-eed3-4d11-b9d4-fa74bfbd47bc-426339-1.c000.snappy.parquet, ... 2712 other sources]
                        PROJECT */21 COLUMNS
SORT BY [col("date"), col("paymentType")]
  AGGREGATE
        [len().alias("num_trips"), col("tripDistance").mean().alias("avg_trip_distance"), col("fareAmount").sum().alias("total_fare"), col("trip_duration").mean().alias("avg_duration"), col("avg_speed").mean().alias("avg_speed"), col("tipAmount").mean().alias("avg_tip")] BY [col("date"), col("paymentType")] FROM
     WITH_COLUMNS:
     [col("tpepPickupDateTime").dt.date().alias("date")]
      FILTER [(col("trip_duration")) > (0.0)] FROM
        FILTER [(col("fareAmount")) < (150.0)] FROM
           WITH_COLUMNS:
           [[([(col("tripDistance")) * (60.0)]) / (col("trip_duration"))].alias("avg_speed")]
             WITH_COLUMNS:
             [[([([(col("tpepDropoffDateTime")) - (col("tpepPickupDateTime"))].strict_cast(Int64).cast(Unknown(Float))) / (dyn float: 1.0000e9)]) / (60.0)].alias("trip_duration")]
              FILTER [(col("startLat")) >= (40.5)] FROM
                FILTER [(col("startLat")) <= (40.9)] FROM
                  FILTER [(col("startLon")) >= (-74.25)] FROM
                    FILTER [(col("startLon")) <= (-73.7)] FROM
                      FILTER col("tpepPickupDateTime").strict_cast(Datetime(Microseconds, None)).is_between([2010-01-01 00:00:00, 2018-01-01 00:00:00]) FROM
                        Parquet SCAN [../data/nyc_yellow_taxi_parquet/part-00000-tid-8898858832658823408-a1de80bd-eed3-4d11-b9d4-fa74bfbd47bc-426339-1.c000.snappy.parquet, ... 2712 other sources]
                        PROJECT */21 COLUMNS

```

**2. Optimised**

```bash
OPTIMIZED Q-PLAN:
 SORT BY [col("date"), col("paymentType")]
  AGGREGATE
        [len().alias("num_trips"), col("tripDistance").mean().alias("avg_trip_distance"), col("fareAmount").sum().alias("total_fare"), col("trip_duration").mean().alias("avg_duration"), col("avg_speed").mean().alias("avg_speed"), col("tipAmount").mean().alias("avg_tip")] BY [col("date"), col("paymentType")] FROM
     WITH_COLUMNS:
     [[([(col("tripDistance")) * (60.0)]) / (col("trip_duration"))].alias("avg_speed"), col("tpepPickupDateTime").dt.date().alias("date")]
      FILTER [(col("trip_duration")) > (0.0)] FROM
         WITH_COLUMNS:
         [[([([(col("tpepDropoffDateTime")) - (col("tpepPickupDateTime"))].strict_cast(Int64).cast(Unknown(Float))) / (dyn float: 1.0000e9)]) / (60.0)].alias("trip_duration")]
          Parquet SCAN [../data/nyc_yellow_taxi_parquet/part-00000-tid-8898858832658823408-a1de80bd-eed3-4d11-b9d4-fa74bfbd47bc-426339-1.c000.snappy.parquet, ... 2712 other sources]
          PROJECT 8/21 COLUMNS
          SELECTION: [([([([(col("fareAmount")) < (150.0)]) & ([([(col("startLat")) <= (40.9)]) & ([(col("startLat")) >= (40.5)])])]) & ([([(col("startLon")) <= (-73.7)]) & ([(col("startLon")) >= (-74.25)])])]) & (col("tpepPickupDateTime").strict_cast(Datetime(Microseconds, None)).is_between([2010-01-01 00:00:00, 2018-01-01 00:00:00]))]


```

The first to mention is how to even read this -- one starts from the bottom and
works their way up the tree. With that in mind, what's happening? Well, first
difference to notice is the reduction in lines. _Ok great, what does that even
mean?_

Each line represents a computation on an intermediate set of data. This is
particularly apparent with these lines from the unoptimised version:

```bash
...
  FILTER [(col("startLat")) >= (40.5)] FROM
    FILTER [(col("startLat")) <= (40.9)] FROM
      FILTER [(col("startLon")) >= (-74.25)] FROM
        FILTER [(col("startLon")) <= (-73.7)] FROM
          FILTER col("tpepPickupDateTime").strict_cast(Datetime(Microseconds, None)).is_between([2010-01-01 00:00:00, 2018-01-01 00:00:00]) FROM
..
```

This means we are applying a filter, creating a new intermediate set of data and
then moving on to the next bit and so on, very inefficient.

Another big one is this difference:

```diff
-          PROJECT */21 COLUMNS
+          PROJECT 8/21 COLUMNS

```

When you hear `PROJECT` then we are dealing with _columns_, so one is saying
give me **all** columns with the `*` and the other is saying "oh, don't mind
me, I only need 8 out of all 21 columns you have Mr. parquet file". In concrete
terms it means that only 8 columns are read from disk which can be huge,
especially in this case!

When we run the final query in `src/main.py` on a 32 GB M2 Mac it takes about

```console
real    2m8.376s
user    4m11.232s
sys     0m56.340s

```

So, long story short, a query optimiser is key in efficient processing of big
data!

## `cuDF`

Yeah Ok, that's cool and all but doesn't that all go about the window when I
have a beefy GPU sitting here.

Well, the go-to DataFrame library at the moment for running queries is `cuDF`
developed by the RAPIDS team at NVIDIA[^1].

The main issue with `cuDF` even though it is indeed blazingly fast and allows
for massive parallelism, it **does not** have an inbuilt optimiser or query
planner. So this means to run queries require the **full** dataset to be brought
into VRAM.

Now with what we saw above, that just seems silly.

[^1]: We are only really considering NVIDIA chips, for reasons. Good reasons.

`cuDF` is primarily designed for eager execution—much like `pandas`, but on the
GPU, so it doesn't include a built‑in lazy query planning engine like Polars
does. Instead, `cuDF`’s optimisations come from highly optimised GPU kernels and
vectorised operations that execute immediately.

That said, when reading data (for example, from `parquet` files), some predicate
pushdown may be performed by the underlying file reader (often via Apache
Arrow), which can reduce the amount of data loaded into memory. But beyond that,
`cuDF` processes queries eagerly without a separate query planning phase.

_But I want to use my super cool gold box (NVIDIA DGX)!_

For workflows that benefit from lazy evaluation and query planning, you might
consider combining Polars’ lazy API with a conversion step to cuDF (or using
Dask‑cuDF for distributed GPU processing). This way we can filter and reduce
data first using Polars, and then hand the smaller dataset off to `cuDF` for
GPU‑accelerated computations.

With recent updates to `polars` that is now possible with
`lf.collect(engine="gpu")`. This really gives up best of both worlds where we
can leverage `polars` query planning and optimisations and also the
computational brute force of a GPU. By clever construction of our query we
minimise data movement and also the amount that has to sit on the GPU at all.

<!-- TODO: Rewrite below -->

It's worth noting that `collect()` offers several parameters to control the execution:

- You can enable streaming mode with `streaming=True` to process the query in
  batches for larger-than-memory data.
- You can select the engine (CPU or GPU) with the `engine` parameter.
- You can run the query in the background with `background=True` to get a handle
  to the query.

For example:

```python
result = lf.collect(streaming=True, engine="cpu")
```

Remember that the GPU engine and streaming mode are considered unstable features
[Api > Python > Stable > Reference > Lazyframe > Api >
polars.LazyFrame.collect](<[https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect.html](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect.html)>).

Based on the knowledge sources provided, I can explain predicate pushdown in
Polars and its relation to GPU features. However, it's important to note that
the information about GPU usage with predicate pushdown is limited in the given
sources.

Predicate pushdown is an optimization technique used in Polars to improve query
performance. It involves moving filter conditions (predicates) as close as
possible to the data source in the query execution plan. This optimization helps
reduce the amount of data that needs to be processed in subsequent steps of the
query.

Here's an example of how predicate pushdown works in Polars:

```python
q1 = (
    pl.scan_csv("docs/assets/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
)

q1.explain()
```

The optimized query plan output shows:

```
 WITH_COLUMNS:
 [col("name").str.uppercase()]

    CSV SCAN data/reddit.csv
    PROJECT */6 COLUMNS
    SELECTION: [(col("comment_karma")) > (0)]
```

In this case, the query optimizer has pushed down the filter condition
`(col("comment_karma")) > (0)` to be applied while the CSV is being read, rather
than reading the entire file into memory and then applying the filter
[User-guide > Lazy > Query
plan](<[https://docs.pola.rs/user-guide/lazy/query-plan/#query-plan](https://docs.pola.rs/user-guide/lazy/query-plan/#query-plan)>).

Regarding GPU usage, Polars does offer a GPU engine option. However, the
knowledge sources don't provide specific information about how predicate
pushdown interacts with GPU processing. When using the GPU engine, you can
specify it in the `collect()` method:

```python
df = lf.collect(engine="gpu")
```

It's worth noting that GPU mode is considered unstable, and not all queries will
run successfully on the GPU. However, they should fall back transparently to the
default engine if execution is not supported on the GPU [Api > Python > Stable >
Reference > Lazyframe > Api >
polars.LazyFrame.collect](<[https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect.html](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect.html)>).

To get more detailed information about how a query is optimized and whether it
can use GPU processing, you can use the `explain()` method on a LazyFrame.

Polars employs a sophisticated query planning and optimization system as part of
its lazy evaluation strategy. This system is designed to improve query
performance and efficiency. Here's an overview of how Polars handles query
planning and optimization:

1.  Lazy Evaluation: Polars uses lazy evaluation, which means that when you
    define a query, it doesn't immediately execute. Instead, it builds a query
    plan. This plan is only executed when you call methods like `collect()` or
    `fetch()` [User-guide > Concepts > Lazy
    API](<[https://docs.pola.rs/user-guide/concepts/lazy-api/#lazy-api](https://docs.pola.rs/user-guide/concepts/lazy-api/#lazy-api)>).

1.  Query Optimizer: When you execute a lazy query, Polars runs it through a
    query optimizer. This optimizer analyzes the entire query plan and applies
    various optimizations to make the query more efficient [User-guide > Lazy >
    Optimizations](<[https://docs.pola.rs/user-guide/lazy/optimizations/#optimizations](https://docs.pola.rs/user-guide/lazy/optimizations/#optimizations)>).

1.  Types of Optimizations: Polars applies several types of optimizations,
    including:

        * Predicate pushdown: Applies filters as early as possible, often at the
          scan level.
        * Projection pushdown: Selects only the necessary columns at the scan
          level.
        * Slice pushdown: Only loads the required slice from the scan level.
        * Common subplan elimination: Caches subtrees/file scans used by
          multiple subtrees in the query plan.
        * Simplify expressions: Performs various optimizations like constant
          folding and replacing expensive operations with faster alternatives.
        * Join ordering: Estimates which branches of joins should be executed
          first to reduce memory pressure.
        * Type coercion: Coerces types for successful operations and minimal
          memory usage.
        * Cardinality estimation: Estimates cardinality to determine the optimal
          group-by strategy.

[User-guide > Lazy >
Optimizations](https://docs.pola.rs/use[r-guide/lazy/optimizations/#optimizations)

](https://docs.pola.rs/user-guide/lazy/optimizations/#optimizations)1.
[Optimization
Frequ](https://docs.pola.rs/user-guide/lazy/optimizations/#optimizations)ency:
Some optimizations run once, while others run multiple times until a fixed point
is reached. For example, predicate pushdown runs once, while simplify
expressions runs until a fixed point is reached.

1.  Visualizing and Explaining Query Plans: Polars provides tools to visualize
    and explain query plans:

            * `show_graph()`: Creates a Graphviz visualization of the query plan.
            * `explain()`: Prints a text representation of the query plan.

These tools can show both the non-optimized and optimized query plans
[User-guide > Lazy > Query
plan](https://docs.pola.rs/user-guid[e/lazy/query-plan/#query-plan).

](https://docs.pola.rs/user-guide/lazy/query-plan/#query-plan)1.
[Example:](https://docs.pola.rs/user-guide/lazy/query-plan/#query-plan) [Here's
an exam](https://docs.pola.rs/user-guide/lazy/query-plan/#query-plan)ple of how
you can see the optimized query plan:

```python
q1 = (
    pl.scan_csv("docs/assets/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
)

print(q1.explain())
```

This might output:

```
 WITH_COLUMNS:
 [col("name").str.uppercase()]

    CSV SCAN data/reddit.csv
    PROJECT */6 COLUMNS
    SELECTION: [(col("comment_karma")) > (0)]
```

This optimized plan shows that the filter has been pushed down to the CSV scan level [User-guide > Lazy > Query plan](https://docs.pola.rs/user-guid[e/lazy/query-plan/#query-plan).

By leveraging these query ](https://docs.pola.rs/user-guide/lazy/query-plan/#query-plan)planning and optimization techniques, Polars can significantly improve query performance, especially for complex queries on large datasets.

```python

shape: (1, 1)
┌────────────┐
│ len        │
│ ---        │
│ u32        │
╞════════════╡
│ 1571671152 │
└────────────┘
Schema([('vendorID', String),
        ('tpepPickupDateTime', Datetime(time_unit='ns', time_zone=None)),
        ('tpepDropoffDateTime', Datetime(time_unit='ns', time_zone=None)),
        ('passengerCount', Int32),
        ('tripDistance', Float64),
        ('puLocationId', String),
        ('doLocationId', String),
        ('startLon', Float64),
        ('startLat', Float64),
        ('endLon', Float64),
        ('endLat', Float64),
        ('rateCodeId', Int32),
        ('storeAndFwdFlag', String),
        ('paymentType', String),
        ('fareAmount', Float64),
        ('extra', Float64),
        ('mtaTax', Float64),
        ('improvementSurcharge', String),
        ('tipAmount', Float64),
        ('tollsAmount', Float64),
        ('totalAmount', Float64)])
shape: (10, 21)
┌──────────┬─────────────────────┬─────────────────────┬────────────────┬───┬──────────────────────┬───────────┬─────────────┬─────────────┐
│ vendorID ┆ tpepPickupDateTime  ┆ tpepDropoffDateTime ┆ passengerCount ┆ … ┆ improvementSurcharge ┆ tipAmount ┆ tollsAmount ┆ totalAmount │
│ ---      ┆ ---                 ┆ ---                 ┆ ---            ┆   ┆ ---                  ┆ ---       ┆ ---         ┆ ---         │
│ str      ┆ datetime[ns]        ┆ datetime[ns]        ┆ i32            ┆   ┆ str                  ┆ f64       ┆ f64         ┆ f64         │
╞══════════╪═════════════════════╪═════════════════════╪════════════════╪═══╪══════════════════════╪═══════════╪═════════════╪═════════════╡
│ 2        ┆ 2002-12-31 23:58:47 ┆ 2003-01-01 00:09:49 ┆ 1              ┆ … ┆ 0.3                  ┆ 0.0       ┆ 0.0         ┆ 9.3         │
│ 2        ┆ 2002-12-31 23:04:50 ┆ 2003-01-01 06:42:58 ┆ 2              ┆ … ┆ 0.3                  ┆ 0.0       ┆ 0.0         ┆ 13.8        │
│ CMT      ┆ 2009-04-30 23:50:17 ┆ 2009-05-01 01:13:28 ┆ 1              ┆ … ┆ null                 ┆ 20.0      ┆ 0.0         ┆ 160.0       │
│ CMT      ┆ 2009-04-30 23:56:20 ┆ 2009-05-01 00:16:47 ┆ 1              ┆ … ┆ null                 ┆ 0.0       ┆ 0.0         ┆ 14.5        │
│ VTS      ┆ 2009-04-30 23:57:00 ┆ 2009-05-01 00:16:00 ┆ 1              ┆ … ┆ null                 ┆ 0.0       ┆ 0.0         ┆ 23.8        │
│ VTS      ┆ 2009-04-30 23:59:00 ┆ 2009-05-01 00:12:00 ┆ 2              ┆ … ┆ null                 ┆ 0.0       ┆ 0.0         ┆ 13.4        │
│ VTS      ┆ 2009-04-30 23:42:00 ┆ 2009-05-01 00:20:00 ┆ 1              ┆ … ┆ null                 ┆ 0.0       ┆ 0.0         ┆ 25.4        │
│ CMT      ┆ 2009-04-30 23:43:07 ┆ 2009-05-01 00:02:41 ┆ 1              ┆ … ┆ null                 ┆ 0.0       ┆ 4.15        ┆ 32.25       │
│ VTS      ┆ 2009-04-30 23:56:00 ┆ 2009-05-01 00:13:00 ┆ 1              ┆ … ┆ null                 ┆ 0.0       ┆ 0.0         ┆ 16.2        │
│ CMT      ┆ 2009-04-30 23:53:00 ┆ 2009-05-01 00:27:11 ┆ 1              ┆ … ┆ null                 ┆ 0.0       ┆ 0.0         ┆ 22.9        │
└──────────┴─────────────────────┴─────────────────────┴────────────────┴───┴──────────────────────┴───────────┴─────────────┴─────────────┘

```
