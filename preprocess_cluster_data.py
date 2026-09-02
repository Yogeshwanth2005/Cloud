"""
Preprocess the Google Cluster Data 2011-2 trace (day-14 sample) into clean,
analysis-ready tables for the Active Causal Orchestration project.

Input : google_cluster_2011/{machine_events,machine_attributes,job_events,
                              task_events,task_usage}/*.csv.gz
Output: google_cluster_2011/processed/*.parquet

Column names follow the widely-reproduced schema from the trace's
"v2.1 format + schema" document (Reiss & Wilkes, 2011). Trace timestamps are
microseconds since trace start; trace start = 19:00 EDT, Sun May 1 2011
(trace-relative timestamp 600s per the official docs), so a synthetic
wall-clock column is derived for diurnal alignment with solar data.
"""

import glob
import os
import gzip
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = os.path.join("google_cluster_2011")
OUT_DIR = os.path.join(RAW_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Trace-relative timestamp 600s corresponds to 19:00 EDT on 2011-05-01.
# => trace timestamp 0 corresponds to 2011-05-01 18:50:00 EDT (naive, no tz math needed
#    since we only need relative/diurnal alignment, not absolute EDT correctness).
TRACE_EPOCH = pd.Timestamp("2011-05-01 18:50:00")

# task_usage timestamps are microsecond-precision and fall on whatever
# instant a task happened to start or end -- not on a clean 5-minute grid.
# Bin start_time onto 5-minute boundaries before aggregating, otherwise each
# partial-interval sample (a task starting/ending mid-window) becomes its own
# near-empty row instead of being summed into the interval it belongs to.
BIN_MICROS = 5 * 60 * 1_000_000


def load_shards(table_name, columns, usecols=None, dtype=None):
    """Read every csv.gz shard for a table, concatenate, assign column names."""
    pattern = os.path.join(RAW_DIR, table_name, "*.csv.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shards found for {table_name} at {pattern}")
    frames = []
    for f in files:
        df = pd.read_csv(
            f,
            header=None,
            names=columns,
            usecols=usecols,
            dtype=dtype,
            na_values=["", "NA"],
            compression="gzip",
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_walltime(df, ts_col, out_col="wall_time"):
    """Convert microsecond trace timestamps to a synthetic wall-clock column."""
    df[out_col] = TRACE_EPOCH + pd.to_timedelta(df[ts_col], unit="us")
    df["hour_of_day"] = df[out_col].dt.hour + df[out_col].dt.minute / 60.0
    return df


# ---------------------------------------------------------------------------
# 1. machine_events -> per-machine capacity timeline
# ---------------------------------------------------------------------------
print("Processing machine_events ...")
me_cols = ["timestamp", "machine_id", "event_type", "platform_id", "cpus", "memory"]
me = load_shards("machine_events", me_cols)

EVENT_TYPE = {0: "ADD", 1: "REMOVE", 2: "UPDATE"}
me["event_name"] = me["event_type"].map(EVENT_TYPE)
me = add_walltime(me, "timestamp")

# Drop exact duplicate rows, sort chronologically for downstream state reconstruction
me = me.drop_duplicates().sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

me.to_parquet(os.path.join(OUT_DIR, "machine_events.parquet"), index=False)
print(f"  -> machine_events: {len(me):,} rows, "
      f"{me['machine_id'].nunique():,} distinct machines")

# Derive a simple "known machine capacity" snapshot from ADD/UPDATE events
capacity = (
    me[me["event_name"].isin(["ADD", "UPDATE"])]
    .sort_values("timestamp")
    .groupby("machine_id")
    .last()[["cpus", "memory"]]
    .reset_index()
)
capacity.to_parquet(os.path.join(OUT_DIR, "machine_capacity_snapshot.parquet"), index=False)
print(f"  -> machine_capacity_snapshot: {len(capacity):,} machines")

# ---------------------------------------------------------------------------
# 2. machine_attributes -> latest non-deleted attribute per machine
# ---------------------------------------------------------------------------
print("Processing machine_attributes ...")
ma_cols = ["timestamp", "machine_id", "attribute_name", "attribute_value", "attribute_deleted"]
ma = load_shards("machine_attributes", ma_cols)
ma["attribute_deleted"] = ma["attribute_deleted"].astype(bool)
ma = ma.drop_duplicates()
ma.to_parquet(os.path.join(OUT_DIR, "machine_attributes.parquet"), index=False)
print(f"  -> machine_attributes: {len(ma):,} rows")

# ---------------------------------------------------------------------------
# 3. job_events -> cleaned job lifecycle events
# ---------------------------------------------------------------------------
print("Processing job_events ...")
je_cols = ["timestamp", "missing_info", "job_id", "event_type", "user",
           "scheduling_class", "job_name", "logical_job_name"]
je = load_shards("job_events", je_cols)

JOB_EVENT_TYPE = {
    0: "SUBMIT", 1: "SCHEDULE", 2: "EVICT", 3: "FAIL", 4: "FINISH",
    5: "KILL", 6: "LOST", 7: "UPDATE_PENDING", 8: "UPDATE_RUNNING",
}
je["event_name"] = je["event_type"].map(JOB_EVENT_TYPE)
je = add_walltime(je, "timestamp")
je = je.drop_duplicates().sort_values(["job_id", "timestamp"]).reset_index(drop=True)
je.to_parquet(os.path.join(OUT_DIR, "job_events.parquet"), index=False)
print(f"  -> job_events: {len(je):,} rows, {je['job_id'].nunique():,} distinct jobs")

# ---------------------------------------------------------------------------
# 4. task_events -> cleaned task lifecycle + resource *requests*
# ---------------------------------------------------------------------------
print("Processing task_events ...")
te_cols = ["timestamp", "missing_info", "job_id", "task_index", "machine_id",
           "event_type", "user", "scheduling_class", "priority",
           "cpu_request", "memory_request", "disk_space_request",
           "different_machine_constraint"]
te = load_shards("task_events", te_cols)

TASK_EVENT_TYPE = {
    0: "SUBMIT", 1: "SCHEDULE", 2: "EVICT", 3: "FAIL", 4: "FINISH",
    5: "KILL", 6: "LOST", 7: "UPDATE_PENDING", 8: "UPDATE_RUNNING",
}
te["event_name"] = te["event_type"].map(TASK_EVENT_TYPE)
te = add_walltime(te, "timestamp")

# Sanity-clean: negative resource requests are not physically meaningful
for col in ["cpu_request", "memory_request", "disk_space_request"]:
    n_bad = (te[col] < 0).sum()
    if n_bad:
        print(f"  ! dropping {n_bad} rows with negative {col}")
    te = te[(te[col].isna()) | (te[col] >= 0)]

te = te.drop_duplicates().sort_values(["job_id", "task_index", "timestamp"]).reset_index(drop=True)
te.to_parquet(os.path.join(OUT_DIR, "task_events.parquet"), index=False)
print(f"  -> task_events: {len(te):,} rows, {te['job_id'].nunique():,} distinct jobs")

# ---------------------------------------------------------------------------
# 5. task_usage -> cleaned, machine x 5-min-interval resource USAGE
# ---------------------------------------------------------------------------
print("Processing task_usage (largest table, this takes longest) ...")
tu_cols = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory_usage", "assigned_memory_usage",
    "unmapped_page_cache", "total_page_cache", "maximum_memory_usage",
    "disk_io_time", "local_disk_space_usage", "maximum_cpu_rate",
    "maximum_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_portion",
    "aggregation_type", "sampled_cpu_usage",
]
# 48M rows x 20 float64 columns does not fit comfortably in RAM on this
# machine (~16GB) once pandas' filtering/copy overhead is added, so
# task_usage is streamed shard-by-shard instead of loaded whole:
#   - float32 instead of float64 for usage metrics (halves memory)
#   - each shard is cleaned and appended straight to a parquet file on disk
#   - the 5-min machine-level aggregate is accumulated incrementally, so the
#     only things resident in memory at once are one shard (~2.7M rows) and
#     the much smaller running aggregate (<= n_machines x n_intervals rows)
usage_cols = ["cpu_rate", "canonical_memory_usage", "assigned_memory_usage",
              "maximum_memory_usage", "disk_io_time", "local_disk_space_usage"]
f32_cols = usage_cols + ["unmapped_page_cache", "total_page_cache",
                          "maximum_cpu_rate", "maximum_disk_io_time",
                          "cycles_per_instruction", "memory_accesses_per_instruction",
                          "sample_portion", "sampled_cpu_usage"]
tu_dtype = {c: "float32" for c in f32_cols}
tu_dtype.update({"job_id": "int64", "task_index": "int64", "machine_id": "int64",
                  "start_time": "int64", "end_time": "int64", "aggregation_type": "float32"})

shard_files = sorted(glob.glob(os.path.join(RAW_DIR, "task_usage", "*.csv.gz")))
row_writer = None
row_path = os.path.join(OUT_DIR, "task_usage.parquet")
partial_aggs = []
total_rows_before = 0
total_rows_after_neg = 0
total_rows_after_cpi = 0
machines_seen = set()

for i, f in enumerate(shard_files, 1):
    chunk = pd.read_csv(
        f, header=None, names=tu_cols, dtype=tu_dtype,
        na_values=["", "NA"], compression="gzip",
    )
    total_rows_before += len(chunk)

    mask = pd.Series(True, index=chunk.index)
    for col in usage_cols:
        mask &= chunk[col].isna() | (chunk[col] >= 0)
    chunk = chunk[mask]
    total_rows_after_neg += len(chunk)

    chunk = chunk[(chunk["cycles_per_instruction"].isna()) |
                   ((chunk["cycles_per_instruction"] > 0) & (chunk["cycles_per_instruction"] < 100))]
    chunk = chunk[(chunk["memory_accesses_per_instruction"].isna()) |
                   ((chunk["memory_accesses_per_instruction"] >= 0) & (chunk["memory_accesses_per_instruction"] < 1))]
    total_rows_after_cpi += len(chunk)

    chunk = chunk.drop_duplicates().reset_index(drop=True)
    chunk = add_walltime(chunk, "start_time")
    machines_seen.update(chunk["machine_id"].unique().tolist())

    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if row_writer is None:
        row_writer = pq.ParquetWriter(row_path, table.schema)
    row_writer.write_table(table)

    chunk["bin_start"] = (chunk["start_time"] // BIN_MICROS) * BIN_MICROS
    partial_aggs.append(
        chunk.groupby(["machine_id", "bin_start"])
        .agg(
            cpu_rate_sum=("cpu_rate", "sum"),
            canonical_memory_usage_sum=("canonical_memory_usage", "sum"),
            assigned_memory_usage_sum=("assigned_memory_usage", "sum"),
            disk_io_time_sum=("disk_io_time", "sum"),
            n_tasks=("job_id", "count"),
        )
        .reset_index()
    )
    print(f"  shard {i}/{len(shard_files)} processed ({len(chunk):,} clean rows)")
    del chunk, table

if row_writer is not None:
    row_writer.close()

print(f"  ! dropped {total_rows_before - total_rows_after_neg:,} rows with negative usage values")
print(f"  ! dropped {total_rows_after_neg - total_rows_after_cpi:,} rows with implausible CPI/MAI")
print(f"  -> task_usage: {total_rows_after_cpi:,} rows, {len(machines_seen):,} distinct machines")

# Combine the small per-shard aggregates and re-sum in case the same
# (machine_id, start_time) pair happened to appear in more than one shard.
print("Aggregating task_usage -> per-machine 5-min utilization series ...")
machine_ts = (
    pd.concat(partial_aggs, ignore_index=True)
    .groupby(["machine_id", "bin_start"])
    .agg(
        cpu_rate_sum=("cpu_rate_sum", "sum"),
        canonical_memory_usage_sum=("canonical_memory_usage_sum", "sum"),
        assigned_memory_usage_sum=("assigned_memory_usage_sum", "sum"),
        disk_io_time_sum=("disk_io_time_sum", "sum"),
        n_tasks=("n_tasks", "sum"),
    )
    .reset_index()
    .rename(columns={"bin_start": "start_time"})
)
machine_ts = add_walltime(machine_ts, "start_time")
machine_ts = machine_ts.sort_values(["machine_id", "start_time"]).reset_index(drop=True)
machine_ts.to_parquet(os.path.join(OUT_DIR, "machine_utilization_5min.parquet"), index=False)
print(f"  -> machine_utilization_5min: {len(machine_ts):,} rows")

print("\nAll tables written to:", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    path = os.path.join(OUT_DIR, f)
    print(f"  {f:<40s} {os.path.getsize(path)/1024/1024:8.1f} MB")
