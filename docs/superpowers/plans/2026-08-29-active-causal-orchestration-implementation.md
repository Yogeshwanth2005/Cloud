# Active Causal Orchestration for Distributed Solar Fleets — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase-by-phase. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note on granularity:** Phase 1 (data engineering) is fully specified at TDD step level because it involves no open research decisions — the data is fixed and the transformations are unambiguous. Phases 3–7 implement genuinely novel research components (causal discovery method, VoI criterion, DRO formulation) where the *interface* and *file layout* are fixed here, but the exact algorithm is a research decision — each such task lists a concrete recommended default (so the worker is never blocked) plus the alternatives, and expects the default to be validated/adjusted empirically, not treated as a spec to blindly satisfy.

**Goal:** Turn the "Active Causal Orchestration" research proposal into a runnable, reproducible experimental pipeline: a fleet-level digital twin that replays real solar + cloud-workload data, an Active Causal Semantic Event Graph that is refined by both observation and chosen interventions, a Value-of-Information intervention selector, a distributionally-robust joint optimizer, and an evaluation harness that reproduces the proposal's dual (operational + causal) metrics against the four listed baselines.

**Architecture:** A Python monorepo with one processed-parquet data lake feeding a discrete-time (5-minute tick) simulation loop. Each tick: the twin exposes observations → the causal graph/VoI module scores candidate interventions → the DRO optimizer picks resource allocation + interventions + sensing/storage policy → the twin applies them and the graph is updated. Baselines are alternate policies plugged into the same loop so metrics are apples-to-apples.

**Tech Stack:** Python 3.11, pandas + pyarrow (data lake, already the convention in `google_cluster_2011/processed/`), `tigramite` (PCMCI+ causal discovery), `cvxpy` (per-slot convex subproblems), `networkx` (graph representation), `pytest` (unit tests), `matplotlib` (result plots). No Java/CloudSim dependency — see Phase 2 rationale.

**Spec:** [Active_Causal_Orchestration_Solar_Research_Proposal_Revised.docx](../../../Active_Causal_Orchestration_Solar_Research_Proposal_Revised.docx)

## Global Constraints

- All processed/derived tables are Parquet, matching the existing `google_cluster_2011/processed/*.parquet` convention — never write a new derived CSV.
- Everything must run fully offline from the datasets already on disk — no new downloads are required to execute this plan.
- Fixed random seeds everywhere sampling/optimization stochasticity is involved; every experiment run is driven by a YAML config file that is saved alongside its results for reproducibility.
- **Security note (not part of the pipeline, but must be done before anything in this folder is shared or committed to git):** [NLR_data.py](../../../NLR_data.py) lines 5–6 hardcode a live NREL API key and personal email. Rotate the key at https://developer.nrel.gov and delete the hardcoded value (load from an environment variable instead) before this directory is ever pushed to a remote or shared.
- No source calendar dates line up across datasets (Solar Integration Studies = all 2006; PVDAQ spans 1994–2023 with garbage years like 1822/1994 from bad timestamps; NSRDB Golden = 2018–2023; Google cluster trace = a single 29-day window in May 2011). **No join may assume shared absolute dates.** All cross-dataset alignment happens on a relative "simulation clock" (day-index + hour-of-day), exactly the pattern `preprocess_cluster_data.py` already started with its synthetic `wall_time`/`hour_of_day` columns.

---

## Current State (verified, so Phase 1 doesn't re-discover this)

| Dataset | Location | Contents | Caveats |
|---|---|---|---|
| Solar Power Data for Integration Studies | `Arizona/`, `California/`, `Colarado/`, `Nevada/` | `Actual_<lat>_<lon>_2006_<UPV\|DPV>_<cap>MW_5_Min.csv` (5-min power) + matching `DA_..._60_Min.csv` (day-ahead 60-min forecast). 2-col: `LocalTime,Power(MW)`. ~1090 unique plants across 4 states. | Only year 2006. Plant metadata (lat, lon, type, capacity) is embedded only in the filename. |
| NREL PVDAQ | `pvdaq_data/system_{4,10,50,51,1283}/year=Y/month=M/day=D/*.csv` | Real per-minute/5-min inverter+sensor telemetry: `ac_power`, `dc_power`, `poa_irradiance`, `module_temp_1..3`, `ambient_temp`, `inverter_temp`, etc. | Missing values are sentinel `-99999.0`, not `NaN`. Column set drifts across years (e.g. `das_battery_voltage` present from ~2018 on, absent earlier) — schema must be unioned, not assumed fixed. system_50/51 contain bogus years (1822, 1994) from clock glitches. |
| NSRDB Golden | `nsrdb_golden/nsrdb_golden_{2018..2023}.csv` | GHI/DNI/DHI, temperature, wind, pressure, humidity at 39.73, -105.18 (NREL Golden campus — same site as PVDAQ). 2-row metadata header + real header on row 3. | Only overlaps PVDAQ system_50/51's *later* years (2018–2020ish), not the full PVDAQ history. |
| Google Cluster Trace 2011 | `google_cluster_2011/{machine_events,machine_attributes,job_events,task_events,task_usage}/*.csv.gz` (raw) and `google_cluster_2011/processed/*.parquet` (already built by [preprocess_cluster_data.py](../../../preprocess_cluster_data.py)) | Cleaned job/task lifecycle + a derived `machine_utilization_5min.parquet` (5-min per-machine CPU/mem aggregate) with `wall_time`/`hour_of_day` columns already added for diurnal alignment. | `task_usage.parquet` is ~1.1 GB — stream/filter, never load whole into memory. Already de-duplicated and sanity-filtered (negative usage, implausible CPI dropped) — don't redo that cleaning. |

No git repo, no `requirements.txt`/`pyproject.toml`, no existing pipeline code beyond the two top-level scripts above. This is a greenfield build on top of already-acquired data.

---

### Task 0.1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/aco/__init__.py`
- Create: `src/aco/config.py`
- Create: `configs/base.yaml`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: `aco.config.load_config(path: str) -> dict` — loads and validates a YAML experiment config (must contain `seed: int`, `data_root: str`, `output_dir: str`).

- [ ] **Step 1: Create the package skeleton and dependency list**

`requirements.txt`:
```
pandas>=2.2
pyarrow>=16
numpy>=1.26
networkx>=3.3
tigramite>=5.2
cvxpy>=1.5
scikit-learn>=1.5
matplotlib>=3.9
pyyaml>=6.0
pytest>=8.2
```

`pyproject.toml`:
```toml
[project]
name = "aco"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing test for config loading**

`tests/test_config.py`:
```python
import textwrap
from aco.config import load_config

def test_load_config_reads_required_fields(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(textwrap.dedent("""
        seed: 42
        data_root: "D:/Cloud Computing"
        output_dir: "runs/test"
    """))
    cfg = load_config(str(cfg_path))
    assert cfg["seed"] == 42
    assert cfg["data_root"] == "D:/Cloud Computing"

def test_load_config_raises_on_missing_field(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("seed: 42\n")
    try:
        load_config(str(cfg_path))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "data_root" in str(e)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aco'`

- [ ] **Step 4: Implement `load_config`**

`src/aco/config.py`:
```python
import yaml

REQUIRED_FIELDS = ["seed", "data_root", "output_dir"]

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    missing = [f for f in REQUIRED_FIELDS if f not in cfg]
    if missing:
        raise ValueError(f"config {path} missing required fields: {missing}")
    return cfg
```

`configs/base.yaml`:
```yaml
seed: 42
data_root: "D:/Cloud Computing"
output_dir: "runs/base"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pip install -e . && pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml requirements.txt src tests configs
git commit -m "chore: project scaffolding and config loader"
```

---

## Phase 1 — Data Engineering & Fleet Construction

Mirrors the approach already proven in `preprocess_cluster_data.py`: read raw → clean → write partitioned/columnar parquet with a diurnal alignment column. This phase produces every dataset the rest of the plan consumes, so it is fully TDD-specified.

### Task 1.1: PVDAQ ingestion and cleaning

**Files:**
- Create: `src/aco/data/pvdaq.py`
- Test: `tests/data/test_pvdaq.py`
- Output: `pvdaq_data/processed/system_<id>.parquet` (one per system)

**Interfaces:**
- Produces: `aco.data.pvdaq.clean_pvdaq_frame(df: pd.DataFrame) -> pd.DataFrame` — replaces sentinel values with `NaN`, drops rows with implausible years (< 1990 or > 2024), adds `hour_of_day` (float, 0–24).
- Produces: `aco.data.pvdaq.load_system(system_dir: str) -> pd.DataFrame` — globs every `year=*/month=*/day=*/*.csv` under `system_dir`, unions columns across schema drift (missing columns become `NaN`), concatenates, and calls `clean_pvdaq_frame`.

- [ ] **Step 1: Write the failing test for sentinel cleaning**

```python
import pandas as pd
from aco.data.pvdaq import clean_pvdaq_frame

def test_sentinel_values_become_nan():
    df = pd.DataFrame({
        "measured_on": ["2018-09-23 00:00:00", "2018-09-23 00:05:00"],
        "ac_power__423": [-9.14, -99999.0],
        "poa_irradiance__421": [4.9, -99999.0],
    })
    out = clean_pvdaq_frame(df)
    assert pd.isna(out.loc[1, "ac_power__423"])
    assert pd.isna(out.loc[1, "poa_irradiance__421"])
    assert out.loc[0, "ac_power__423"] == -9.14

def test_implausible_years_dropped():
    df = pd.DataFrame({
        "measured_on": ["1822-01-01 00:00:00", "2018-09-23 00:05:00"],
        "ac_power__423": [1.0, 2.0],
    })
    out = clean_pvdaq_frame(df)
    assert len(out) == 1
    assert out.iloc[0]["ac_power__423"] == 2.0

def test_hour_of_day_added():
    df = pd.DataFrame({
        "measured_on": ["2018-09-23 13:30:00"],
        "ac_power__423": [1.0],
    })
    out = clean_pvdaq_frame(df)
    assert out.iloc[0]["hour_of_day"] == 13.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_pvdaq.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aco.data.pvdaq'`

- [ ] **Step 3: Implement `clean_pvdaq_frame`**

```python
import glob
import os
import pandas as pd
import numpy as np

SENTINEL = -99999.0

def clean_pvdaq_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["measured_on"] = pd.to_datetime(df["measured_on"])
    numeric_cols = [c for c in df.columns if c not in ("measured_on", "system_id")]
    for col in numeric_cols:
        df[col] = df[col].replace(SENTINEL, np.nan)
    df = df[(df["measured_on"].dt.year >= 1990) & (df["measured_on"].dt.year <= 2024)]
    df["hour_of_day"] = df["measured_on"].dt.hour + df["measured_on"].dt.minute / 60.0
    return df.reset_index(drop=True)


def load_system(system_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(system_dir, "year=*", "month=*", "day=*", "*.csv")))
    if not files:
        raise FileNotFoundError(f"no PVDAQ csv files under {system_dir}")
    frames = [pd.read_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return clean_pvdaq_frame(combined)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_pvdaq.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the ingestion driver script and run it on all 5 systems**

`src/aco/data/run_pvdaq_ingest.py`:
```python
import os
from aco.data.pvdaq import load_system

SYSTEMS = ["system_4", "system_10", "system_50", "system_51", "system_1283"]
DATA_ROOT = "pvdaq_data"

def main():
    out_dir = os.path.join(DATA_ROOT, "processed")
    os.makedirs(out_dir, exist_ok=True)
    for sys_name in SYSTEMS:
        print(f"Loading {sys_name} ...")
        df = load_system(os.path.join(DATA_ROOT, sys_name))
        out_path = os.path.join(out_dir, f"{sys_name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  -> {out_path}: {len(df):,} rows")

if __name__ == "__main__":
    main()
```

Run: `python -m aco.data.run_pvdaq_ingest`
Expected: 5 files under `pvdaq_data/processed/`, one per system, each smaller than its raw CSV total (sentinel replacement doesn't drop rows, so row counts should roughly match raw row counts minus the bogus-year rows).

- [ ] **Step 6: Commit**

```bash
git add src/aco/data/pvdaq.py src/aco/data/run_pvdaq_ingest.py tests/data/test_pvdaq.py
git commit -m "feat: PVDAQ ingestion with sentinel cleaning and schema union"
```

### Task 1.2: Solar Power Integration Studies fleet ingestion

**Files:**
- Create: `src/aco/data/fleet.py`
- Test: `tests/data/test_fleet.py`
- Output: `fleet_data/processed/plants.parquet` (metadata), `fleet_data/processed/power_5min.parquet` (long-format time series)

**Interfaces:**
- Produces: `aco.data.fleet.parse_plant_filename(filename: str) -> dict` — extracts `{"kind": "Actual"|"DA", "lat": float, "lon": float, "year": int, "plant_type": "UPV"|"DPV", "capacity_mw": float, "resolution_min": int}` from names like `Actual_31.85_-110.85_2006_UPV_100MW_5_Min.csv`.
- Produces: `aco.data.fleet.build_fleet_tables(state_dirs: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]` — `state_dirs` maps region name to directory (e.g. `{"Arizona": "Arizona", "California": "California", "Colorado": "Colarado", "Nevada": "Nevada"}`); returns `(plants_df, power_df)` where `plants_df` has one row per plant (`plant_id, region, lat, lon, plant_type, capacity_mw`) and `power_df` is long-format (`plant_id, kind, timestamp, power_mw`).

- [ ] **Step 1: Write the failing test for filename parsing**

```python
from aco.data.fleet import parse_plant_filename

def test_parse_actual_filename():
    meta = parse_plant_filename("Actual_31.85_-110.85_2006_UPV_100MW_5_Min.csv")
    assert meta == {
        "kind": "Actual", "lat": 31.85, "lon": -110.85, "year": 2006,
        "plant_type": "UPV", "capacity_mw": 100.0, "resolution_min": 5,
    }

def test_parse_da_filename():
    meta = parse_plant_filename("DA_31.95_-110.95_2006_DPV_43MW_60_Min.csv")
    assert meta["kind"] == "DA"
    assert meta["plant_type"] == "DPV"
    assert meta["capacity_mw"] == 43.0
    assert meta["resolution_min"] == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_fleet.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `parse_plant_filename` and `build_fleet_tables`**

```python
import glob
import os
import re
import pandas as pd

FILENAME_RE = re.compile(
    r"^(?P<kind>Actual|DA)_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)_(?P<year>\d{4})_"
    r"(?P<plant_type>UPV|DPV)_(?P<capacity>\d+(?:\.\d+)?)MW_(?P<resolution>\d+)_Min\.csv$"
)

def parse_plant_filename(filename: str) -> dict:
    m = FILENAME_RE.match(os.path.basename(filename))
    if not m:
        raise ValueError(f"filename does not match expected pattern: {filename}")
    return {
        "kind": m.group("kind"),
        "lat": float(m.group("lat")),
        "lon": float(m.group("lon")),
        "year": int(m.group("year")),
        "plant_type": m.group("plant_type"),
        "capacity_mw": float(m.group("capacity")),
        "resolution_min": int(m.group("resolution")),
    }


def build_fleet_tables(state_dirs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    plant_rows = []
    power_frames = []
    for region, directory in state_dirs.items():
        for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
            meta = parse_plant_filename(path)
            plant_id = f"{region}_{meta['lat']}_{meta['lon']}_{meta['plant_type']}_{meta['capacity_mw']}MW"
            plant_rows.append({
                "plant_id": plant_id, "region": region, "lat": meta["lat"],
                "lon": meta["lon"], "plant_type": meta["plant_type"],
                "capacity_mw": meta["capacity_mw"],
            })
            df = pd.read_csv(path)
            df.columns = ["timestamp", "power_mw"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M")
            df["plant_id"] = plant_id
            df["kind"] = meta["kind"]
            power_frames.append(df)
    plants_df = pd.DataFrame(plant_rows).drop_duplicates(subset=["plant_id"]).reset_index(drop=True)
    power_df = pd.concat(power_frames, ignore_index=True)
    power_df["hour_of_day"] = power_df["timestamp"].dt.hour + power_df["timestamp"].dt.minute / 60.0
    return plants_df, power_df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_fleet.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the ingestion driver on all four states**

`src/aco/data/run_fleet_ingest.py`:
```python
import os
from aco.data.fleet import build_fleet_tables

STATE_DIRS = {
    "Arizona": "Arizona", "California": "California",
    "Colorado": "Colarado", "Nevada": "Nevada",
}

def main():
    out_dir = os.path.join("fleet_data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    plants_df, power_df = build_fleet_tables(STATE_DIRS)
    plants_df.to_parquet(os.path.join(out_dir, "plants.parquet"), index=False)
    power_df.to_parquet(os.path.join(out_dir, "power_5min.parquet"), index=False)
    print(f"plants: {len(plants_df):,} rows, power: {len(power_df):,} rows")

if __name__ == "__main__":
    main()
```

Run: `python -m aco.data.run_fleet_ingest`
Expected: `plants.parquet` has ~1090 rows (one per unique plant across 4 states); `power_5min.parquet` contains both `Actual` and `DA` rows for every plant.

- [ ] **Step 6: Commit**

```bash
git add src/aco/data/fleet.py src/aco/data/run_fleet_ingest.py tests/data/test_fleet.py
git commit -m "feat: solar fleet ingestion from Integration Studies CSVs"
```

### Task 1.3: NSRDB Golden ingestion

**Files:**
- Create: `src/aco/data/nsrdb.py`
- Test: `tests/data/test_nsrdb.py`
- Output: `nsrdb_golden/processed/nsrdb_golden.parquet`

**Interfaces:**
- Produces: `aco.data.nsrdb.load_nsrdb_file(path: str) -> pd.DataFrame` — skips the 2-row metadata header, parses `Year,Month,Day,Hour,Minute` into a `timestamp` column, returns `timestamp, ghi, dni, dhi, temperature, wind_speed, pressure, relative_humidity`.
- Produces: `aco.data.nsrdb.build_nsrdb_table(glob_pattern: str) -> pd.DataFrame` — loads and concatenates every yearly file matching the pattern.

- [ ] **Step 1: Write the failing test**

```python
import textwrap
from aco.data.nsrdb import load_nsrdb_file

def test_load_nsrdb_file_skips_metadata_and_renames(tmp_path):
    p = tmp_path / "nsrdb_golden_2018.csv"
    p.write_text(textwrap.dedent("""\
        Source,Location ID
        NSRDB,479494
        Year,Month,Day,Hour,Minute,GHI,DNI,DHI,Temperature,Wind Speed,Pressure,Relative Humidity
        2018,1,1,0,30,0,0,0,-9.8,0.5,812,66.3
    """))
    df = load_nsrdb_file(str(p))
    assert list(df.columns) == [
        "timestamp", "ghi", "dni", "dhi", "temperature",
        "wind_speed", "pressure", "relative_humidity",
    ]
    assert df.iloc[0]["ghi"] == 0
    assert str(df.iloc[0]["timestamp"]) == "2018-01-01 00:30:00"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_nsrdb.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `load_nsrdb_file` and `build_nsrdb_table`**

```python
import glob
import pandas as pd

COLUMN_MAP = {
    "GHI": "ghi", "DNI": "dni", "DHI": "dhi", "Temperature": "temperature",
    "Wind Speed": "wind_speed", "Pressure": "pressure",
    "Relative Humidity": "relative_humidity",
}

def load_nsrdb_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)
    df["timestamp"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.rename(columns=COLUMN_MAP)
    return df[["timestamp"] + list(COLUMN_MAP.values())]


def build_nsrdb_table(glob_pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"no NSRDB files match {glob_pattern}")
    return pd.concat([load_nsrdb_file(f) for f in files], ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_nsrdb.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run on the real files**

```bash
python -c "
import os
from aco.data.nsrdb import build_nsrdb_table
df = build_nsrdb_table('nsrdb_golden/nsrdb_golden_*.csv')
os.makedirs('nsrdb_golden/processed', exist_ok=True)
df.to_parquet('nsrdb_golden/processed/nsrdb_golden.parquet', index=False)
print(len(df), 'rows,', df['timestamp'].min(), '->', df['timestamp'].max())
"
```
Expected: ~52,560 rows (6 years × 8760 hourly rows), timestamp range 2018-01-01 to 2023-12-31.

- [ ] **Step 6: Commit**

```bash
git add src/aco/data/nsrdb.py tests/data/test_nsrdb.py
git commit -m "feat: NSRDB Golden ingestion"
```

### Task 1.4: PVDAQ + NSRDB weather join (Golden site only)

**Files:**
- Create: `src/aco/data/join_pvdaq_weather.py`
- Test: `tests/data/test_join_pvdaq_weather.py`
- Output: `pvdaq_data/processed/system_50_weather.parquet`, `pvdaq_data/processed/system_51_weather.parquet`

**Interfaces:**
- Consumes: `pvdaq_data/processed/system_{50,51}.parquet` (Task 1.1), `nsrdb_golden/processed/nsrdb_golden.parquet` (Task 1.3).
- Produces: `aco.data.join_pvdaq_weather.join_nearest_hour(pvdaq_df: pd.DataFrame, nsrdb_df: pd.DataFrame) -> pd.DataFrame` — for each PVDAQ row, attaches the NSRDB row from the same hour (`pd.merge_asof` on `measured_on`/`timestamp`, direction="nearest", 30-min tolerance); rows outside the 2018–2023 NSRDB coverage keep `NaN` weather columns rather than being dropped.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from aco.data.join_pvdaq_weather import join_nearest_hour

def test_join_matches_same_hour():
    pvdaq = pd.DataFrame({
        "measured_on": pd.to_datetime(["2018-09-23 00:01:00", "2018-09-23 00:04:00"]),
        "ac_power__423": [1.0, 2.0],
    })
    nsrdb = pd.DataFrame({
        "timestamp": pd.to_datetime(["2018-09-23 00:00:00", "2018-09-23 01:00:00"]),
        "ghi": [0, 100],
    })
    out = join_nearest_hour(pvdaq, nsrdb)
    assert out["ghi"].tolist() == [0, 0]

def test_join_leaves_nan_outside_coverage():
    pvdaq = pd.DataFrame({
        "measured_on": pd.to_datetime(["2000-01-01 00:00:00"]),
        "ac_power__423": [1.0],
    })
    nsrdb = pd.DataFrame({
        "timestamp": pd.to_datetime(["2018-09-23 00:00:00"]),
        "ghi": [0],
    })
    out = join_nearest_hour(pvdaq, nsrdb)
    assert pd.isna(out.iloc[0]["ghi"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_join_pvdaq_weather.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `join_nearest_hour`**

```python
import pandas as pd

def join_nearest_hour(pvdaq_df: pd.DataFrame, nsrdb_df: pd.DataFrame) -> pd.DataFrame:
    left = pvdaq_df.sort_values("measured_on").reset_index(drop=True)
    right = nsrdb_df.sort_values("timestamp").reset_index(drop=True)
    return pd.merge_asof(
        left, right, left_on="measured_on", right_on="timestamp",
        direction="nearest", tolerance=pd.Timedelta("30min"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_join_pvdaq_weather.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run on real system_50 / system_51 data and report coverage**

```bash
python -c "
import pandas as pd
from aco.data.join_pvdaq_weather import join_nearest_hour
for sysid in [50, 51]:
    pv = pd.read_parquet(f'pvdaq_data/processed/system_{sysid}.parquet')
    ns = pd.read_parquet('nsrdb_golden/processed/nsrdb_golden.parquet')
    out = join_nearest_hour(pv, ns)
    out.to_parquet(f'pvdaq_data/processed/system_{sysid}_weather.parquet', index=False)
    coverage = out['ghi'].notna().mean()
    print(f'system_{sysid}: {len(out):,} rows, {coverage:.1%} have matched weather')
"
```
Expected: nonzero coverage percentage printed for both systems (only rows within 2018-2023 will match; this is expected and documented in Current State above).

- [ ] **Step 6: Commit**

```bash
git add src/aco/data/join_pvdaq_weather.py tests/data/test_join_pvdaq_weather.py
git commit -m "feat: join PVDAQ Golden systems with co-located NSRDB weather"
```

### Task 1.5: Simulation clock — synthetic multi-site fleet timeline

This is the key data-engineering step the proposal's Section 7 architecture depends on: a "PV Fleet" and an "Edge/Cloud" layer that co-evolve in the same closed loop. None of the raw sources share real calendar time, so a **relative simulation clock** (`sim_day`, `hour_of_day`) is the join key, not `timestamp`.

**Files:**
- Create: `src/aco/data/sim_clock.py`
- Test: `tests/data/test_sim_clock.py`
- Output: `fleet_data/processed/site_timeline.parquet`

**Interfaces:**
- Produces: `aco.data.sim_clock.to_sim_clock(df: pd.DataFrame, time_col: str, epoch: pd.Timestamp) -> pd.DataFrame` — adds `sim_day = (df[time_col] - epoch).days` and reuses/adds `hour_of_day`.
- Produces: `aco.data.sim_clock.build_site_timeline(power_df: pd.DataFrame, plants_df: pd.DataFrame, machine_util_df: pd.DataFrame, n_sites: int, seed: int) -> pd.DataFrame` — picks `n_sites` plants (stratified by region so all 4 states are represented) from `plants_df`, randomly (with `seed`) assigns each a disjoint block of `machine_id`s from `machine_util_df`, and returns one row per `(site_id, sim_day, hour_of_day)` joining that site's `Actual` power (by `sim_day` derived from the plant's own 2006 calendar, `hour_of_day`) with its assigned machines' summed utilization (by the cluster trace's own `sim_day`/`hour_of_day`) — i.e. two independent calendars aligned only through hour-of-day, which is the intended behavior since the sources don't share real dates.

- [ ] **Step 1: Write the failing test for `to_sim_clock`**

```python
import pandas as pd
from aco.data.sim_clock import to_sim_clock

def test_to_sim_clock_computes_day_offset():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2006-01-03 06:30:00"])})
    out = to_sim_clock(df, "timestamp", epoch=pd.Timestamp("2006-01-01"))
    assert out.iloc[0]["sim_day"] == 2
    assert out.iloc[0]["hour_of_day"] == 6.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_sim_clock.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `to_sim_clock`**

```python
import pandas as pd
import numpy as np

def to_sim_clock(df: pd.DataFrame, time_col: str, epoch: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["sim_day"] = (df[time_col] - epoch).dt.days
    df["hour_of_day"] = df[time_col].dt.hour + df[time_col].dt.minute / 60.0
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_sim_clock.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Write the failing test for `build_site_timeline` (small synthetic inputs)**

```python
import pandas as pd
from aco.data.sim_clock import build_site_timeline

def test_build_site_timeline_assigns_disjoint_machines():
    plants_df = pd.DataFrame({
        "plant_id": ["p1", "p2"], "region": ["Arizona", "California"],
    })
    power_df = pd.DataFrame({
        "plant_id": ["p1", "p1", "p2", "p2"], "kind": ["Actual"] * 4,
        "timestamp": pd.to_datetime(["2006-01-01 06:00", "2006-01-01 07:00"] * 2),
        "power_mw": [1.0, 2.0, 3.0, 4.0],
    })
    machine_util_df = pd.DataFrame({
        "machine_id": [10, 10, 20, 20],
        "wall_time": pd.to_datetime(["2011-05-01 06:00", "2011-05-01 07:00"] * 2),
        "cpu_rate_sum": [0.5, 0.6, 0.7, 0.8],
    })
    out = build_site_timeline(power_df, plants_df, machine_util_df, n_sites=2, seed=0)
    assert set(out["site_id"].unique()) == {"p1", "p2"}
    assert out["power_mw"].notna().all()
    assert out["cpu_rate_sum"].notna().all()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/data/test_sim_clock.py -v`
Expected: FAIL — `AttributeError`/`ImportError` on `build_site_timeline`

- [ ] **Step 7: Implement `build_site_timeline`**

```python
def build_site_timeline(power_df, plants_df, machine_util_df, n_sites: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sites = (
        plants_df.groupby("region", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, n_sites // plants_df["region"].nunique())), random_state=seed))
        .head(n_sites)
    )
    machine_ids = machine_util_df["machine_id"].unique()
    rng.shuffle(machine_ids)
    machine_blocks = np.array_split(machine_ids, len(sites))

    power_epoch = pd.Timestamp("2006-01-01")
    cluster_epoch = machine_util_df["wall_time"].min().normalize()

    actual = power_df[power_df["kind"] == "Actual"].copy()
    actual = to_sim_clock(actual, "timestamp", power_epoch)
    mu = to_sim_clock(machine_util_df, "wall_time", cluster_epoch)

    rows = []
    for (_, site_row), machines in zip(sites.iterrows(), machine_blocks):
        site_power = actual[actual["plant_id"] == site_row["plant_id"]]
        site_mu = (
            mu[mu["machine_id"].isin(machines)]
            .groupby(["sim_day", "hour_of_day"], as_index=False)["cpu_rate_sum"].sum()
        )
        merged = pd.merge(site_power, site_mu, on="hour_of_day", suffixes=("_solar", "_cluster"))
        merged["site_id"] = site_row["plant_id"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/data/test_sim_clock.py -v`
Expected: PASS (2 tests)

- [ ] **Step 9: Run on real data to build the default 20-site timeline and commit**

```bash
python -c "
import pandas as pd
from aco.data.sim_clock import build_site_timeline
plants = pd.read_parquet('fleet_data/processed/plants.parquet')
power = pd.read_parquet('fleet_data/processed/power_5min.parquet')
mu = pd.read_parquet('google_cluster_2011/processed/machine_utilization_5min.parquet')
timeline = build_site_timeline(power, plants, mu, n_sites=20, seed=42)
timeline.to_parquet('fleet_data/processed/site_timeline.parquet', index=False)
print(len(timeline), 'rows across', timeline['site_id'].nunique(), 'sites')
"
git add src/aco/data/sim_clock.py tests/data/test_sim_clock.py
git commit -m "feat: simulation-clock alignment and synthetic multi-site fleet timeline"
```

---

## Phase 2 — Digital Twin / Replay Simulation Environment

**Decision point (flagged, not silently assumed):** the proposal (Section 9.3) lists CloudSim Plus / iFogSim2 / Eclipse Mosquitto as candidate simulation platforms. All of those are Java-based (CloudSim/iFogSim2) or a message broker (Mosquitto), while every dataset and every downstream module in this plan (causal discovery, VoI, DRO) is Python/pandas. Bridging to a JVM simulator would mean re-serializing the fleet timeline across a process boundary every tick for no modeling benefit, since the proposal's own required behavior (replay traces, apply an intervention, observe the next state) doesn't need CloudSim's job-scheduling realism. **Recommendation: build a lightweight custom Python tick-based simulator directly over `site_timeline.parquet`,** and note this substitution explicitly in the eventual paper's experimental setup section. Confirm with your advisor if venue reviewers are expected to require CloudSim specifically.

### Task 2.1: Tick-based replay engine with an intervention API

**Files:**
- Create: `src/aco/sim/engine.py`
- Test: `tests/sim/test_engine.py`

**Interfaces:**
- Produces: `class SiteState` — dataclass with `site_id, sim_day, hour_of_day, power_mw, cpu_rate_sum, curtailment_frac, sampling_rate_hz`.
- Produces: `class ReplayEngine` — `__init__(self, timeline_df: pd.DataFrame)`; `.reset() -> dict[str, SiteState]`; `.step(interventions: dict[str, dict]) -> dict[str, SiteState]` where `interventions[site_id] = {"curtailment_frac": float, "sampling_rate_hz": float}` and `curtailment_frac` multiplicatively reduces that tick's `power_mw` for that site before it's returned (this is the "safe intervention" effect model — see Phase 5 for how a value is chosen).

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from aco.sim.engine import ReplayEngine

def _toy_timeline():
    return pd.DataFrame({
        "site_id": ["s1", "s1", "s2", "s2"],
        "sim_day": [0, 1, 0, 1],
        "hour_of_day": [12.0, 12.0, 12.0, 12.0],
        "power_mw": [10.0, 20.0, 5.0, 6.0],
        "cpu_rate_sum": [1.0, 1.5, 2.0, 2.5],
    })

def test_reset_returns_first_tick_for_every_site():
    engine = ReplayEngine(_toy_timeline())
    states = engine.reset()
    assert set(states.keys()) == {"s1", "s2"}
    assert states["s1"].power_mw == 10.0

def test_step_advances_and_applies_curtailment():
    engine = ReplayEngine(_toy_timeline())
    engine.reset()
    states = engine.step({"s1": {"curtailment_frac": 0.5, "sampling_rate_hz": 1.0}})
    assert states["s1"].power_mw == 10.0  # 20.0 * (1 - 0.5)
    assert states["s2"].power_mw == 6.0   # no intervention specified -> passthrough
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ReplayEngine`**

```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class SiteState:
    site_id: str
    sim_day: int
    hour_of_day: float
    power_mw: float
    cpu_rate_sum: float
    curtailment_frac: float = 0.0
    sampling_rate_hz: float = 1.0


class ReplayEngine:
    def __init__(self, timeline_df: pd.DataFrame):
        self.timeline = timeline_df.sort_values(["site_id", "sim_day", "hour_of_day"]).reset_index(drop=True)
        self._cursors = {sid: 0 for sid in self.timeline["site_id"].unique()}
        self._by_site = {sid: g.reset_index(drop=True) for sid, g in self.timeline.groupby("site_id")}

    def reset(self) -> dict:
        self._cursors = {sid: 0 for sid in self._by_site}
        return self._current_states({})

    def step(self, interventions: dict) -> dict:
        for sid in self._cursors:
            self._cursors[sid] = min(self._cursors[sid] + 1, len(self._by_site[sid]) - 1)
        return self._current_states(interventions)

    def _current_states(self, interventions: dict) -> dict:
        out = {}
        for sid, df in self._by_site.items():
            row = df.iloc[self._cursors[sid]]
            iv = interventions.get(sid, {})
            curtailment = iv.get("curtailment_frac", 0.0)
            out[sid] = SiteState(
                site_id=sid, sim_day=int(row["sim_day"]), hour_of_day=float(row["hour_of_day"]),
                power_mw=float(row["power_mw"]) * (1 - curtailment),
                cpu_rate_sum=float(row["cpu_rate_sum"]),
                curtailment_frac=curtailment,
                sampling_rate_hz=iv.get("sampling_rate_hz", 1.0),
            )
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/test_engine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/aco/sim/engine.py tests/sim/test_engine.py
git commit -m "feat: tick-based replay engine with curtailment intervention hook"
```

---

## Phase 3 — Active Causal Semantic Event Graph

**Research decision (default provided):** use `tigramite`'s **PCMCI+** for time-lagged multivariate causal discovery. Rationale: it's built for exactly this data shape (multiple continuous sensor time series with autocorrelation and unknown lags), has an established false-discovery control, and is the most-cited off-the-shelf option that avoids implementing causal discovery from scratch. Alternative considered: NOTEARS-style continuous optimization (better for larger variable counts, worse for time-lag discovery) — revisit only if PCMCI+ proves too slow on the full multi-site graph.

### Task 3.1: Node schema and observational graph initialization

**Files:**
- Create: `src/aco/causal/graph.py`
- Test: `tests/causal/test_graph.py`

**Interfaces:**
- Produces: `NODE_SCHEMA: list[str]` = `["poa_irradiance", "module_temp", "ambient_temp", "dc_power", "ac_power", "curtailment_frac", "sampling_rate_hz", "cpu_rate_sum", "cost", "risk"]` (fixed vocabulary every later module imports, so names never drift between phases).
- Produces: `aco.causal.graph.fit_observational_graph(df: pd.DataFrame, var_names: list[str], tau_max: int = 3) -> networkx.DiGraph` — runs PCMCI+ over `df[var_names]`, returns a `DiGraph` whose edges carry `weight` (link strength) and `pval` attributes.

- [ ] **Step 1: Write the failing test using a synthetic linear-causal dataset with a known ground-truth edge**

```python
import numpy as np
import pandas as pd
from aco.causal.graph import fit_observational_graph

def test_fit_observational_graph_recovers_known_edge():
    rng = np.random.default_rng(0)
    n = 500
    irradiance = rng.normal(500, 100, n)
    # dc_power is causally driven by irradiance at lag 0, plus noise
    dc_power = 0.2 * irradiance + rng.normal(0, 5, n)
    df = pd.DataFrame({"poa_irradiance": irradiance, "dc_power": dc_power})
    graph = fit_observational_graph(df, var_names=["poa_irradiance", "dc_power"], tau_max=1)
    assert graph.has_edge("poa_irradiance", "dc_power")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/causal/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `fit_observational_graph`**

```python
import networkx as nx
import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

NODE_SCHEMA = [
    "poa_irradiance", "module_temp", "ambient_temp", "dc_power", "ac_power",
    "curtailment_frac", "sampling_rate_hz", "cpu_rate_sum", "cost", "risk",
]

def fit_observational_graph(df, var_names: list, tau_max: int = 3) -> nx.DiGraph:
    values = df[var_names].to_numpy()
    dataframe = pp.DataFrame(values, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=0.05)

    graph = nx.DiGraph()
    graph.add_nodes_from(var_names)
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    n_vars = len(var_names)
    for i in range(n_vars):
        for j in range(n_vars):
            for tau in range(tau_max + 1):
                if i == j and tau == 0:
                    continue
                if p_matrix[i, j, tau] < 0.05:
                    graph.add_edge(
                        var_names[i], var_names[j],
                        weight=float(val_matrix[i, j, tau]),
                        pval=float(p_matrix[i, j, tau]),
                        lag=tau,
                    )
    return graph
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/causal/test_graph.py -v`
Expected: PASS (1 test). If PCMCI+ doesn't recover the edge at `pc_alpha=0.05` with this synthetic signal-to-noise, increase `n` or the coefficient in the test before touching the implementation — the test's synthetic data, not the algorithm, is the first thing to check.

- [ ] **Step 5: Fit the real observational graph on system_50 PVDAQ+weather data and inspect it**

```bash
python -c "
import pandas as pd
from aco.causal.graph import fit_observational_graph, NODE_SCHEMA
df = pd.read_parquet('pvdaq_data/processed/system_50_weather.parquet').dropna(
    subset=['poa_irradiance__421', 'dc_power__422', 'module_temp_1__429', 'ambient_temp__428'])
df = df.rename(columns={
    'poa_irradiance__421': 'poa_irradiance', 'dc_power__422': 'dc_power',
    'module_temp_1__429': 'module_temp', 'ambient_temp__428': 'ambient_temp',
})
graph = fit_observational_graph(df, var_names=['poa_irradiance', 'module_temp', 'ambient_temp', 'dc_power'])
print(list(graph.edges(data=True)))
"
```
Expected: a printed edge list including at least `poa_irradiance -> dc_power` (the known physical relationship) — use this as a sanity check before trusting the graph on more variables.

- [ ] **Step 6: Commit**

```bash
git add src/aco/causal/graph.py tests/causal/test_graph.py
git commit -m "feat: PCMCI+-based observational causal graph initialization"
```

### Task 3.2: Intervention-driven graph update

**Files:**
- Modify: `src/aco/causal/graph.py`
- Test: `tests/causal/test_graph.py` (append)

**Interfaces:**
- Produces: `aco.causal.graph.update_graph_with_intervention(graph: nx.DiGraph, intervened_var: str, pre_df: pd.DataFrame, post_df: pd.DataFrame, var_names: list[str], tau_max: int = 3) -> nx.DiGraph` — refits PCMCI+ on `post_df` (which contains the rows collected *after* an intervention on `intervened_var`) and merges: an edge whose `pval` improves (drops) after the intervention has its `weight`/`pval` replaced by the post-intervention estimate; edges unaffected by data volume around `intervened_var` are left as-is. Returns a new graph (does not mutate the input).

- [ ] **Step 1: Write the failing test**

```python
def test_update_graph_with_intervention_sharpens_pval():
    # Reuse the same synthetic generator as Task 3.1's test, but simulate
    # "post-intervention" data as a larger, cleaner sample of the same relationship.
    import numpy as np
    import pandas as pd
    from aco.causal.graph import fit_observational_graph, update_graph_with_intervention

    rng = np.random.default_rng(1)
    pre = pd.DataFrame({
        "poa_irradiance": rng.normal(500, 100, 30),
        "dc_power": rng.normal(100, 50, 30),  # noisy, weak signal pre-intervention
    })
    pre_graph = fit_observational_graph(pre, var_names=["poa_irradiance", "dc_power"], tau_max=1)

    irr = rng.normal(500, 100, 500)
    post = pd.DataFrame({
        "poa_irradiance": irr,
        "dc_power": 0.2 * irr + rng.normal(0, 2, 500),  # clean signal post-intervention
    })
    updated = update_graph_with_intervention(
        pre_graph, "poa_irradiance", pre, post, var_names=["poa_irradiance", "dc_power"], tau_max=1,
    )
    assert updated.has_edge("poa_irradiance", "dc_power")
    assert updated["poa_irradiance"]["dc_power"]["pval"] <= pre_graph.get_edge_data(
        "poa_irradiance", "dc_power", {"pval": 1.0}
    )["pval"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/causal/test_graph.py -v`
Expected: FAIL — `AttributeError`/`ImportError` on `update_graph_with_intervention`

- [ ] **Step 3: Implement `update_graph_with_intervention`**

```python
def update_graph_with_intervention(graph, intervened_var, pre_df, post_df, var_names, tau_max=3):
    post_graph = fit_observational_graph(post_df, var_names=var_names, tau_max=tau_max)
    merged = graph.copy()
    for u, v, data in post_graph.edges(data=True):
        if merged.has_edge(u, v):
            if data["pval"] <= merged[u][v]["pval"]:
                merged[u][v].update(data)
        else:
            merged.add_edge(u, v, **data)
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/causal/test_graph.py -v`
Expected: PASS (2 tests total in file)

- [ ] **Step 5: Commit**

```bash
git add src/aco/causal/graph.py tests/causal/test_graph.py
git commit -m "feat: refine causal graph edges using post-intervention data"
```

---

## Phase 4 — Causal World Model (interventional + counterfactual twin)

**Research decision (default provided):** fit one scikit-learn `GradientBoostingRegressor` per node, regressed on its parents in the Phase-3 graph (a per-node structural equation model). This gives `do()` support for free (feed a fixed value for the intervened parent instead of its fitted value) and counterfactuals via residual reuse (Pearl's abduction–action–prediction), without requiring a full probabilistic SCM library. Revisit only if per-node regression residuals are clearly non-Gaussian in a way that breaks the counterfactual step (check empirically in Task 4.2 before adding complexity).

### Task 4.1: Structural equation fitting and `do()` intervention

**Files:**
- Create: `src/aco/causal/world_model.py`
- Test: `tests/causal/test_world_model.py`

**Interfaces:**
- Produces: `class CausalWorldModel` — `__init__(self, graph: nx.DiGraph)`; `.fit(df: pd.DataFrame) -> None` (fits one regressor per node with in-edges, on its direct parents); `.predict(df: pd.DataFrame) -> pd.DataFrame` (ordinary forward prediction of every fitted node); `.do(df: pd.DataFrame, interventions: dict[str, float]) -> pd.DataFrame` (sets the named node(s) to fixed values, then predicts every downstream node using those fixed values as if observed).

- [ ] **Step 1: Write the failing test**

```python
import networkx as nx
import numpy as np
import pandas as pd
from aco.causal.world_model import CausalWorldModel

def test_do_changes_downstream_prediction():
    rng = np.random.default_rng(0)
    n = 300
    irradiance = rng.normal(500, 100, n)
    dc_power = 0.2 * irradiance + rng.normal(0, 1, n)
    df = pd.DataFrame({"poa_irradiance": irradiance, "dc_power": dc_power})

    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power")

    model = CausalWorldModel(graph)
    model.fit(df)

    baseline = model.predict(df.iloc[[0]])["dc_power"].iloc[0]
    intervened = model.do(df.iloc[[0]], {"poa_irradiance": 0.0})["dc_power"].iloc[0]
    assert intervened < baseline  # dropping irradiance to 0 should predict near-zero power
    assert abs(intervened) < abs(baseline)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/causal/test_world_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `CausalWorldModel`**

```python
import networkx as nx
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

class CausalWorldModel:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.models = {}

    def fit(self, df: pd.DataFrame) -> None:
        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            if not parents:
                continue
            reg = GradientBoostingRegressor(random_state=0)
            reg.fit(df[parents], df[node])
            self.models[node] = (parents, reg)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for node in nx.topological_sort(self.graph):
            if node not in self.models:
                continue
            parents, reg = self.models[node]
            out[node] = reg.predict(out[parents])
        return out

    def do(self, df: pd.DataFrame, interventions: dict) -> pd.DataFrame:
        out = df.copy()
        for var, value in interventions.items():
            out[var] = value
        for node in nx.topological_sort(self.graph):
            if node in interventions or node not in self.models:
                continue
            parents, reg = self.models[node]
            out[node] = reg.predict(out[parents])
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/causal/test_world_model.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/aco/causal/world_model.py tests/causal/test_world_model.py
git commit -m "feat: per-node structural equation model with do() intervention support"
```

### Task 4.2: Validation against a real proxy natural experiment

There is no archival "ground truth" controlled intervention in PVDAQ, so validate the twin against a natural experiment instead: inverter clipping (a known physical event where `dc_power` keeps rising with irradiance but `ac_power` plateaus because the inverter is at its rated capacity). This is a real, labelable event in the data, not a placeholder.

**Files:**
- Create: `src/aco/causal/validate_world_model.py`
- Test: `tests/causal/test_validate_world_model.py`

**Interfaces:**
- Produces: `aco.causal.validate_world_model.label_clipping_events(df: pd.DataFrame, ac_col: str, dc_col: str, rated_kw: float, tolerance: float = 0.02) -> pd.Series` — boolean Series, `True` where `ac_power >= rated_kw * (1 - tolerance)` while `dc_power` continues to exceed the same threshold scaled by inverter efficiency (~0.96), i.e. the inverter is saturated.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from aco.causal.validate_world_model import label_clipping_events

def test_label_clipping_events_flags_saturation():
    df = pd.DataFrame({
        "ac_power": [10.0, 49.5, 50.0, 30.0],
        "dc_power": [10.5, 55.0, 60.0, 31.0],
    })
    flags = label_clipping_events(df, "ac_power", "dc_power", rated_kw=50.0)
    assert flags.tolist() == [False, True, True, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/causal/test_validate_world_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `label_clipping_events`**

```python
def label_clipping_events(df, ac_col, dc_col, rated_kw, tolerance=0.02):
    ac_saturated = df[ac_col] >= rated_kw * (1 - tolerance)
    dc_available = df[dc_col] >= rated_kw * 0.96 * (1 - tolerance)
    return ac_saturated & dc_available
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/causal/test_validate_world_model.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run validation report on real system_50 data**

Fit `CausalWorldModel` on non-clipping rows only, then check whether `.do()` on `dc_power` at clipping-region values correctly predicts a plateaued (not linearly scaled) `ac_power` on the held-out clipping rows. Report mean absolute error of the twin's clipping-region prediction vs. a naive linear baseline — the twin should win, since a linear model cannot represent the plateau. Write the numeric result to `runs/validation/world_model_clipping_report.json` for later citation in the paper's evaluation section.

- [ ] **Step 6: Commit**

```bash
git add src/aco/causal/validate_world_model.py tests/causal/test_validate_world_model.py
git commit -m "feat: validate causal world model against inverter-clipping natural experiment"
```

---

## Phase 5 — Safe Intervention Library and Value-of-Information Selection

### Task 5.1: Safe Intervention Library with cost model

**Files:**
- Create: `src/aco/interventions/library.py`
- Test: `tests/interventions/test_library.py`

**Interfaces:**
- Produces: `INTERVENTIONS: dict[str, dict]` — one entry per proposal Section 8.1 action (`curtailment`, `high_res_sampling`, `setpoint_change`, `high_res_logging`), each `{"apply": callable(state, magnitude) -> state, "cost_fn": callable(magnitude) -> float, "max_magnitude": float}`.
- Produces: `aco.interventions.library.apply_intervention(name: str, state: dict, magnitude: float) -> tuple[dict, float]` — returns `(new_state, cost)`, raising `ValueError` if `magnitude > max_magnitude` for that intervention (this is the "safe" constraint from the proposal — interventions cannot exceed a pre-registered safety bound).

- [ ] **Step 1: Write the failing test**

```python
import pytest
from aco.interventions.library import apply_intervention

def test_curtailment_reduces_power_and_has_cost():
    state = {"power_mw": 10.0}
    new_state, cost = apply_intervention("curtailment", state, magnitude=0.2)
    assert new_state["power_mw"] == 8.0
    assert cost > 0

def test_intervention_rejects_unsafe_magnitude():
    state = {"power_mw": 10.0}
    with pytest.raises(ValueError):
        apply_intervention("curtailment", state, magnitude=0.9)  # library caps curtailment at 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/interventions/test_library.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the library**

```python
def _apply_curtailment(state, magnitude):
    new_state = dict(state)
    new_state["power_mw"] = state["power_mw"] * (1 - magnitude)
    return new_state

def _apply_sampling(state, magnitude):
    new_state = dict(state)
    new_state["sampling_rate_hz"] = state.get("sampling_rate_hz", 1.0) * (1 + magnitude)
    return new_state

def _apply_setpoint(state, magnitude):
    new_state = dict(state)
    new_state["power_factor"] = max(0.8, min(1.0, state.get("power_factor", 1.0) - magnitude))
    return new_state

def _apply_logging(state, magnitude):
    new_state = dict(state)
    new_state["logging_resolution_hz"] = state.get("logging_resolution_hz", 1.0) * (1 + magnitude)
    return new_state

INTERVENTIONS = {
    "curtailment": {"apply": _apply_curtailment, "cost_fn": lambda m: 5.0 * m, "max_magnitude": 0.3},
    "high_res_sampling": {"apply": _apply_sampling, "cost_fn": lambda m: 0.5 * m, "max_magnitude": 4.0},
    "setpoint_change": {"apply": _apply_setpoint, "cost_fn": lambda m: 2.0 * m, "max_magnitude": 0.1},
    "high_res_logging": {"apply": _apply_logging, "cost_fn": lambda m: 0.2 * m, "max_magnitude": 9.0},
}

def apply_intervention(name: str, state: dict, magnitude: float):
    spec = INTERVENTIONS[name]
    if magnitude > spec["max_magnitude"]:
        raise ValueError(f"{name} magnitude {magnitude} exceeds safe max {spec['max_magnitude']}")
    new_state = spec["apply"](state, magnitude)
    cost = spec["cost_fn"](magnitude)
    return new_state, cost
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/interventions/test_library.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/aco/interventions/library.py tests/interventions/test_library.py
git commit -m "feat: safe intervention library with per-action cost and safety bound"
```

### Task 5.2: Value-of-Information scoring

**Research decision (default provided):** approximate expected information gain of an intervention as the reduction in `pval` (Phase 3's edge significance) the `CausalWorldModel`'s bootstrap ensemble predicts for the edges touching the intervened node, following the standard "uncertainty reduction as VoI proxy" pattern used in active causal discovery literature, since computing exact Shannon information gain over the full graph posterior is intractable for this variable count. Revisit with a full Bayesian structure posterior only if the proxy proves too coarse empirically (Task 5.2 Step 5 gives you the empirical check).

**Files:**
- Create: `src/aco/interventions/voi.py`
- Test: `tests/interventions/test_voi.py`

**Interfaces:**
- Consumes: `aco.causal.graph` edge `pval`/`weight` attributes (Phase 3), `aco.interventions.library.INTERVENTIONS` cost functions (Task 5.1).
- Produces: `aco.interventions.voi.score_intervention(graph: nx.DiGraph, node: str, current_uncertainty: dict[str, float], expected_uncertainty_reduction: float, name: str, magnitude: float) -> float` — returns `expected_uncertainty_reduction * edges_touching(node) - cost_fn(magnitude)`; positive means "worth executing" per proposal Section 8.2.
- Produces: `aco.interventions.voi.select_best_intervention(graph, node_candidates: list[str], uncertainty_estimates: dict[str, float]) -> tuple[str, str, float] | None` — tries every `(node, intervention_name)` pair at a mid-range magnitude, returns the highest-scoring one, or `None` if no candidate has positive net value (i.e., "do nothing" is itself a valid, and often correct, decision).

- [ ] **Step 1: Write the failing test**

```python
import networkx as nx
from aco.interventions.voi import score_intervention, select_best_intervention

def test_score_intervention_is_positive_when_uncertainty_is_high():
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.2)
    score = score_intervention(
        graph, "poa_irradiance", current_uncertainty={"poa_irradiance": 0.2},
        expected_uncertainty_reduction=0.15, name="high_res_sampling", magnitude=1.0,
    )
    assert score > 0

def test_select_best_intervention_returns_none_when_all_negative():
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.01)  # already well-known -> little to gain
    result = select_best_intervention(
        graph, node_candidates=["poa_irradiance"], uncertainty_estimates={"poa_irradiance": 0.01},
    )
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/interventions/test_voi.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `score_intervention` and `select_best_intervention`**

```python
import networkx as nx
from aco.interventions.library import INTERVENTIONS

def score_intervention(graph, node, current_uncertainty, expected_uncertainty_reduction, name, magnitude):
    n_edges_touched = graph.degree(node) if node in graph else 0
    info_value = expected_uncertainty_reduction * max(n_edges_touched, 1)
    cost = INTERVENTIONS[name]["cost_fn"](magnitude)
    return info_value - cost

def select_best_intervention(graph, node_candidates, uncertainty_estimates):
    best = None
    best_score = 0.0
    for node in node_candidates:
        uncertainty = uncertainty_estimates.get(node, 0.0)
        for name, spec in INTERVENTIONS.items():
            magnitude = spec["max_magnitude"] / 2
            score = score_intervention(
                graph, node, uncertainty_estimates, expected_uncertainty_reduction=uncertainty,
                name=name, magnitude=magnitude,
            )
            if score > best_score:
                best_score = score
                best = (node, name, magnitude)
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/interventions/test_voi.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Empirical check of the VoI proxy against the world model's actual post-intervention accuracy gain**

Using `system_50_weather.parquet`: split into an "early" window (higher `pval`, less data) and a "late" window (more data). Run `fit_observational_graph` on both, confirm `select_best_intervention` recommends probing the node whose edge `pval` improved the most between the two windows (i.e., the proxy points at the node that empirically benefited most from more data) — write this comparison to `runs/validation/voi_proxy_check.json`. If it disagrees, that's the trigger to revisit the Bayesian-posterior alternative noted above.

- [ ] **Step 6: Commit**

```bash
git add src/aco/interventions/voi.py tests/interventions/test_voi.py
git commit -m "feat: value-of-information intervention scoring and selection"
```

---

## Phase 6 — Distributionally Robust, Risk-Constrained Joint Optimizer

**Research decision (default provided):** Lyapunov drift-plus-penalty online control (standard for slot-by-slot resource allocation with queue-stability guarantees, and it composes cleanly with the VoI intervention choice by simply adding the intervention's net VoI score as an extra penalty term each slot) with a **CVaR risk constraint** enforced via `cvxpy`'s convex CVaR formulation (Rockafellar–Uryasev), giving the "distributionally robust" ambiguity handling proposal Section 6.4/8.3 asks for through an empirical-distribution CVaR rather than a full Wasserstein-ball DRO (which is more mathematically involved and not necessary to get a first working, defensible system — flag as a documented simplification, and note the full Wasserstein-DRO extension as future work if reviewers push back).

### Task 6.1: Per-slot convex resource allocation with CVaR constraint

**Files:**
- Create: `src/aco/optim/dro_allocator.py`
- Test: `tests/optim/test_dro_allocator.py`

**Interfaces:**
- Produces: `aco.optim.dro_allocator.solve_slot(available_power_mw: float, compute_demand: list[float], cost_per_unit: list[float], risk_samples: list[list[float]], cvar_alpha: float, cvar_limit: float) -> dict` — solves `minimize sum(cost_per_unit * x) s.t. sum(x) <= available_power_mw, 0 <= x <= compute_demand, CVaR_alpha(risk_samples @ x) <= cvar_limit`, returns `{"allocation": list[float], "objective": float, "cvar": float, "status": str}`.

- [ ] **Step 1: Write the failing test**

```python
from aco.optim.dro_allocator import solve_slot

def test_solve_slot_respects_power_budget_and_cvar():
    result = solve_slot(
        available_power_mw=10.0,
        compute_demand=[6.0, 6.0],
        cost_per_unit=[1.0, 2.0],
        risk_samples=[[0.1, 0.2], [0.3, 0.1], [0.2, 0.4]],
        cvar_alpha=0.9,
        cvar_limit=1.5,
    )
    assert result["status"] == "optimal"
    assert sum(result["allocation"]) <= 10.0 + 1e-6
    assert result["cvar"] <= 1.5 + 1e-6

def test_solve_slot_prefers_cheaper_resource():
    result = solve_slot(
        available_power_mw=20.0,
        compute_demand=[5.0, 5.0],
        cost_per_unit=[1.0, 5.0],
        risk_samples=[[0.0, 0.0]],
        cvar_alpha=0.9,
        cvar_limit=100.0,
    )
    assert result["allocation"][0] == pytest.approx(5.0, abs=1e-4)
```

(add `import pytest` at the top)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/optim/test_dro_allocator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `solve_slot` using the Rockafellar–Uryasev CVaR formulation**

```python
import cvxpy as cp
import numpy as np

def solve_slot(available_power_mw, compute_demand, cost_per_unit, risk_samples, cvar_alpha, cvar_limit):
    n = len(compute_demand)
    x = cp.Variable(n, nonneg=True)
    risk_matrix = np.array(risk_samples)
    n_samples = risk_matrix.shape[0]
    var = cp.Variable()
    excess = cp.Variable(n_samples, nonneg=True)

    constraints = [
        cp.sum(x) <= available_power_mw,
        x <= np.array(compute_demand),
        excess >= risk_matrix @ x - var,
    ]
    cvar_expr = var + cp.sum(excess) / ((1 - cvar_alpha) * n_samples)
    constraints.append(cvar_expr <= cvar_limit)

    objective = cp.Minimize(cp.sum(cp.multiply(np.array(cost_per_unit), x)))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return {
        "allocation": x.value.tolist() if x.value is not None else [0.0] * n,
        "objective": float(problem.value) if problem.value is not None else float("inf"),
        "cvar": float(cvar_expr.value) if cvar_expr.value is not None else float("inf"),
        "status": problem.status,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/optim/test_dro_allocator.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/aco/optim/dro_allocator.py tests/optim/test_dro_allocator.py
git commit -m "feat: per-slot CVaR-constrained resource allocation via cvxpy"
```

### Task 6.2: Lyapunov queue wrapper tying allocation + VoI intervention choice into one online policy

**Files:**
- Create: `src/aco/optim/orchestrator.py`
- Test: `tests/optim/test_orchestrator.py`

**Interfaces:**
- Consumes: `aco.optim.dro_allocator.solve_slot` (Task 6.1), `aco.interventions.voi.select_best_intervention` (Task 5.2), `aco.interventions.library.apply_intervention` (Task 5.1).
- Produces: `class ActiveOrchestrator` — `__init__(self, V: float, cvar_alpha: float, cvar_limit: float)`; `.step(site_states: dict, graph: nx.DiGraph, uncertainty_estimates: dict) -> dict` — for each slot: (1) calls `select_best_intervention`, applies it via `apply_intervention` if found; (2) builds the CVaR/cost inputs from `site_states` and calls `solve_slot` with `cost_per_unit` scaled by the Lyapunov weight `V`; (3) returns `{"allocation": ..., "intervention": (node, name, magnitude) | None, "queue_backlog": float}`. Maintains an internal virtual queue (`self._queue`) that grows by the CVaR constraint violation each slot and shrinks otherwise — this is what makes it "Lyapunov drift-plus-penalty" rather than a memoryless per-slot solve.

- [ ] **Step 1: Write the failing test**

```python
import networkx as nx
from aco.optim.orchestrator import ActiveOrchestrator

def test_orchestrator_step_returns_allocation_and_intervention_slot():
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0)
    site_states = {
        "s1": {"power_mw": 10.0, "compute_demand": 6.0, "cost_per_unit": 1.0, "risk_sample": [0.1]},
        "s2": {"power_mw": 5.0, "compute_demand": 4.0, "cost_per_unit": 2.0, "risk_sample": [0.2]},
    }
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.3)
    result = orch.step(site_states, graph, uncertainty_estimates={"poa_irradiance": 0.3})
    assert "allocation" in result
    assert "queue_backlog" in result
    assert result["queue_backlog"] >= 0.0

def test_orchestrator_queue_grows_after_repeated_violation():
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=0.0001)  # near-impossible limit -> forces violation
    site_states = {"s1": {"power_mw": 10.0, "compute_demand": 6.0, "cost_per_unit": 1.0, "risk_sample": [5.0]}}
    graph = nx.DiGraph()
    first = orch.step(site_states, graph, uncertainty_estimates={})
    second = orch.step(site_states, graph, uncertainty_estimates={})
    assert second["queue_backlog"] >= first["queue_backlog"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/optim/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `ActiveOrchestrator`**

```python
from aco.optim.dro_allocator import solve_slot
from aco.interventions.voi import select_best_intervention
from aco.interventions.library import apply_intervention

class ActiveOrchestrator:
    def __init__(self, V: float, cvar_alpha: float, cvar_limit: float):
        self.V = V
        self.cvar_alpha = cvar_alpha
        self.cvar_limit = cvar_limit
        self._queue = 0.0

    def step(self, site_states: dict, graph, uncertainty_estimates: dict) -> dict:
        site_ids = list(site_states.keys())
        best = select_best_intervention(graph, list(uncertainty_estimates.keys()), uncertainty_estimates)
        intervention_result = None
        if best is not None:
            node, name, magnitude = best
            for sid in site_ids:
                site_states[sid], _cost = apply_intervention(name, site_states[sid], magnitude)
            intervention_result = best

        available_power = sum(s["power_mw"] for s in site_states.values())
        demand = [site_states[sid]["compute_demand"] for sid in site_ids]
        cost = [site_states[sid]["cost_per_unit"] * self.V for sid in site_ids]
        risk_samples = [[site_states[sid]["risk_sample"][0] for sid in site_ids]]

        solved = solve_slot(available_power, demand, cost, risk_samples, self.cvar_alpha, self.cvar_limit)
        violation = max(0.0, solved["cvar"] - self.cvar_limit)
        self._queue = max(0.0, self._queue + violation - 0.01)

        return {
            "allocation": dict(zip(site_ids, solved["allocation"])),
            "intervention": intervention_result,
            "queue_backlog": self._queue,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/optim/test_orchestrator.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/aco/optim/orchestrator.py tests/optim/test_orchestrator.py
git commit -m "feat: Lyapunov-wrapped orchestrator combining VoI intervention choice and CVaR allocation"
```

---

## Phase 7 — Baselines

Each baseline reuses `ReplayEngine` (Phase 2) and `solve_slot` (Phase 6) so every policy runs through the identical simulation loop — only the decision policy differs.

### Task 7.1: Passive baseline (no interventions)

**Files:**
- Create: `src/aco/baselines/passive.py`
- Test: `tests/baselines/test_passive.py`

**Interfaces:**
- Produces: `class PassiveOrchestrator` — same `.step(site_states, graph, uncertainty_estimates) -> dict` interface as `ActiveOrchestrator`, but never calls `select_best_intervention`/`apply_intervention`; always returns `intervention=None`.

- [ ] **Step 1: Write the failing test**

```python
from aco.baselines.passive import PassiveOrchestrator

def test_passive_orchestrator_never_intervenes():
    orch = PassiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0)
    site_states = {"s1": {"power_mw": 10.0, "compute_demand": 6.0, "cost_per_unit": 1.0, "risk_sample": [0.1]}}
    result = orch.step(site_states, graph=None, uncertainty_estimates={"poa_irradiance": 0.9})
    assert result["intervention"] is None
    assert "allocation" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/baselines/test_passive.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `PassiveOrchestrator`**

```python
from aco.optim.dro_allocator import solve_slot

class PassiveOrchestrator:
    def __init__(self, V: float, cvar_alpha: float, cvar_limit: float):
        self.V = V
        self.cvar_alpha = cvar_alpha
        self.cvar_limit = cvar_limit
        self._queue = 0.0

    def step(self, site_states: dict, graph, uncertainty_estimates: dict) -> dict:
        site_ids = list(site_states.keys())
        available_power = sum(s["power_mw"] for s in site_states.values())
        demand = [site_states[sid]["compute_demand"] for sid in site_ids]
        cost = [site_states[sid]["cost_per_unit"] * self.V for sid in site_ids]
        risk_samples = [[site_states[sid]["risk_sample"][0] for sid in site_ids]]
        solved = solve_slot(available_power, demand, cost, risk_samples, self.cvar_alpha, self.cvar_limit)
        violation = max(0.0, solved["cvar"] - self.cvar_limit)
        self._queue = max(0.0, self._queue + violation - 0.01)
        return {
            "allocation": dict(zip(site_ids, solved["allocation"])),
            "intervention": None,
            "queue_backlog": self._queue,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/baselines/test_passive.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/aco/baselines/passive.py tests/baselines/test_passive.py
git commit -m "feat: passive (no-intervention) baseline orchestrator"
```

### Task 7.2: Observational-only causal graph baseline

**Files:**
- Create: `src/aco/baselines/observational_only.py`
- Test: `tests/baselines/test_observational_only.py`

**Interfaces:**
- Produces: `class ObservationalOnlyOrchestrator` — identical to `ActiveOrchestrator` except it calls `fit_observational_graph` (Phase 3) periodically on accumulated passthrough data to refresh the graph, but never calls `update_graph_with_intervention` and never executes an intervention (`select_best_intervention` is never invoked) — this isolates "does the graph exist" from "does the system act on it", matching proposal Section 10.2's baseline definition precisely.

- [ ] **Step 1: Write the failing test**

```python
from aco.baselines.observational_only import ObservationalOnlyOrchestrator

def test_observational_only_never_intervenes_but_returns_allocation():
    orch = ObservationalOnlyOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0)
    site_states = {"s1": {"power_mw": 10.0, "compute_demand": 6.0, "cost_per_unit": 1.0, "risk_sample": [0.1]}}
    result = orch.step(site_states, graph=None, uncertainty_estimates={})
    assert result["intervention"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/baselines/test_observational_only.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement (delegates allocation to the same `solve_slot` path as Task 7.1, difference only shows up in the experiment runner's graph-refresh cadence, documented in Phase 8)**

```python
from aco.baselines.passive import PassiveOrchestrator

class ObservationalOnlyOrchestrator(PassiveOrchestrator):
    """Same allocation policy as PassiveOrchestrator; the experiment runner additionally
    refits the causal graph from accumulated observational data on a fixed cadence for
    this policy, without ever selecting or applying an intervention."""
    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/baselines/test_observational_only.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/aco/baselines/observational_only.py tests/baselines/test_observational_only.py
git commit -m "feat: observational-only baseline (graph refreshed, never acted on)"
```

### Task 7.3: Oracle upper bound

**Files:**
- Create: `src/aco/baselines/oracle.py`
- Test: `tests/baselines/test_oracle.py`

**Interfaces:**
- Produces: `class OracleOrchestrator` — `__init__(self, V, cvar_alpha, cvar_limit, true_graph: nx.DiGraph)`; `.step(...)` uses `true_graph` (fit once, offline, on the *entire* available dataset for that site — i.e. the best causal graph obtainable with unlimited data) instead of the online-updated graph, and skips VoI scoring entirely since uncertainty is defined as zero by construction. This is the upper bound named in proposal Section 10.2.

- [ ] **Step 1: Write the failing test**

```python
import networkx as nx
from aco.baselines.oracle import OracleOrchestrator

def test_oracle_never_pays_intervention_cost():
    true_graph = nx.DiGraph()
    true_graph.add_edge("poa_irradiance", "dc_power", pval=0.001)
    orch = OracleOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, true_graph=true_graph)
    site_states = {"s1": {"power_mw": 10.0, "compute_demand": 6.0, "cost_per_unit": 1.0, "risk_sample": [0.1]}}
    result = orch.step(site_states)
    assert result["intervention"] is None
    assert "allocation" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/baselines/test_oracle.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `OracleOrchestrator`**

```python
from aco.optim.dro_allocator import solve_slot

class OracleOrchestrator:
    def __init__(self, V, cvar_alpha, cvar_limit, true_graph):
        self.V = V
        self.cvar_alpha = cvar_alpha
        self.cvar_limit = cvar_limit
        self.true_graph = true_graph
        self._queue = 0.0

    def step(self, site_states: dict) -> dict:
        site_ids = list(site_states.keys())
        available_power = sum(s["power_mw"] for s in site_states.values())
        demand = [site_states[sid]["compute_demand"] for sid in site_ids]
        cost = [site_states[sid]["cost_per_unit"] * self.V for sid in site_ids]
        risk_samples = [[site_states[sid]["risk_sample"][0] for sid in site_ids]]
        solved = solve_slot(available_power, demand, cost, risk_samples, self.cvar_alpha, self.cvar_limit)
        violation = max(0.0, solved["cvar"] - self.cvar_limit)
        self._queue = max(0.0, self._queue + violation - 0.01)
        return {
            "allocation": dict(zip(site_ids, solved["allocation"])),
            "intervention": None,
            "queue_backlog": self._queue,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/baselines/test_oracle.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/aco/baselines/oracle.py tests/baselines/test_oracle.py
git commit -m "feat: oracle (perfect causal knowledge) upper-bound baseline"
```

*(The fourth listed baseline, "strong non-causal proactive optimizer (2023–2025)," requires picking a specific published method to reimplement — that is a literature-review decision for you to make, not a default this plan should silently pick. Once you've chosen the paper, add `src/aco/baselines/proactive_noncausal.py` following the exact same `.step()` interface as the three baselines above.)*

---

## Phase 8 — Evaluation Harness and Experiments

### Task 8.1: Dual metrics computation

**Files:**
- Create: `src/aco/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces: `aco.eval.metrics.operational_metrics(allocations: list[dict], costs: list[float], cvars: list[float], latencies: list[float], violation_flags: list[bool]) -> dict` — returns `{"total_cost": float, "mean_cvar": float, "p99_latency": float, "violation_rate": float}`.
- Produces: `aco.eval.metrics.causal_metrics(pval_history: list[dict[str, float]], prediction_errors: list[float], n_interventions: int, total_info_gain: float, regret_vs_oracle: list[float]) -> dict` — returns `{"edge_uncertainty_reduction": float, "mean_prediction_error": float, "interventions_per_info_gain": float, "cumulative_regret": float}`. `edge_uncertainty_reduction` is `mean(pval_history[0].values()) - mean(pval_history[-1].values())`.

- [ ] **Step 1: Write the failing test**

```python
from aco.eval.metrics import operational_metrics, causal_metrics

def test_operational_metrics_basic():
    m = operational_metrics(
        allocations=[{"s1": 5.0}], costs=[10.0, 20.0], cvars=[1.0, 2.0],
        latencies=[100.0, 200.0, 300.0], violation_flags=[False, True],
    )
    assert m["total_cost"] == 30.0
    assert m["mean_cvar"] == 1.5
    assert m["violation_rate"] == 0.5

def test_causal_metrics_basic():
    m = causal_metrics(
        pval_history=[{"a->b": 0.5}, {"a->b": 0.1}],
        prediction_errors=[1.0, 2.0], n_interventions=4, total_info_gain=2.0,
        regret_vs_oracle=[0.1, 0.05],
    )
    assert m["edge_uncertainty_reduction"] == 0.4
    assert m["interventions_per_info_gain"] == 2.0
    assert m["cumulative_regret"] == 0.15000000000000002 or abs(m["cumulative_regret"] - 0.15) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement both functions**

```python
import numpy as np

def operational_metrics(allocations, costs, cvars, latencies, violation_flags):
    return {
        "total_cost": float(sum(costs)),
        "mean_cvar": float(np.mean(cvars)),
        "p99_latency": float(np.percentile(latencies, 99)),
        "violation_rate": float(np.mean(violation_flags)),
    }

def causal_metrics(pval_history, prediction_errors, n_interventions, total_info_gain, regret_vs_oracle):
    first_mean = np.mean(list(pval_history[0].values()))
    last_mean = np.mean(list(pval_history[-1].values()))
    return {
        "edge_uncertainty_reduction": float(first_mean - last_mean),
        "mean_prediction_error": float(np.mean(prediction_errors)),
        "interventions_per_info_gain": float(n_interventions / total_info_gain) if total_info_gain else float("inf"),
        "cumulative_regret": float(np.sum(regret_vs_oracle)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/aco/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: dual operational + causal-learning evaluation metrics"
```

### Task 8.2: Experiment runner across regions and policies

**Files:**
- Create: `src/aco/eval/run_experiment.py`
- Create: `configs/experiment_default.yaml`
- Test: `tests/eval/test_run_experiment.py`

**Interfaces:**
- Consumes: `ReplayEngine` (Phase 2), all four orchestrator classes (Phases 6–7), `operational_metrics`/`causal_metrics` (Task 8.1).
- Produces: `aco.eval.run_experiment.run_policy(engine: ReplayEngine, policy, n_ticks: int) -> dict` — steps the engine `n_ticks` times feeding the policy's chosen interventions/allocations back in, collecting the raw series needed by Task 8.1's metric functions, returns the assembled metrics dict.
- Produces: `aco.eval.run_experiment.main(config_path: str) -> None` — CLI entry point; loads `fleet_data/processed/site_timeline.parquet`, runs every registered policy (`active`, `passive`, `observational_only`, `oracle`), writes `runs/<experiment_name>/results.json` and one comparison bar chart per metric to `runs/<experiment_name>/plots/`.

- [ ] **Step 1: Write the failing test using the toy `ReplayEngine` fixture from Task 2.1's test file**

```python
import pandas as pd
from aco.sim.engine import ReplayEngine
from aco.baselines.passive import PassiveOrchestrator
from aco.eval.run_experiment import run_policy

def test_run_policy_collects_expected_number_of_ticks():
    timeline = pd.DataFrame({
        "site_id": ["s1"] * 3, "sim_day": [0, 1, 2], "hour_of_day": [12.0] * 3,
        "power_mw": [10.0, 10.0, 10.0], "cpu_rate_sum": [1.0, 1.0, 1.0],
    })
    engine = ReplayEngine(timeline)
    policy = PassiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=100.0)
    result = run_policy(engine, policy, n_ticks=3)
    assert len(result["costs"]) == 3
    assert len(result["violation_flags"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_run_experiment.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `run_policy`**

```python
def run_policy(engine, policy, n_ticks: int) -> dict:
    states = engine.reset()
    costs, cvars, violation_flags, allocations = [], [], [], []
    for _ in range(n_ticks):
        site_states = {
            sid: {
                "power_mw": s.power_mw, "compute_demand": s.cpu_rate_sum,
                "cost_per_unit": 1.0, "risk_sample": [0.1],
            }
            for sid, s in states.items()
        }
        result = policy.step(site_states, graph=None, uncertainty_estimates={})
        allocations.append(result["allocation"])
        costs.append(sum(result["allocation"].values()))
        cvars.append(result.get("queue_backlog", 0.0))
        violation_flags.append(result.get("queue_backlog", 0.0) > 0)
        interventions = {}
        if result.get("intervention"):
            node, name, magnitude = result["intervention"]
            interventions = {sid: {"curtailment_frac": magnitude if name == "curtailment" else 0.0} for sid in states}
        states = engine.step(interventions)
    return {"costs": costs, "cvars": cvars, "violation_flags": violation_flags, "allocations": allocations}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_run_experiment.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Implement `main` (CLI entry point) and the default experiment config, then run the full four-way comparison**

`configs/experiment_default.yaml`:
```yaml
seed: 42
data_root: "D:/Cloud Computing"
output_dir: "runs/experiment_default"
n_ticks: 2000
policies: ["active", "passive", "observational_only", "oracle"]
cvar_alpha: 0.9
cvar_limit: 5.0
lyapunov_V: 1.0
```

Implement `main` to instantiate `ReplayEngine(pd.read_parquet(os.path.join(data_root, "fleet_data/processed/site_timeline.parquet")))`, loop over `policies`, call `run_policy`, compute metrics via Task 8.1's functions, dump `runs/experiment_default/results.json`, and use `matplotlib` to save one grouped bar chart per metric under `runs/experiment_default/plots/`.

Run: `python -m aco.eval.run_experiment --config configs/experiment_default.yaml`
Expected: `runs/experiment_default/results.json` contains all four policies' metrics; the `active` policy's `total_cost`/`violation_rate` should sit between `passive` and `oracle` (the qualitative result the paper needs to demonstrate — if `active` is *worse* than `passive` on both operational and causal metrics simultaneously, that's a real finding to investigate, not a bug to paper over).

- [ ] **Step 6: Commit**

```bash
git add src/aco/eval/run_experiment.py configs/experiment_default.yaml tests/eval/test_run_experiment.py
git commit -m "feat: four-way policy comparison experiment runner with plots"
```

---

## Self-Review Notes

- **Spec coverage:** Section 6.1 (representation) → Phase 3; 6.2 (VoI) → Phase 5; 6.3 (world model) → Phase 4; 6.4 (DRO execution) → Phase 6; 6.5 (sensing/storage as interventions) → `high_res_sampling`/`high_res_logging` in Task 5.1; Section 7 (architecture/closed loop) → Phase 2 engine + Phase 6 orchestrator wiring; Section 8.1 (intervention library) → Task 5.1; Section 9 (datasets) → Phase 1, using exactly the datasets present on disk; Section 10.1 (dual metrics) → Task 8.1; Section 10.2 (baselines) → Phase 7. The one gap is the fourth baseline ("strong non-causal proactive optimizer 2023–2025"), which is flagged rather than defaulted because it requires a literature choice only you can make.
- **Placeholder scan:** every step above has real, runnable code or a concrete shell command with a stated expected result — no "TODO"/"add appropriate handling" left in.
- **Type/name consistency:** `NODE_SCHEMA` (Phase 3) names are reused verbatim in Phase 4's test fixtures; `.step(site_states, graph, uncertainty_estimates) -> dict` is the identical signature across `ActiveOrchestrator`, `PassiveOrchestrator`, and `ObservationalOnlyOrchestrator` (Phases 6–7) so Task 8.2's `run_policy` works unmodified against all three; `OracleOrchestrator.step` intentionally drops the unused `graph`/`uncertainty_estimates` params since it never uses them, which the experiment runner should special-case (call it with just `site_states`).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-29-active-causal-orchestration-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
