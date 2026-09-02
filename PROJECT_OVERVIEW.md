# Project Overview — Active Causal Orchestration for Distributed Solar Fleets

This document explains what this repository is, what the research idea behind it is, how the
codebase is meant to be structured, and how far the build has actually gotten.

Source documents it summarizes:

- `Active_Causal_Orchestration_Solar_Research_Proposal_Revised.docx` — the research proposal
- `docs/superpowers/plans/2026-08-29-active-causal-orchestration-implementation.md` — the 1,920-line implementation plan derived from it

---

## 1. What the project is

A research project that turns a proposal into a runnable experimental pipeline. The target
venues named in the proposal are IEEE Transactions on Smart Grid, IEEE IoT Journal, IEEE
Transactions on Cloud Computing, and TPDS.

The domain is **distributed photovoltaic (PV) fleets** — many solar plants, each streaming
irradiance, module temperature, inverter and weather telemetry into an edge–cloud
infrastructure that has to forecast output, respond to anomalies, curtail power, and provide
grid services.

## 2. The core argument

Existing edge–cloud orchestration for solar is **passive**. It observes streams, builds
correlational or feature-based models, and then allocates resources. It treats the causal
structure of the physical process — how irradiance drives DC power, how module temperature
suppresses it, how ramps propagate — as fixed background knowledge that it can only hope more
data will eventually clarify.

The proposal's single headline contribution is the shift to **Active Causal Orchestration**:
the orchestrator is allowed to *act* in order to learn. At each decision epoch it may choose a
safe, low-cost intervention on the physical fleet whose purpose is dual:

1. reduce uncertainty about a causal relationship it cares about, and
2. improve long-term operational performance under risk constraints.

So the system optimizes resource allocation *and the quality of its own causal understanding*
in the same closed loop.

## 3. The supporting machinery

Everything else in the proposal exists to make that paradigm concrete. Four pieces, each with
a distinct role:

| Role | Component | What it does |
|---|---|---|
| **Representation** | Active Causal Semantic Event Graph | A fleet-level causal graph refined by *both* observational streams and orchestrator-chosen interventions — interventional data gives stronger causal identification than observation alone. |
| **Decision criterion** | Value-of-Information (VoI) under risk constraints | Scores each candidate intervention: expected information gain against its immediate operational cost and risk. Only positive-net-value interventions are admitted. |
| **Evaluation environment** | Causal World Model | An interventional + counterfactual digital twin of the fleet *and* the edge–cloud infrastructure. Answers "what would have happened under a different allocation" and "what will happen if we apply this probe", so candidates are scored before execution. Updated after every real intervention. |
| **Execution mechanism** | Distributionally robust, risk-constrained optimizer | Produces the actual allocations and intervention decisions, robust to an ambiguity set over *residual* causal uncertainty, with performance bounds (causal regret / Lyapunov-style) that depend on the interventions the system chose. |

A fifth idea ties it together: **sensing and storage decisions are interventions too**. Sampling
rate, compression level, retention period, and replication all affect future causal model
quality, so the same VoI criterion governs them alongside ordinary resource allocation.

## 4. Safe intervention library

The interventions are deliberately modest and pre-registered with safety bounds — this is what
makes the idea defensible on real plants:

- temporary, limited-duration curtailment of a small subset of inverters
- short-term increase of sampling rate on selected sensors
- controlled reactive-power / power-factor set-point changes within grid-code limits
- selective activation of high-resolution logging on specific plants

## 5. Closed loop (proposal Section 7)

```
PV Fleet (multi-plant sensors + controllable inverters)
   |  observational streams          ^  safe interventions
   v                                 |
Edge Gateways - feature extraction | local causal updates | intervention execution
   v
Active Causal Semantic Event Graph                          [representation]
   v
Causal World Model (interventional + counterfactual twin)   [evaluation environment]
   v
Value-of-Information scoring of candidates                  [decision criterion]
   v
Distributionally Robust Joint Optimizer                     [execution mechanism]
   -> resource allocation + intervention selection + sensing/storage policy
   v
Edge / Cloud Execution -> Updated Causal Model -> Next Cycle
```

## 6. Evaluation design

Two metric families are reported side by side, which is the point — the claim is that paying a
short-term operational cost for information buys long-term operational gain.

**Operational:** economic cost, energy, bandwidth, storage footprint; CVaR of economic and
grid-stability risk; latency for high-severity events; constraint violation rate.

**Causal learning:** reduction in causal edge uncertainty over time; interventional /
counterfactual prediction accuracy; interventions executed against information gained; causal
regret relative to a genie with perfect causal knowledge.

**Baselines:** (1) the same system with interventions switched off — the critical one;
(2) strong non-causal proactive optimizers from 2023–2025; (3) observational-only causal graphs;
(4) an oracle with perfect causal knowledge as an upper bound.

---

## 7. The datasets (all public, all already on disk)

| Dataset | Location | Contents | Caveats that shaped the code |
|---|---|---|---|
| NREL Solar Power Data for Integration Studies | `Arizona/`, `California/`, `Colarado/`, `Nevada/` | 5-min actual power + 60-min day-ahead forecasts, ~2,200 CSVs, ~1,090 unique plants | Year 2006 only. Plant metadata (lat, lon, type, capacity) exists **only in the filename**, hence the regex parser. |
| NREL PVDAQ | `pvdaq_data/system_{4,10,50,51,1283}/year=/month=/day=/*.csv` | Real per-minute inverter + sensor telemetry: `ac_power`, `dc_power`, `poa_irradiance`, `module_temp_1..3`, `ambient_temp` | Missing values are the sentinel `-99999.0`, not `NaN`. Column names carry sensor-id suffixes (`ac_power__423`) that drift across years. Systems 50/51 contain bogus years (1822, 1994) from clock glitches. |
| NSRDB Golden | `nsrdb_golden/nsrdb_golden_{2018..2023}.csv` | GHI/DNI/DHI, temperature, wind, pressure, humidity at 39.73, −105.18 — the same NREL campus as the PVDAQ systems | Two metadata rows before the real header. Only overlaps PVDAQ's later years. |
| Google Cluster Trace 2011 | `google_cluster_2011/` raw + `processed/*.parquet` | Job/task lifecycle plus a derived 5-min per-machine CPU/mem utilization table | `task_usage.parquet` is ~1.1 GB — must be streamed, never loaded whole. Already de-duplicated by `preprocess_cluster_data.py`. |

### The alignment problem, and the simulation clock

No two datasets share calendar time: Integration Studies is all 2006, PVDAQ spans 1994–2023,
NSRDB is 2018–2023, and the Google trace is a single 29-day window in May 2011. So **no join may
assume shared absolute dates**. Everything is aligned on a relative *simulation clock* —
`sim_day` (days since each source's own epoch) plus `hour_of_day` — so the solar and cloud
calendars meet only through diurnal position. That is a deliberate modeling choice, not a bug.

---

## 8. Implementation plan structure

Nine phases. Phase 1 is fully specified at TDD step level because the data transformations are
unambiguous; Phases 3–7 fix the *interfaces and file layout* but leave the algorithm as a
research decision, each with a recommended default so work is never blocked.

| Phase | Scope | Key modules | Chosen default |
|---|---|---|---|
| 0 | Scaffolding, YAML config loader | `src/aco/config.py` | — |
| 1 | Data engineering — ingest, clean, join, build the fleet timeline | `src/aco/data/` | Parquet everywhere, sim-clock alignment |
| 2 | Digital twin — tick-based replay engine with an intervention API | `src/aco/sim/engine.py` | **Custom Python simulator, not CloudSim/iFogSim2.** Those are JVM-based and would mean re-serializing the fleet timeline across a process boundary every tick for no modeling benefit. Flagged for disclosure in the paper's experimental setup. |
| 3 | Active Causal Semantic Event Graph | `src/aco/causal/graph.py` | `tigramite` PCMCI+ for time-lagged discovery (alternative considered: NOTEARS) |
| 4 | Causal World Model | `src/aco/causal/world_model.py` | One `GradientBoostingRegressor` per node on its graph parents; `do()` by clamping a parent, counterfactuals by residual reuse (abduction–action–prediction) |
| 5 | Safe intervention library + VoI scoring | `src/aco/interventions/` | Uncertainty (p-value) reduction as the VoI proxy — exact Shannon gain over the graph posterior is intractable at this variable count |
| 6 | Risk-constrained joint optimizer | `src/aco/optim/` | Lyapunov drift-plus-penalty plus a CVaR constraint via `cvxpy` (Rockafellar–Uryasev), rather than full Wasserstein-ball DRO — documented simplification, full DRO listed as future work |
| 7 | Baselines | `src/aco/baselines/` | Passive, observational-only and oracle share the identical `.step()` signature so the experiment runner is policy-agnostic |
| 8 | Evaluation harness | `src/aco/eval/` | Dual metrics plus a runner writing `runs/<name>/results.json` and comparison plots |

**Tech stack:** Python 3.11, pandas + pyarrow, `tigramite`, `cvxpy`, `networkx`, `scikit-learn`,
`pytest`, `matplotlib`.

**Global constraints:** all derived tables are Parquet (never new CSVs); everything runs fully
offline from data already on disk; fixed seeds wherever there is stochasticity; every run is
driven by a YAML config saved alongside its results.

---

## 9. Current state of the build

**Done — Phase 0 and Phase 1 code is written and tested.** 16 tests pass (`python -m pytest -q`).

| Module | Purpose |
|---|---|
| `src/aco/config.py` | YAML config loader; raises if `seed` / `data_root` / `output_dir` are missing |
| `src/aco/data/pvdaq.py` | Sentinel to NaN, implausible-year filter, sensor-suffix canonicalization (averaging duplicate channels), restriction to 15 core columns, float32 downcast |
| `src/aco/data/fleet.py` | Filename regex to plant metadata; builds the `plants` and `power_5min` tables |
| `src/aco/data/nsrdb.py` | Skips the 2 metadata rows, rebuilds timestamps from Y/M/D/H/M, renames to snake_case |
| `src/aco/data/join_pvdaq_weather.py` | `merge_asof` nearest-hour join with a 30-minute tolerance |
| `src/aco/data/sim_clock.py` | `to_sim_clock` plus `build_site_timeline` — assigns each PV site a disjoint block of cluster machines and joins on hour-of-day |
| `src/aco/data/run_*_ingest.py` | Driver scripts that write the processed Parquet lake |

**Processed artifacts on disk:**

- `fleet_data/processed/` — `plants.parquet`, `power_5min.parquet` (302 MB), `site_timeline.parquet`
- `pvdaq_data/processed/` — `system_4`, `system_10`, `system_50`, `system_1283`
- `nsrdb_golden/processed/nsrdb_golden.parquet`
- `google_cluster_2011/processed/` — 7 tables including `machine_utilization_5min.parquet`

**Not started — Phases 2 through 8.** No `src/aco/sim/`, `causal/`, `interventions/`, `optim/`,
`baselines/` or `eval/` directories exist yet. That is every novel research component: the
replay engine, the causal graph, the world model, VoI selection, the DRO optimizer, the
baselines and the evaluation harness. The plan's 113 step checkboxes are all still unticked,
including the Phase 0/1 ones whose work is in fact complete.

### Open items worth attention

1. **`pvdaq_data/processed/system_51.parquet` is missing.** The other four systems ingested;
   system_51 did not. Its raw directory starts at `year=1994`, so it is the system most affected
   by the clock-glitch years. Re-run `python -m aco.data.run_pvdaq_ingest` and see what fails.
2. **`NLR_data.py` line 5 is currently `API_KEY` with no assignment** — a bare name expression
   that raises `NameError` as soon as the script runs. The hardcoded NREL key was removed, which
   was the right call, but no replacement was written. It should read from an environment
   variable, e.g. `API_KEY = os.environ["NREL_API_KEY"]`. The file is tracked in git, and the
   plan flagged rotating the previously-committed key at https://developer.nrel.gov before this
   repo is pushed anywhere — worth confirming that happened, since the old value is still in
   history.
3. **The fourth baseline is an open decision.** "Strong non-causal proactive optimizer
   (2023–2025)" was deliberately left unspecified in the plan because it needs a literature
   choice.
4. **Two documented simplifications to disclose in the paper:** the custom Python simulator in
   place of CloudSim, and empirical-distribution CVaR in place of full Wasserstein-ball DRO.

### Next step

Phase 2, Task 2.1 — the tick-based `ReplayEngine` over `site_timeline.parquet`. It is the
smallest piece that unblocks everything downstream, since every phase from 3 to 8 plugs into
its `.step(interventions) -> dict[str, SiteState]` loop.
