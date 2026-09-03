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

**Done — Phases 0 through 5 are written and tested.** 35 tests pass (`python -m pytest -q`).

| Module | Purpose |
|---|---|
| `src/aco/config.py` | YAML config loader; raises if `seed` / `data_root` / `output_dir` are missing |
| `src/aco/data/pvdaq.py` | Sentinel to NaN, implausible-year filter, sensor-suffix canonicalization (averaging duplicate channels), restriction to 15 core columns, float32 downcast |
| `src/aco/data/fleet.py` | Filename regex to plant metadata; builds the `plants` and `power_5min` tables |
| `src/aco/data/nsrdb.py` | Skips the 2 metadata rows, rebuilds timestamps from Y/M/D/H/M, renames to snake_case |
| `src/aco/data/join_pvdaq_weather.py` | `merge_asof` nearest-hour join with a 30-minute tolerance |
| `src/aco/data/sim_clock.py` | `to_sim_clock` plus `build_site_timeline` — assigns each PV site a disjoint block of cluster machines and joins on hour-of-day |
| `src/aco/data/run_*_ingest.py` | Driver scripts that write the processed Parquet lake |
| `src/aco/sim/engine.py` | Tick-based `ReplayEngine` over `site_timeline.parquet` with a curtailment intervention hook |
| `src/aco/causal/graph.py` | `NODE_SCHEMA`; `fit_observational_graph` (PCMCI+, orientation-aware, keeps the most-significant lag per pair); `update_graph_with_intervention` (merges post-intervention-sharpened edges into a prior graph) |
| `src/aco/causal/world_model.py` | `CausalWorldModel` — one `GradientBoostingRegressor` per node on its Phase-3 graph parents; `.fit()` / `.predict()` / `.do()` (interventional prediction) |
| `src/aco/causal/validate_world_model.py` | `label_clipping_events` (flags AC/DC saturation rows); `efficiency_by_power_bin` / `has_clipping_plateau` — upper-range efficiency diagnostic that detects a real inverter cap without the part-load confound |
| `src/aco/causal/run_clipping_validation.py` | Reproducible Task 4.2 driver; writes `runs/validation/world_model_clipping_report.json` |
| `src/aco/interventions/library.py` | `INTERVENTIONS` (curtailment, high-res sampling, setpoint change, high-res logging) with per-action cost and a pre-registered safety bound; `apply_intervention` |
| `src/aco/interventions/voi.py` | `score_intervention` / `select_best_intervention` — Value-of-Information proxy (uncertainty reduction vs. cost) over Phase-3 graph edges |

**Processed artifacts on disk:**

- `fleet_data/processed/` — `plants.parquet`, `power_5min.parquet` (302 MB), `site_timeline.parquet`
- `pvdaq_data/processed/` — `system_4`, `system_10`, `system_50` (+ `_weather`), `system_51` (+ `_weather`), `system_1283`
- `nsrdb_golden/processed/nsrdb_golden.parquet`
- `google_cluster_2011/processed/` — 7 tables including `machine_utilization_5min.parquet`
- `runs/validation/world_model_clipping_report.json` — Task 4.2's real-data result: a null experiment (see item 1 below)
- `runs/validation/voi_proxy_check.json` — Task 5.2's real-data validation of the VoI proxy (see item 2 below)

**Not started — Phases 6 through 8.** No `src/aco/optim/`, `baselines/` or `eval/` directories
exist yet: the risk-constrained joint optimizer, the baselines, and the evaluation harness. The
plan's step checkboxes remain unticked throughout (tracked separately from actual completion —
see the git history for what's really done).

### Open items worth attention

1. **Task 4.2 is a null experiment: the natural experiment it depends on does not exist in
   PVDAQ.** The plan assumed inverter saturation (AC power plateauing while DC power keeps
   rising) would be observable in system_51 and would let the causal twin beat a naive linear
   baseline. It isn't there. `has_clipping_plateau` bins ac/dc efficiency across the upper half
   of the dc_power range and finds it rising monotonically (0.848 → 0.919) and flat at the top,
   with `ac_power` reaching 7,883 W — 45% *above* the 99th-percentile cut a nameplate rating
   would have to cap. All five PVDAQ systems were checked; none clip. Regenerate with
   `python -m aco.causal.run_clipping_validation`.

   **This is not a negative result about the world model.** The report also carries an
   extrapolation split (train below the cut, test above it) where the twin scores MAE 363 against
   linear's 44. That split measures extrapolation, not causal fidelity — its test set lies almost
   entirely outside the training range and a gradient-boosted model cannot predict past its
   largest leaf value, so it fails there by construction. On a random split over the same data the
   world model wins: **MAE 17.3 vs. linear's 21.5**. The twin's real validation is Phase 8's
   counterfactual prediction accuracy against the `ReplayEngine`, where ground truth exists by
   construction; that is a stronger test than the clipping experiment would have been, since it
   exercises the Phase-3 graph rather than a hand-specified `dc_power -> ac_power` edge. Disclose
   the absent natural experiment as a dataset limitation; do not disclose it as a model failure.
2. **The plan's Task 5.2 reference `score_intervention` formula doesn't clear any real
   intervention's cost at its own worked example's inputs** (`0.15 * 1 - 0.5 = -0.35`, yet the
   test it's meant to satisfy asserts `score > 0`). Fixed by adding an explicit, documented
   `INFO_VALUE_SCALE` conversion constant in `src/aco/interventions/voi.py` so uncertainty-reduction
   units and cost units are on a comparable scale — a legitimate VoI-to-cost exchange-rate knob,
   not a physical constant. The empirical check (Step 5) confirms the proxy correctly identifies
   the node whose causal edges improved most with more real data, with the caveat that the
   late-window graph shows some PCMCI+ orientation artifacts from stride-downsampling (same root
   cause as item 1's fitting approach — see `runs/validation/voi_proxy_check.json`).
3. **The fourth baseline is an open decision.** "Strong non-causal proactive optimizer
   (2023–2025)" was deliberately left unspecified in the plan because it needs a literature
   choice.
4. **Three documented simplifications/limitations to disclose in the paper:** the custom Python
   simulator in place of CloudSim, empirical-distribution CVaR in place of full Wasserstein-ball
   DRO, and item 1 above.

### Next step

Phase 6, Task 6.1 — per-slot convex resource allocation with a CVaR constraint
(`src/aco/optim/`), the Lyapunov drift-plus-penalty optimizer that consumes Task 5.2's VoI scores
as an extra penalty term each slot.
