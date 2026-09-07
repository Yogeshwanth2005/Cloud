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
| Google Cluster Trace 2011 | `google_cluster_2011/` raw + `processed/*.parquet` | Job/task lifecycle plus a derived 5-min per-machine CPU/mem utilization table | `task_usage.parquet` is ~1.1 GB — must be streamed, never loaded whole. Already de-duplicated by `preprocess_cluster_data.py`. **Only 18 of 500 shards are on disk** (`part-00200`–`part-00217`), so the trace spans ~25 hours (2011-05-13 09:20 → 2011-05-14 10:25), *not* the 29-day window the proposal assumes. `machine_utilization_5min.parquet` therefore has only 2 distinct cluster days — with the downstream consequence in [the audit register](docs/AUDIT_2026-09-04.md). |

### The alignment problem, and the simulation clock

No two datasets share calendar time: Integration Studies is all 2006, PVDAQ spans 1994–2023,
NSRDB is 2018–2023, and the Google trace *as published* is a 29-day window in May 2011 (only
~25 hours of it are on disk — see the dataset table). So **no join may
assume shared absolute dates**. Everything is aligned on a relative *simulation clock* — days
since each source's own epoch, plus `hour_of_day` — so the solar and cloud calendars meet only
through diurnal position. That is a deliberate modeling choice, not a bug. Concretely,
`site_timeline.parquet` carries **two** independent day-indices, `sim_day_solar` (spans the full
365-day solar year — this is what `ReplayEngine` ticks over) and `sim_day_cluster` (auxiliary
context, not a tick driver), not a single shared `sim_day` — a single index would have meant
cross-joining solar days onto cluster days that don't correspond to them.

**Known defect in the current `site_timeline.parquet` (audit finding [B1]):** because only 2
cluster days exist on disk, `sim_day_cluster = cluster_days[sim_day_solar % 2]` and the inner
merge on `(sim_day_cluster, hour_of_day)` silently drops every solar slot whose hour is absent
from the cluster day it drew. Even solar days keep 09:20–23:55 (176 slots, mean 28.7 MW); odd
days keep 00:00–10:25 (126 slots, mean 10.2 MW). **52.5% of the solar year survives**, and
available fleet power oscillates ~3× with day parity for a purely artifactual reason. Any
Phase 8 time series over this table carries a period-2 component that is the join, not physics
and not policy. Must be rebuilt before Phase 8 — see [B1] for the recommended fix.


### Data scope: two-tier physical/fleet architecture (Phase 1.5, added 2026-09-02)

A post-hoc audit of completed Phase 1 output found that `site_timeline.parquet` — what Phase 2
replays and Phase 8 evaluates against — has no path to the five physical variables Phase 3's
`NODE_SCHEMA` requires (`poa_irradiance`, `module_temp`, `ambient_temp`, `dc_power`, `ac_power`).
Those exist only in `pvdaq_data/processed/system_{50,51}_weather.parquet` — 2 real inverters at
one campus, on a different calendar than the 20-site fleet. A proposal to derive a synthetic
per-site `dc_power`/`ac_power`/`module_temp` from `power_mw` (already the plant's real output) was
considered and **rejected** as circular: causal discovery over a variable that's just a monotonic
reshaping of another number would recover near-deterministic edges from restating one quantity,
not from learning anything — defeating the actual research point.

**Decision — two tiers, kept structurally separate rather than forced into one shared schema:**

- **Tier 1 (physical causal calibration)** uses only real PVDAQ + NSRDB data — unchanged from how
  the plan already specifies `fit_observational_graph`, `CausalWorldModel`, Task 4.2's clipping
  validation, and Task 5.2's VoI-proxy check. Primary source: **`system_51`, 2015–2023**
  (corr(poa_irradiance, dc_power) = 0.961, n≈3.02M rows). `system_50` 2015–2023 (corr collapses to
  0.121 — real inverter failure) and both systems' pre-2011 windows (clock-glitch / commissioning
  noise) are excluded from calibration.
- **Tier 2 (fleet-scale orchestration)** uses only variables the 20 fleet sites actually have —
  no physical node is fabricated per site. `NODE_SCHEMA` gained `power_mw` as an 11th name (the
  fleet's modelled plant output, kept distinct from PVDAQ's measured `ac_power`, not an alias for
  it); fleet-side graph fits scope `var_names` to `["power_mw", "curtailment_frac",
  "sampling_rate_hz", "cpu_rate_sum", "cost", "risk"]`.
- **Correction (2026-09-04):** this section previously called the fleet sites' `power_mw` their
  "real metered output". It is not. NREL Solar Power Data for Integration Studies is **model
  output** — the proposal's own §9.1 calls them "simulated plants", as does the plan's Current
  State table. The Tier-2 tier is a simulation study and must be described as one.
- **The actual reason two tiers are needed is spatial diversity, not variable availability.**
  All five PVDAQ systems carry the physical variables and overlap in real calendar time, so a
  real multi-system panel would need no simulation clock at all. But cross-system POA
  irradiance correlation (2018, hourly) is 0.95–0.98 between systems 10, 4 and 51 — the same
  sky — with system_1283 at ≈0.8 and system_50 broken. Real data gives **two distinct skies,
  not twenty sites**, so fleet-scale ramp propagation and cross-site orchestration are not
  testable on it. This is a stronger justification than the variable-availability framing above
  and should replace it in the paper. See [audit Group F](docs/AUDIT_2026-09-04.md).
- **Tier 1 should be widened from 2 systems to 4.** Phase 1.5 picked `system_51` primary and
  `system_50` secondary — the two weakest. `system_10` (6.66M rows, corr 0.94–0.99 over fifteen
  years) and `system_4` are comparable-or-better and currently unused; the co-location evidence
  above suggests the NSRDB Golden weather join extends to them. Confirm against PVDAQ's
  published lat/lon metadata first. See [audit F1](docs/AUDIT_2026-09-04.md).
- **Paper implication:** the "is this causal edge real" discovery claim is scoped to where real
  sensor data exists (Tier 1, 2 systems); the fleet's contribution is testing whether VoI-guided
  orchestration generalizes across many sites using each site's modelled plant output. Must be
  disclosed explicitly, same obligation as the CloudSim and CVaR-vs-DRO simplifications.

Two downstream interface bugs surfaced by this same audit were fixed in place, not deferred:
`ReplayEngine` was reading a `sim_day` column that doesn't exist (see above); and a Task 3.1
worked example still referenced pre-canonicalization PVDAQ column names.

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

**Done — Phases 0 through 6 are written and tested, and the active-learning loop is closed.**
58 tests pass (`python -m pytest -q`). A full-project audit on 2026-09-04 verified this by
running the suite and reading every module rather than trusting the plan's checkboxes, which
remain unticked throughout and track nothing — git history is the real record.

All audit findings live in **[`docs/AUDIT_2026-09-04.md`](docs/AUDIT_2026-09-04.md)**, which is
the authoritative register; the summary below points into it rather than restating it.

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
| `src/aco/causal/graph.py` | `NODE_SCHEMA`; `fit_observational_graph` (PCMCI+, orientation-aware, keeps the most-significant lag per pair); `update_graph_with_intervention` (merges post-intervention-sharpened edges into a prior graph, and severs any edge the refit finds pointing INTO the intervened variable — Pearl's mutilated-graph identification, not just a second observational fit) |
| `src/aco/causal/world_model.py` | `CausalWorldModel` — one `GradientBoostingRegressor` per node on its Phase-3 graph parents; `.fit()` / `.predict()` / `.do()` (interventional prediction); `.estimate_uncertainty_reduction()` (simulates probing a node and refits to estimate the VoI criterion's expected information gain — see item 5) |
| `src/aco/causal/validate_world_model.py` | `label_clipping_events` (flags AC/DC saturation rows); `efficiency_by_power_bin` / `has_clipping_plateau` — upper-range efficiency diagnostic that detects a real inverter cap without the part-load confound |
| `src/aco/causal/run_clipping_validation.py` | Reproducible Task 4.2 driver; writes `runs/validation/world_model_clipping_report.json` |
| `src/aco/interventions/library.py` | `INTERVENTIONS` (curtailment, high-res sampling, setpoint change, high-res logging) with per-action cost and a pre-registered safety bound; `apply_intervention` |
| `src/aco/interventions/voi.py` | `score_intervention` / `select_best_intervention` — Value-of-Information-under-risk proxy (world-model-estimated uncertainty reduction vs. cost vs. risk) over Phase-3 graph edges |
| `src/aco/interventions/run_voi_proxy_check.py` | Reproducible Task 5.2 Step 5 driver; writes `runs/validation/voi_proxy_check.json` (see item 2) |
| `src/aco/optim/dro_allocator.py` | `solve_slot` — per-slot convex resource allocation (`cvxpy`) minimizing cost to serve compute demand within a power budget and a Rockafellar–Uryasev CVaR risk bound |
| `src/aco/optim/orchestrator.py` | `ActiveOrchestrator` — Lyapunov drift-plus-penalty wrapper closing the proposal's §8.4 loop. Each slot: records the new observation against the intervention in flight and, once its window is full, folds it back via `update_graph_with_intervention` and refits the world model; then selects and applies the next intervention — but only when none is in flight, so every window stays attributable to exactly one intervention; then solves the allocation via `solve_slot`. Contract: **one `.step()` == one new observation, taken as `df`'s last row**, counted in an orchestrator-owned buffer rather than by differencing `len(df)`, so windows still fill under a sliding or rebased frame. Returns `graph` and `causal_update` alongside the allocation |

**Processed artifacts on disk:**

- `fleet_data/processed/` — `plants.parquet`, `power_5min.parquet` (302 MB), `site_timeline.parquet`
- `pvdaq_data/processed/` — `system_4`, `system_10`, `system_50` (+ `_weather`), `system_51` (+ `_weather`), `system_1283`
- `nsrdb_golden/processed/nsrdb_golden.parquet`
- `google_cluster_2011/processed/` — 7 tables including `machine_utilization_5min.parquet`
- `runs/validation/world_model_clipping_report.json` — Task 4.2's real-data result: a null experiment (see item 1 below)
- `runs/validation/voi_proxy_check.json` — Task 5.2's real-data validation of the VoI proxy; now
  reproducible via `run_voi_proxy_check.py`, and regenerating it changed the finding (see item 2)

**Not started — Phases 7 and 8.** No `src/aco/baselines/` or `eval/` directories exist yet: the
baselines and the evaluation harness. The plan's step checkboxes remain unticked throughout
(tracked separately from actual completion — see the git history for what's really done).

### Open items worth attention

Full detail, evidence and reproduction steps: **[`docs/AUDIT_2026-09-04.md`](docs/AUDIT_2026-09-04.md)**.
IDs below are that register's.

**Fixed since the audit (2026-09-04):**

- **[A1-a]** The four-item intervention library collapsed to one: `select_best_intervention`
  pins `magnitude = max_magnitude/2`, making the risk term a constant 2.50 for all four, so the
  argmax was always the cheapest — `setpoint_change`, every time. Fixed by `target_var`.
- **[A1-b]** The probe node and the manipulated variable were decoupled, so a probe of
  `poa_irradiance` could be executed by changing `power_factor` — and incoming edges to
  irradiance were then severed on the strength of an intervention that never touched it. Fixed:
  each intervention declares `target_var`, a test asserts the declaration matches what `apply()`
  writes, and only compatible node/intervention pairs are considered.
- **§8.4's update leg**, previously missing entirely: the orchestrator intervened but never
  learned from the result.
- **`update_graph_with_intervention`'s `pre_df`** was accepted and never read. §8.4 defines the
  update as *prior graph + post-intervention evidence*, not a pre/post window comparison, so the
  parameter was removed rather than left implying a comparison the function does not perform.

**Open, in recommended order:**

| ID | Item |
|---|---|
| [B1] | `site_timeline.parquet` loses 47.5% of slots in an alternating day-parity pattern, with a ~3× artifactual power swing. Rebuild the cluster join first — everything downstream is measured on it. |
| [A1-c] | No channel from causal knowledge to allocation: `solve_slot` takes no graph. Until this exists, active ≡ passive ≡ observational_only ≡ oracle and the headline claim is untestable. §6.4's causal-uncertainty ambiguity set is the proposal's own answer. |
| [E1] | The intervention is live for one slot while its observation window runs the whole cycle, so the mutilated-graph severing is applied to a ~99%-unclamped window. Also gives §8.1's "limited-duration" a real parameter, and forces a duration-aware cost model. |
| [A2] | Four of six Tier-2 `var_names` don't exist in the fleet timeline; the two actuator columns are constant, so the graph can never see them. |
| [A3][A4] | Orchestrator/baseline signature drift, and the plan's Task 8.2 fixture still uses the removed `sim_day`. |
| [C1][C2] | CVaR runs on a single scenario; `V` collapses to zero allocation past `DEMAND_FULFILLMENT_VALUE / cost_per_unit`. |
| [C4] | No counterfactual method — and Task 4.2 defers the twin's only remaining validation to it. |
| [C3][C5][C6][C7] | Write-only Lyapunov queue; no FDR correction; `tau_max` 3-vs-1 mismatch; uncertainty signal is a cliff, not a gradient. |
| [B4][B5] | Negative sensor values survive cleaning (undocumented sentinels beyond `-99999`), and system_50's irradiance sensor is dead, not just its inverter. |
| [D2][D3] | `ReplayEngine` clamps instead of signalling exhaustion; branch hygiene. |

**Proposal conformance** is tracked as [audit Group E](docs/AUDIT_2026-09-04.md).
Of the five components §1 names as supporting machinery: VoI is substantially conformant,
representation and the world model are partial, and the causal-uncertainty ambiguity set and
sensing/storage treatment are largely absent. Two items need an explicit decision rather than
code: whether to build the **event/semantic layer** (§6.1) or rename the contribution, and
which **fourth baseline** (§10.2) to reimplement.

One thing that is *not* a deviation: §9.3's own last bullet sanctions a "custom simulation
layer", so the Python simulator standing in for CloudSim is within the proposal's stated
tooling. It still deserves a sentence in the experimental setup, but not the
limitations-section treatment this document previously gave it.

### Next step

**Phase 1.1 — [E1]:** hold the intervention across its observation window, and make `cost_fn`
duration-aware. Small, and it is the difference between "the loop runs" and "the loop learns
from interventional data".

Then the roadmap agreed on 2026-09-04: **Phase 2** connect causal uncertainty to the optimizer
([A1-c], resolving [C7] as part of its design) → **Phase 3** counterfactual reasoning ([C4]) →
**Phase 4** sensing/storage mechanism (§6.5, which also supplies the missing bandwidth/storage
metrics) → **Phase 5** per-site intervention subsets and a magnitude search → **Phase 6**
metrics and experiments (Phases 7–8 of the original plan).

[B1] is a data-engineering prerequisite that can proceed in parallel and must land before any
Phase 8 numbers are reported.
