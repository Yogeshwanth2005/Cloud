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
assume shared absolute dates**. Everything is aligned on a relative *simulation clock* — days
since each source's own epoch, plus `hour_of_day` — so the solar and cloud calendars meet only
through diurnal position. That is a deliberate modeling choice, not a bug. Concretely,
`site_timeline.parquet` carries **two** independent day-indices, `sim_day_solar` (spans the full
365-day solar year — this is what `ReplayEngine` ticks over) and `sim_day_cluster` (only ever
takes as many distinct values as the Google trace's 29-day window; auxiliary context, not a
tick driver), not a single shared `sim_day` — a single index would have meant cross-joining solar
days onto cluster days that don't correspond to them.

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
  fleet's real metered output, kept distinct from PVDAQ's measured `ac_power`, not an alias for
  it); fleet-side graph fits scope `var_names` to `["power_mw", "curtailment_frac",
  "sampling_rate_hz", "cpu_rate_sum", "cost", "risk"]`.
- **Paper implication:** the "is this causal edge real" discovery claim is scoped to where real
  sensor data exists (Tier 1, 2 systems); the fleet's contribution is testing whether VoI-guided
  orchestration generalizes across many sites using each site's real metered output. Must be
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

**Done — Phases 0 through 6 are written and tested.** 47 tests pass (`python -m pytest -q`).

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
| `src/aco/optim/orchestrator.py` | `ActiveOrchestrator` — Lyapunov drift-plus-penalty wrapper: each slot picks the best-scoring intervention via `select_best_intervention`, applies it, then solves the resulting allocation via `solve_slot`; tracks a virtual queue of CVaR-constraint backlog |

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
   not a physical constant.

   The empirical check (Step 5) previously existed only as an ad-hoc, uncommitted result with no
   way to regenerate it. It now has a real driver,
   `src/aco/interventions/run_voi_proxy_check.py` (`python -m aco.interventions.run_voi_proxy_check`),
   which also caught a genuine latent bug the ad-hoc version never exercised: `NODE_SCHEMA` names a
   single `module_temp` node, but `system_51_weather.parquet` only has `module_temp_1/2/3` —
   `CORE_COLUMNS` (`src/aco/data/pvdaq.py`) deliberately keeps them as three distinct real sensors
   rather than one canonical column. The driver reconciles this locally (averages the three, same
   spirit as `canonicalize_columns`'s duplicate-channel averaging) since nothing upstream of it had
   ever actually loaded `module_temp` from real data.

   Run against the real driver, **the proxy no longer agrees with the empirical best node** (it was
   previously reported as agreeing, from the uncommitted run) — `select_best_intervention` returns
   `None` entirely over the 3-day early window. Cause: with only 3 days of data the early graph has
   a single edge (`dc_power -> ac_power`); every other node is isolated, so `CausalWorldModel.fit()`
   never trains a regressor for it (`if not parents: continue`), and probing an isolated node
   propagates nowhere, giving `estimate_uncertainty_reduction` ≈ 0 for every candidate. This means
   the "narrow the early window to 3 days to avoid saturation" fix documented in the check's own
   `note_on_window_choice` is fragile in the other direction — too narrow starves the graph of
   structure entirely, and which specific 3 calendar days get picked evidently matters. This
   proxy-vs-empirical validation should be treated as unresolved, not confirmed, until a window size
   is found that is neither saturated (~90 days) nor structurally empty (~3 days). Same PCMCI+
   orientation-artifact caveat as before applies to the late-window graph (stride-downsampling; see
   `runs/validation/voi_proxy_check.json`).
3. **The fourth baseline is an open decision.** "Strong non-causal proactive optimizer
   (2023–2025)" was deliberately left unspecified in the plan because it needs a literature
   choice.
4. **Four documented simplifications/limitations to disclose in the paper:** the custom Python
   simulator in place of CloudSim, empirical-distribution CVaR in place of full Wasserstein-ball
   DRO, item 1 above, and item 7 below.
5. **A post-hoc audit found three places where Phases 3–5's code satisfied its own tests without
   satisfying the proposal claim its docstring made — all three are now fixed, with regression
   tests:**
   - `update_graph_with_intervention` accepted `intervened_var` but never used it in the body — it
     was just two observational PCMCI+ fits merged by whichever had the lower pval, which is not
     what gives interventional data stronger causal identification than observation alone (Section
     6.1). Fixed: any edge the post-intervention refit finds pointing INTO `intervened_var` is now
     discarded before merging (Pearl's "mutilated graph" — intervening on a variable severs its
     incoming edges for that window, so no such edge can be real causation regardless of its pval).
   - `score_intervention`'s `expected_uncertainty_reduction` was a caller-supplied float with no
     path from `CausalWorldModel` — Section 8.2 says the world model estimates this, not the
     caller. Fixed: `CausalWorldModel.estimate_uncertainty_reduction()` simulates a probe (node
     rescaled by `1 + magnitude` over a recent time-ordered window, propagated through `do()`,
     refit via `update_graph_with_intervention`) and `select_best_intervention` now calls it
     instead of accepting the value as an argument. The dead `current_uncertainty` parameter was
     also removed.
   - `score_intervention` had no risk term despite Section 6.2's title being "Value-of-Information
     **under Risk Constraints**." Fixed: `risk = (magnitude / max_magnitude) * RISK_SCALE` — the
     fraction of an intervention's own pre-registered safety bound a given magnitude consumes,
     netted against info value and cost. Kept separate from `cost_fn` because they measure
     different things (dollar cost vs. safety-margin consumption) that don't scale together across
     the four registered interventions.

   Two related gaps are *not* fixed and remain open: `CausalWorldModel` has no counterfactual
   (abduction–action–prediction) method, only interventional `do()`, despite Section 6.3 requiring
   both; and the safe intervention library covers sampling but not compression, retention, or
   replication (Section 6.5), and `sampling_rate_hz` is still write-only in `sim/engine.py` — it's
   recorded on `SiteState` but nothing reads it back to change an outcome.
6. **Task 6.1's plan reference objective didn't clear its own "prefers cheaper resource" test**
   — same category of gap as item 2. `minimize sum(cost_per_unit * x)` with only upper-bound and
   budget constraints has a trivial optimum of allocating nothing (cost 0), so the solver never
   actually served demand. Fixed the same way as item 2: an explicit, documented
   `DEMAND_FULFILLMENT_VALUE` constant in `src/aco/optim/dro_allocator.py` that dominates
   `cost_per_unit` when netted into the objective, so serving demand becomes the primary goal and
   cost minimization the tie-breaker among cheap vs. expensive resources. `solve_slot`'s returned
   `objective` field still reports the real, uninflated operational cost.
7. **The Lyapunov virtual queue in `ActiveOrchestrator` structurally can't register a soft
   violation.** `solve_slot` enforces the CVaR limit as a *hard* per-slot constraint, and
   allocating nothing always has CVaR = 0, so any non-negative `cvar_limit` is always trivially
   feasible — the post-hoc `violation = max(0, cvar - cvar_limit)` check in `orchestrator.py` is
   therefore ~0 every slot under normal operation, confirmed numerically (queue stays pinned at
   0.0 across repeated calls with `cvar_limit=0.0001`). The queue only moves at all when the limit
   is outright infeasible (e.g. negative), where it jumps straight to `inf` rather than
   accumulating gradually — verified with `cvar_limit=-1.0` in
   `tests/optim/test_orchestrator.py::test_orchestrator_queue_grows_after_repeated_violation`. A
   hard per-slot CVaR constraint and a backlog-tracking queue are in tension by construction; a
   real fix would relax the constraint to a soft, queue-weighted penalty term in `solve_slot`'s
   objective instead of a hard `cvxpy` constraint — out of scope for Task 6.2, flagged as future
   work (see item 4).

### Next step

Phase 7, Task 7.1 — the passive baseline (`src/aco/baselines/`), reusing `ReplayEngine` (Phase 2)
and `solve_slot` (Phase 6) so every baseline policy runs through the identical simulation loop.
