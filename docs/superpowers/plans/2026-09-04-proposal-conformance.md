# Proposal Conformance — Closing the Active Causal Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the running system implement what the proposal claims — a closed loop in which the orchestrator's interventions measurably reduce causal uncertainty, and reduced causal uncertainty measurably improves allocation.

**Architecture:** The existing pipeline already has all five components as separate pieces (graph, world model, VoI, CVaR optimizer, replay engine). What it lacks is the *couplings* between them that the proposal's Section 7 loop specifies. This plan builds those couplings: interventions that persist long enough to be attributable (§8.1), a causal-uncertainty measure that behaves like a gradient rather than a cliff (§8.4), that measure driving the optimizer's ambiguity set (§6.4), counterfactual queries (§6.3), and sensing/storage as priced decisions (§6.5). Two data-integrity tasks remove artifacts that would otherwise contaminate every measurement.

**Tech Stack:** Python 3.11, pandas + pyarrow, `tigramite` (PCMCI+), `cvxpy`, `networkx`, `scikit-learn`, `pytest`.

**Spec:** [Active_Causal_Orchestration_Solar_Research_Proposal_Revised.docx](../../../Active_Causal_Orchestration_Solar_Research_Proposal_Revised.docx)
**Findings register this plan discharges:** [docs/AUDIT_2026-09-04.md](../../AUDIT_2026-09-04.md)
**Predecessor plan (Phases 0–8, still authoritative for Phases 7–8):** [2026-08-29-active-causal-orchestration-implementation.md](2026-08-29-active-causal-orchestration-implementation.md)

## Global Constraints

- All derived tables are Parquet — never write a new derived CSV.
- Everything runs fully offline from data already on disk; no new downloads.
- Fixed random seeds wherever there is stochasticity; every experiment run is driven by a YAML config saved alongside its results.
- TDD throughout: write the failing test, watch it fail for the right reason, write the minimal implementation, watch it pass, commit. A test that passes the first time you run it proves nothing.
- No join may assume shared absolute dates across datasets — alignment is on the relative simulation clock (`sim_day_solar` + `hour_of_day`).
- Tier 1 (physical causal calibration) uses only real PVDAQ + NSRDB data. Tier 2 (fleet orchestration) uses only variables the fleet sites actually have. No physical variable is ever fabricated per fleet site.
- The current suite is **58 tests passing**. Every task ends with the full suite green, not just its own tests.

## Scope

This plan covers the mechanism gaps only. **Baselines (§10.2) and the evaluation harness (§10.1) stay in the predecessor plan's Phases 7–8** and should get their own plan once these couplings exist — building baselines now would mean writing four policies against interfaces this plan is about to change.

Two items are explicitly **out of scope and require a decision, not code**:
1. **The event/semantic layer (§6.1).** The contribution is named an Active Causal *Semantic Event* Graph; the implementation is a time-lagged causal graph over numeric telemetry. Build the layer or rename the contribution — this plan does neither.
2. **The fourth baseline (§10.2)**, "strong non-causal proactive optimizer (2023–2025)", which needs a literature choice.

## File Structure

| File | Responsibility |
|---|---|
| `src/aco/interventions/library.py` (modify) | Gains `max_duration_slots` per intervention — §8.1's "limited-duration" safety bound |
| `src/aco/interventions/voi.py` (modify) | Duration-aware cost and risk; rejects interventions that cannot be held long enough to yield an attributable window |
| `src/aco/optim/orchestrator.py` (modify) | Holds the intervention across its observation window; records uncertainty history; feeds causal risk scenarios to the optimizer |
| `src/aco/causal/uncertainty.py` (create) | `edge_uncertainty` / `node_uncertainty` over a *fixed* candidate-pair denominator, replacing the cliff-shaped `_avg_pval`; `UncertaintyHistory` for §10.1's over-time metric |
| `src/aco/optim/causal_risk.py` (create) | Derives the optimizer's risk scenarios from residual causal uncertainty — §6.4's ambiguity set |
| `src/aco/causal/world_model.py` (modify) | `counterfactual()` — abduction–action–prediction, §6.3 |
| `src/aco/sim/telemetry.py` (create) | Sampling/compression/retention/replication → bandwidth and storage footprint, §6.5 and §10.1 |
| `src/aco/data/pvdaq.py` (modify) | Physical-plausibility filter — undocumented sentinels beyond `-99999` |
| `src/aco/data/sim_clock.py` (modify) | `build_diurnal_cluster_profile` — removes the day-parity coverage artifact |

---

## Task 1: Interventions persist across their observation window (§8.1)

Proposal §8.1 specifies "temporary, **limited-duration** curtailment". The intervention is currently applied for exactly one slot while its observation window runs the whole cycle, so at `min_post_obs=120` roughly 1 row in 120 is actually clamped — and `update_graph_with_intervention` severs incoming edges on the strength of an intervention that was not in force for 99% of the window (audit E1).

**Files:**
- Modify: `src/aco/interventions/library.py`
- Modify: `src/aco/interventions/voi.py`
- Modify: `src/aco/optim/orchestrator.py`
- Test: `tests/interventions/test_library.py`, `tests/interventions/test_voi.py`, `tests/optim/test_orchestrator.py`

**Interfaces:**
- Consumes: `INTERVENTIONS[name]` with keys `apply`, `cost_fn`, `max_magnitude`, `target_var` (current shape).
- Produces: `INTERVENTIONS[name]["max_duration_slots"]: int` — pre-registered upper bound on how many consecutive slots an intervention may be held.
- Produces: `score_intervention(graph, node, expected_uncertainty_reduction, name, magnitude, duration_slots=1) -> float` — cost becomes `cost_fn(magnitude) * duration_slots`; risk becomes `(magnitude / max_magnitude) * (duration_slots / max_duration_slots) * RISK_SCALE`.
- Produces: `select_best_intervention(world_model, df, graph, node_candidates, var_names, tau_max=1, duration_slots=1)` — skips any intervention whose `max_duration_slots < duration_slots`.
- Produces: `ActiveOrchestrator.step(...)` result gains `"intervention_cost": float` (total cost of the held intervention, charged on the slot it starts).

**Design note to carry into the paper:** with `min_post_obs=120` (the window PCMCI+ needs to orient a lag-0 edge — measured, see audit E1) and `curtailment.max_duration_slots = 36`, curtailment becomes **inadmissible**: it cannot be held long enough to produce a fully-clamped window. That is a real, disclosable finding, not a bug — it says grid-affecting probes are too short to support PCMCI+-scale identification, leaving sampling and logging as the practical probes. Do not "fix" it by inflating the safety bound.

- [ ] **Step 1: Write the failing test for the duration bound**

In `tests/interventions/test_library.py`:

```python
def test_every_intervention_declares_a_duration_safety_bound():
    # Proposal Section 8.1: "temporary, limited-duration". An intervention with
    # no duration bound cannot be held for an attributable observation window
    # without leaving the safe envelope it was pre-registered under.
    for name, spec in INTERVENTIONS.items():
        assert isinstance(spec["max_duration_slots"], int)
        assert spec["max_duration_slots"] >= 1, name
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/interventions/test_library.py -q`
Expected: FAIL — `KeyError: 'max_duration_slots'`

- [ ] **Step 3: Add the bound**

In `src/aco/interventions/library.py`, add `"max_duration_slots"` to each entry. Grid-affecting actions get short bounds; passive observation changes get long ones. At 5-minute ticks, 36 slots = 3 hours, 288 slots = one day:

```python
INTERVENTIONS = {
    "curtailment": {
        "apply": _apply_curtailment, "cost_fn": lambda m: 5.0 * m,
        "max_magnitude": 0.3, "target_var": "power_mw", "max_duration_slots": 36,
    },
    "high_res_sampling": {
        "apply": _apply_sampling, "cost_fn": lambda m: 0.5 * m,
        "max_magnitude": 4.0, "target_var": "sampling_rate_hz", "max_duration_slots": 288,
    },
    "setpoint_change": {
        "apply": _apply_setpoint, "cost_fn": lambda m: 2.0 * m,
        "max_magnitude": 0.1, "target_var": "power_factor", "max_duration_slots": 36,
    },
    "high_res_logging": {
        "apply": _apply_logging, "cost_fn": lambda m: 0.2 * m,
        "max_magnitude": 9.0, "target_var": "logging_resolution_hz", "max_duration_slots": 288,
    },
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest tests/interventions/test_library.py -q`
Expected: PASS

- [ ] **Step 5: Write the failing tests for duration-aware scoring**

In `tests/interventions/test_voi.py`:

```python
def test_score_intervention_charges_cost_for_every_slot_held():
    # A curtailment held for 36 slots costs 36 times what one slot costs;
    # Section 6.2 requires that real cost to be weighed against information gain.
    graph = nx.DiGraph()
    graph.add_edge("power_mw", "cpu_rate_sum", pval=0.3)
    one = score_intervention(graph, "power_mw", 0.5, "curtailment", 0.15, duration_slots=1)
    ten = score_intervention(graph, "power_mw", 0.5, "curtailment", 0.15, duration_slots=10)
    spec = INTERVENTIONS["curtailment"]
    assert one - ten == pytest.approx(
        9 * spec["cost_fn"](0.15) + (0.15 / spec["max_magnitude"]) * RISK_SCALE * 9 / spec["max_duration_slots"]
    )


def test_select_best_intervention_rejects_an_intervention_it_cannot_hold_long_enough():
    # An intervention held for less than the full observation window leaves the
    # window partly unclamped, which is exactly what makes the mutilated-graph
    # severing in update_graph_with_intervention unsound.
    graph = nx.DiGraph()
    graph.add_edge("power_mw", "cpu_rate_sum", pval=0.3)
    too_long = INTERVENTIONS["curtailment"]["max_duration_slots"] + 1
    result = select_best_intervention(
        _HighGain(graph), None, graph,
        node_candidates=["power_mw"], var_names=["power_mw", "cpu_rate_sum"],
        duration_slots=too_long,
    )
    assert result is None
```

- [ ] **Step 6: Run them and watch them fail**

Run: `python -m pytest tests/interventions/test_voi.py -q`
Expected: FAIL — `TypeError: score_intervention() got an unexpected keyword argument 'duration_slots'`

- [ ] **Step 7: Make cost and risk duration-aware**

In `src/aco/interventions/voi.py`, replace the two function bodies:

```python
def score_intervention(graph, node, expected_uncertainty_reduction, name, magnitude,
                       duration_slots=1):
    spec = INTERVENTIONS[name]
    n_edges_touched = graph.degree(node) if node in graph else 0
    info_value = expected_uncertainty_reduction * max(n_edges_touched, 1) * INFO_VALUE_SCALE
    # Cost accrues every slot the intervention is held -- a curtailment held for
    # three hours is three hours of lost generation, not a one-off charge.
    cost = spec["cost_fn"](magnitude) * duration_slots
    # Risk consumes two pre-registered safety margins at once: how far the
    # magnitude goes toward its bound, and how long it is held toward its own.
    risk = (
        (magnitude / spec["max_magnitude"])
        * (duration_slots / spec["max_duration_slots"])
        * RISK_SCALE
    )
    return info_value - cost - risk


def select_best_intervention(world_model, df, graph, node_candidates, var_names,
                             tau_max=1, duration_slots=1):
    best = None
    best_score = 0.0
    for node in node_candidates:
        for name, spec in INTERVENTIONS.items():
            if spec["target_var"] != node:
                continue
            # An intervention that cannot be held for the whole observation
            # window would leave that window partly unclamped, so the graph
            # update could not treat it as interventional data at all.
            if duration_slots > spec["max_duration_slots"]:
                continue
            magnitude = spec["max_magnitude"] / 2
            reduction = world_model.estimate_uncertainty_reduction(
                df, node, magnitude, var_names, tau_max=tau_max,
            )
            score = score_intervention(
                graph, node, expected_uncertainty_reduction=reduction,
                name=name, magnitude=magnitude, duration_slots=duration_slots,
            )
            if score > best_score:
                best_score = score
                best = (node, name, magnitude)
    return best
```

Add `RISK_SCALE` to the test file's imports if it is not already there.

- [ ] **Step 8: Run them and watch them pass**

Run: `python -m pytest tests/interventions -q`
Expected: PASS (all interventions tests)

- [ ] **Step 9: Write the failing test for holding the intervention**

In `tests/optim/test_orchestrator.py`:

```python
def test_intervention_is_held_for_every_slot_of_its_observation_window():
    # Section 8.4's update treats the window as post-intervention data, and
    # update_graph_with_intervention severs incoming edges on the claim that
    # target_var was set rather than caused. That claim only holds while the
    # intervention is actually in force, so it must be re-applied every slot.
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=5)

    hist = pre
    curtailed = []
    for t in range(6):
        hist = pd.concat([hist, post.iloc[[t]]], ignore_index=True)
        states = {"s1": dict(site_states["s1"])}
        orch.step(states, model, hist, graph, node_candidates=["power_mw"], var_names=VARS)
        curtailed.append(states["s1"]["power_mw"] < site_states["s1"]["power_mw"])

    # Slot 0 starts it; slots 1-5 supply the window and must stay clamped.
    assert curtailed == [True] * 6


def test_intervention_cost_is_charged_for_the_whole_held_duration():
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=5)
    hist = pd.concat([pre, post.iloc[[0]]], ignore_index=True)

    first = orch.step(dict(site_states), model, hist, graph,
                      node_candidates=["power_mw"], var_names=VARS)

    _node, name, magnitude = first["intervention"]
    assert first["intervention_cost"] == pytest.approx(
        INTERVENTIONS[name]["cost_fn"](magnitude) * orch.min_post_obs
    )
```

Add `import pytest` and `INTERVENTIONS` to that test file's imports if absent.

- [ ] **Step 10: Run them and watch them fail**

Run: `python -m pytest tests/optim/test_orchestrator.py -q`
Expected: FAIL — the first on `curtailed == [True, False, False, False, False, False]`, the second on `KeyError: 'intervention_cost'`

- [ ] **Step 11: Hold the intervention in the orchestrator**

In `src/aco/optim/orchestrator.py`, `step()`: pass `duration_slots=self.min_post_obs` into selection, store the applied `(name, magnitude)` on `self._pending`, and re-apply it on every slot where an intervention is in flight.

```python
        best = None
        if self._pending is None:
            best = select_best_intervention(
                world_model, df, active_graph, node_candidates, var_names,
                tau_max=tau_max, duration_slots=self.min_post_obs,
            )
        intervention_result = None
        intervention_cost = 0.0
        if best is not None:
            node, name, magnitude = best
            for sid in site_ids:
                site_states[sid], unit_cost = apply_intervention(name, site_states[sid], magnitude)
            intervention_result = best
            # Charged once, on the slot it starts, for the whole duration it
            # will be held -- the orchestrator has already committed to it.
            intervention_cost = unit_cost * self.min_post_obs
            self._pending = {
                "node": node,
                "name": name,
                "magnitude": magnitude,
                "target_var": INTERVENTIONS[name]["target_var"],
                "post": [],
            }
        elif self._pending is not None:
            # Still in force: Section 8.1's "limited-duration" means held across
            # the window, not applied once. Without this the window is unclamped
            # and the graph update has no interventional data to work from.
            for sid in site_ids:
                site_states[sid], _unit_cost = apply_intervention(
                    self._pending["name"], site_states[sid], self._pending["magnitude"]
                )
            intervention_result = (
                self._pending["node"], self._pending["name"], self._pending["magnitude"]
            )
```

Add `"intervention_cost": intervention_cost` to the returned dict.

- [ ] **Step 12: Run them and watch them pass**

Run: `python -m pytest tests/optim/test_orchestrator.py -q`
Expected: PASS

- [ ] **Step 13: Run the whole suite**

Run: `python -m pytest -q`
Expected: PASS. `test_orchestrator_alternates_one_intervention_with_one_full_observation_window` asserts `intervention` is non-None only on cycle starts and will now see it non-None throughout the window — update that test's expected event sequence to `["intervene", "update", "intervene", "update", "intervene"]` counted on *cycle starts* by comparing against `result["causal_update"]`, or assert on `orch._pending is None` transitions instead. Fix the test to describe the new, correct behavior; do not weaken the new behavior to satisfy the old test.

- [ ] **Step 14: Commit**

```bash
git add src/aco/interventions/library.py src/aco/interventions/voi.py src/aco/optim/orchestrator.py tests/
git commit -m "feat(interventions): hold interventions across their observation window

Section 8.1 specifies temporary, limited-duration actions. The intervention
was applied for one slot while its observation window ran the whole cycle, so
at min_post_obs=120 roughly one row in 120 was actually clamped and
update_graph_with_intervention severed incoming edges on the strength of an
intervention not in force for 99% of the window.

Interventions now declare max_duration_slots, are re-applied every slot they
are held, and are rejected outright when they cannot be held for the full
window. Cost accrues per slot held and risk consumes both the magnitude and
duration safety margins."
```

---

## Task 2: Physical-plausibility filter for PVDAQ (audit B4/B5)

`-99999.0` is not the only sentinel. Measured on disk: `-7999`, `-5308.9`, `-50001.8`, `-53999`, leaving physically impossible negative irradiance in the processed lake — 11,180 rows in `system_51` alone, which feed the existing VoI and clipping results.

**Files:**
- Modify: `src/aco/data/pvdaq.py`
- Test: `tests/data/test_pvdaq.py`

**Interfaces:**
- Produces: `aco.data.pvdaq.PHYSICAL_RANGES: dict[str, tuple[float, float]]` — inclusive plausible range per canonical column.
- Produces: `aco.data.pvdaq.apply_physical_ranges(df: pd.DataFrame) -> pd.DataFrame` — values outside their column's range become `NaN`. Called from `clean_pvdaq_frame` after sentinel replacement.

- [ ] **Step 1: Write the failing test**

```python
def test_physically_impossible_values_become_nan():
    # -99999 is not the only sentinel in PVDAQ: -7999, -5308.9, -50001.8 and
    # -53999 all appear on disk. Filtering by physical range catches every
    # variant, including ones not yet seen, where an explicit sentinel list
    # would not.
    from aco.data.pvdaq import apply_physical_ranges

    df = pd.DataFrame({
        "measured_on": pd.to_datetime(["2018-01-01 12:00"] * 4),
        "poa_irradiance": [800.0, -7999.0, -50001.8, -1.0],
        "ambient_temp": [25.0, 25.0, -9999.0, 25.0],
    })
    out = apply_physical_ranges(df)

    assert out["poa_irradiance"].tolist()[0] == 800.0
    assert out["poa_irradiance"].isna().tolist() == [False, True, True, False]  # -1.0 is sensor offset, kept
    assert out["ambient_temp"].isna().tolist() == [False, False, True, False]


def test_clean_pvdaq_frame_applies_physical_ranges():
    df = pd.DataFrame({
        "measured_on": ["2018-01-01 12:00", "2018-01-01 12:01"],
        "poa_irradiance": [800.0, -7999.0],
    })
    out = clean_pvdaq_frame(df)
    assert out["poa_irradiance"].isna().tolist() == [False, True]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/data/test_pvdaq.py -q`
Expected: FAIL — `ImportError: cannot import name 'apply_physical_ranges'`

- [ ] **Step 3: Implement the filter**

In `src/aco/data/pvdaq.py`, after `CORE_COLUMNS`:

```python
# PVDAQ carries several undocumented sentinels besides -99999.0 (-7999,
# -5308.9, -50001.8 and -53999 all appear on disk), so filtering by explicit
# sentinel value misses variants. A physical plausibility range catches every
# one, including sentinels not yet observed. Lower bounds on irradiance and
# power are slightly negative on purpose: real pyranometers read a small
# negative offset at night, and inverters draw a little power in standby.
PHYSICAL_RANGES = {
    "poa_irradiance": (-5.0, 1500.0),
    "ambient_temp": (-60.0, 70.0),
    "module_temp_1": (-60.0, 110.0),
    "module_temp_2": (-60.0, 110.0),
    "module_temp_3": (-60.0, 110.0),
    "inverter_temp": (-60.0, 150.0),
    "das_temp": (-60.0, 150.0),
    "dc_power": (-100.0, 1e7),
    "ac_power": (-100.0, 1e7),
    "ac_voltage": (-10.0, 1000.0),
    "dc_pos_voltage": (-10.0, 2000.0),
    "power_factor": (-1.0, 1.0),
}


def apply_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """NaN out values outside each column's physically plausible range."""
    df = df.copy()
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df
```

In `clean_pvdaq_frame`, immediately after the sentinel-replacement loop:

```python
    df = apply_physical_ranges(df)
```

- [ ] **Step 4: Run them and watch them pass**

Run: `python -m pytest tests/data -q`
Expected: PASS

- [ ] **Step 5: Run the whole suite and commit**

```bash
python -m pytest -q
git add src/aco/data/pvdaq.py tests/data/test_pvdaq.py
git commit -m "fix(pvdaq): filter physically impossible sensor values

-99999.0 is not the only sentinel: -7999, -5308.9, -50001.8 and -53999 all
appear on disk, leaving impossible negative irradiance in the processed lake
(11,180 rows in system_51 alone, feeding the existing VoI and clipping
results). Filtering by physical range catches every variant."
```

- [ ] **Step 6: Regenerate the affected artifacts and record the change**

```bash
python -m aco.data.run_pvdaq_ingest
python -m aco.causal.run_clipping_validation
python -m aco.interventions.run_voi_proxy_check
```

Compare the regenerated `runs/validation/*.json` against the committed versions. If any headline number moved, say so explicitly in the commit message rather than silently overwriting — these are results the paper cites.

```bash
git add runs/validation/
git commit -m "chore: regenerate validation reports after the plausibility filter"
```

---

## Task 3: Causal uncertainty as a tracked, gradient-valued quantity (§8.4, audit C7)

`estimate_uncertainty_reduction`'s `_avg_pval` returns a default of `1.0` when no edge touches the node, so "maximum information gain" is a constant standing in for "the pre-fit found nothing". Measured: reduction `1.0000` at 300 rows, `0.0199` at 301, `0.0000` from 303 — a cliff, not a gradient. §8.4 also requires uncertainty estimates to be *maintained*, and §10.1's first causal metric is "reduction in causal edge uncertainty **over time**", which nothing currently records.

**Files:**
- Create: `src/aco/causal/uncertainty.py`
- Modify: `src/aco/causal/world_model.py`, `src/aco/optim/orchestrator.py`
- Test: `tests/causal/test_uncertainty.py`, `tests/optim/test_orchestrator.py`

**Interfaces:**
- Produces: `aco.causal.uncertainty.edge_uncertainty(graph, var_names: list[str]) -> float` — mean residual uncertainty over **every ordered variable pair**, present edges contributing their `pval` and absent pairs contributing `1.0`. The fixed denominator is the point: discovering one edge moves the measure by `1/n_pairs` instead of from `1.0` to `~0`.
- Produces: `aco.causal.uncertainty.node_uncertainty(graph, node: str, var_names: list[str]) -> float` — same, restricted to pairs touching `node`.
- Produces: `ActiveOrchestrator.uncertainty_history: list[dict]` — one `{"slot": int, "uncertainty": float}` per slot, feeding §10.1's over-time metric.

- [ ] **Step 1: Write the failing tests**

In `tests/causal/test_uncertainty.py`:

```python
import networkx as nx
import pytest

from aco.causal.uncertainty import edge_uncertainty, node_uncertainty

VARS = ["a", "b", "c"]


def test_empty_graph_is_maximally_uncertain():
    assert edge_uncertainty(nx.DiGraph(), VARS) == pytest.approx(1.0)


def test_discovering_one_edge_moves_uncertainty_by_one_pair_not_to_zero():
    # The cliff this replaces: the old measure jumped from 1.0 to ~0 on the
    # first discovered edge, because its denominator was the number of edges
    # found rather than the number of pairs that could exist.
    g = nx.DiGraph()
    g.add_edge("a", "b", pval=0.0)
    # 3 variables -> 6 ordered pairs; one is now certain, five are not.
    assert edge_uncertainty(g, VARS) == pytest.approx(5 / 6)


def test_a_weak_edge_counts_as_partly_uncertain():
    g = nx.DiGraph()
    g.add_edge("a", "b", pval=0.5)
    assert edge_uncertainty(g, VARS) == pytest.approx((0.5 + 5) / 6)


def test_node_uncertainty_covers_only_pairs_touching_that_node():
    g = nx.DiGraph()
    g.add_edge("a", "b", pval=0.0)
    # Pairs touching "a": a->b, a->c, b->a, c->a  -> one certain, three not.
    assert node_uncertainty(g, "a", VARS) == pytest.approx(3 / 4)
    # "c" touches four pairs, none discovered.
    assert node_uncertainty(g, "c", VARS) == pytest.approx(1.0)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/causal/test_uncertainty.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aco.causal.uncertainty'`

- [ ] **Step 3: Implement the measure**

Create `src/aco/causal/uncertainty.py`:

```python
"""Residual causal uncertainty over a discovered graph (proposal Sections 8.4
and 10.1).

Measured over the *candidate* edge set -- every ordered pair of variables --
rather than over the edges that happen to have been discovered. The fixed
denominator is what makes this a gradient: discovering one edge moves the
measure by 1/n_pairs, where a discovered-edges-only average jumps from its
no-edges default straight to near zero, which is what made the VoI signal a
cliff rather than something an optimizer could respond to.
"""

# An undiscovered pair carries no information either way, so it contributes
# the maximum. This is a modelling choice, not a p-value: "we have not
# established this edge" is treated as full residual uncertainty about it.
UNDISCOVERED_UNCERTAINTY = 1.0


def _pair_uncertainty(graph, u, v) -> float:
    if graph.has_edge(u, v):
        return float(graph[u][v].get("pval", UNDISCOVERED_UNCERTAINTY))
    return UNDISCOVERED_UNCERTAINTY


def edge_uncertainty(graph, var_names: list) -> float:
    """Mean residual uncertainty over every ordered pair of `var_names`."""
    pairs = [(u, v) for u in var_names for v in var_names if u != v]
    if not pairs:
        return UNDISCOVERED_UNCERTAINTY
    return sum(_pair_uncertainty(graph, u, v) for u, v in pairs) / len(pairs)


def node_uncertainty(graph, node: str, var_names: list) -> float:
    """Mean residual uncertainty over the ordered pairs touching `node`."""
    pairs = [(u, v) for u in var_names for v in var_names
             if u != v and node in (u, v)]
    if not pairs:
        return UNDISCOVERED_UNCERTAINTY
    return sum(_pair_uncertainty(graph, u, v) for u, v in pairs) / len(pairs)
```

- [ ] **Step 4: Run them and watch them pass**

Run: `python -m pytest tests/causal/test_uncertainty.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Switch `estimate_uncertainty_reduction` onto the new measure**

In `src/aco/causal/world_model.py`, delete the nested `_avg_pval` and replace the return:

```python
        from aco.causal.uncertainty import node_uncertainty

        return max(0.0, node_uncertainty(pre_graph, node, var_names)
                        - node_uncertainty(updated, node, var_names))
```

- [ ] **Step 6: Run the causal tests**

Run: `python -m pytest tests/causal tests/interventions -q`
Expected: PASS. If a VoI test that relied on the old cliff (a reduction of exactly `1.0`) now fails, update it to the new scale — the reduction is now bounded by the fraction of pairs the probe actually sharpened, which is the corrected behavior.

- [ ] **Step 7: Write the failing test for uncertainty history**

In `tests/optim/test_orchestrator.py`:

```python
def test_orchestrator_records_uncertainty_every_slot():
    # Section 10.1's first causal metric is "reduction in causal edge
    # uncertainty over time", which needs a per-slot series, not a final value.
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=5)

    _run_slots(orch, model, graph, site_states, pre, post, 7, FRAME_POLICIES["expanding"])

    assert [h["slot"] for h in orch.uncertainty_history] == list(range(7))
    assert all(0.0 <= h["uncertainty"] <= 1.0 for h in orch.uncertainty_history)
```

- [ ] **Step 8: Run it and watch it fail**

Run: `python -m pytest tests/optim/test_orchestrator.py::test_orchestrator_records_uncertainty_every_slot -q`
Expected: FAIL — `AttributeError: 'ActiveOrchestrator' object has no attribute 'uncertainty_history'`

- [ ] **Step 9: Record it**

In `ActiveOrchestrator.__init__`: `self.uncertainty_history = []`. In `step()`, after the graph update block and before selection:

```python
        self.uncertainty_history.append({
            "slot": len(self.uncertainty_history),
            "uncertainty": edge_uncertainty(active_graph, var_names),
        })
```

with `from aco.causal.uncertainty import edge_uncertainty` at the top.

- [ ] **Step 10: Run the whole suite and commit**

```bash
python -m pytest -q
git add src/aco/causal/uncertainty.py src/aco/causal/world_model.py src/aco/optim/orchestrator.py tests/
git commit -m "feat(causal): measure residual uncertainty over the candidate edge set

_avg_pval returned a default of 1.0 when no edge touched the node, so maximum
information gain was a constant standing in for 'the pre-fit found nothing' --
measured as a cliff: 1.0000 at 300 rows, 0.0199 at 301, 0.0000 from 303 on.
Measuring over every ordered variable pair gives a fixed denominator, so
discovering an edge moves the measure by 1/n_pairs. Also records a per-slot
uncertainty series for Section 10.1's over-time metric."
```

---

## Task 4: Causal uncertainty drives the optimizer's ambiguity set (§6.4, audit A1-c)

This is the coupling the whole proposal turns on. §6.4 specifies an optimizer "distributionally robust with respect to an ambiguity set **over residual causal uncertainty**". `solve_slot` currently takes caller-supplied exogenous `risk_samples` with `n_samples=1`, so nothing connects risk to the causal model — and with no such connection, active, passive, observational-only and oracle produce identical allocations and §11's headline claim cannot be tested.

**Files:**
- Create: `src/aco/optim/causal_risk.py`
- Modify: `src/aco/optim/orchestrator.py`
- Test: `tests/optim/test_causal_risk.py`, `tests/optim/test_orchestrator.py`

**Interfaces:**
- Consumes: `aco.causal.uncertainty.edge_uncertainty` (Task 3).
- Produces: `aco.optim.causal_risk.causal_risk_scenarios(graph, var_names, base_risk: list[float], n_scenarios: int = 32, seed: int = 0) -> list[list[float]]` — `n_scenarios` risk vectors centred on `base_risk` whose dispersion is proportional to `edge_uncertainty(graph, var_names)`. A perfectly known graph gives zero dispersion (every scenario equals `base_risk`); a maximally uncertain graph gives dispersion `AMBIGUITY_SCALE * base_risk`.
- Produces: `AMBIGUITY_SCALE: float = 1.0` — documented tuning knob converting residual causal uncertainty into risk dispersion, same status as `INFO_VALUE_SCALE`.

- [ ] **Step 1: Write the failing tests**

In `tests/optim/test_causal_risk.py`:

```python
import networkx as nx
import numpy as np
import pytest

from aco.optim.causal_risk import causal_risk_scenarios
from aco.optim.dro_allocator import solve_slot

VARS = ["a", "b"]


def _known_graph():
    g = nx.DiGraph()
    g.add_edge("a", "b", pval=0.0)
    g.add_edge("b", "a", pval=0.0)
    return g


def test_a_fully_known_graph_produces_no_risk_dispersion():
    scenarios = causal_risk_scenarios(_known_graph(), VARS, base_risk=[0.1, 0.2], n_scenarios=16)
    assert len(scenarios) == 16
    assert all(s == pytest.approx([0.1, 0.2]) for s in scenarios)


def test_an_unknown_graph_produces_dispersed_risk():
    scenarios = causal_risk_scenarios(nx.DiGraph(), VARS, base_risk=[0.1, 0.2], n_scenarios=64)
    spread = np.array(scenarios).std(axis=0)
    assert (spread > 0).all()


def test_less_causal_uncertainty_means_tighter_risk():
    # This is the mechanism Section 6.4 asks for and Section 11's headline
    # claim depends on: interventions reduce causal uncertainty, which tightens
    # the ambiguity set, which lets the optimizer serve more demand.
    partly = nx.DiGraph()
    partly.add_edge("a", "b", pval=0.0)
    wide = np.array(causal_risk_scenarios(nx.DiGraph(), VARS, [0.1, 0.1], n_scenarios=64)).std()
    tight = np.array(causal_risk_scenarios(partly, VARS, [0.1, 0.1], n_scenarios=64)).std()
    assert tight < wide


def test_a_better_causal_model_lets_the_optimizer_serve_more_demand():
    # The end-to-end property the paper claims. Same demand, same budget, same
    # CVaR limit -- only the causal model differs.
    def served(graph):
        scenarios = causal_risk_scenarios(graph, VARS, base_risk=[0.1, 0.1], n_scenarios=64)
        return sum(solve_slot(10.0, [6.0, 6.0], [1.0, 1.0], scenarios, 0.9, 0.6)["allocation"])

    assert served(_known_graph()) > served(nx.DiGraph())
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/optim/test_causal_risk.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aco.optim.causal_risk'`

- [ ] **Step 3: Implement the ambiguity set**

Create `src/aco/optim/causal_risk.py`:

```python
"""The optimizer's ambiguity set, derived from residual causal uncertainty
(proposal Section 6.4).

Section 6.4 asks for an optimizer "distributionally robust with respect to an
ambiguity set over residual causal uncertainty", with bounds that "explicitly
depend on the quality of the causal model and on the interventions the system
has chosen". This module is that dependence: the less the orchestrator knows
about the causal structure, the wider the risk distribution the CVaR
constraint must hold against, so a poorly-understood fleet is allocated
conservatively and a well-understood one is not.

That is also the mechanism behind the proposal's headline claim -- an
intervention that reduces causal uncertainty tightens this set, which relaxes
the CVaR constraint, which lets the optimizer serve more demand. Without it,
better causal knowledge has no path to better operational performance.
"""
import numpy as np

from aco.causal.uncertainty import edge_uncertainty

# Converts residual causal uncertainty (0 = fully known, 1 = nothing known)
# into risk dispersion as a fraction of the base risk level. A research/tuning
# knob, not a physical constant -- same status as INFO_VALUE_SCALE and
# RISK_SCALE in aco.interventions.voi.
AMBIGUITY_SCALE = 1.0


def causal_risk_scenarios(graph, var_names: list, base_risk: list,
                          n_scenarios: int = 32, seed: int = 0) -> list:
    """Risk scenarios whose spread is proportional to residual causal uncertainty.

    Returns `n_scenarios` vectors, each the same length as `base_risk`. With a
    fully determined graph every scenario equals `base_risk` exactly, which
    collapses CVaR to the nominal risk; with nothing known the spread reaches
    `AMBIGUITY_SCALE * base_risk`.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(base_risk, dtype=float)
    uncertainty = edge_uncertainty(graph, var_names)
    scale = AMBIGUITY_SCALE * uncertainty * np.abs(base)
    if not np.any(scale > 0):
        return [base.tolist() for _ in range(n_scenarios)]
    draws = rng.normal(loc=base, scale=scale, size=(n_scenarios, base.size))
    # Risk is a magnitude; a negative realization is not meaningful.
    return np.clip(draws, 0.0, None).tolist()
```

- [ ] **Step 4: Run them and watch them pass**

Run: `python -m pytest tests/optim/test_causal_risk.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the failing test for the orchestrator wiring**

In `tests/optim/test_orchestrator.py`:

```python
def test_orchestrator_builds_its_risk_scenarios_from_the_causal_graph():
    # Previously the orchestrator passed a single scenario, which collapsed
    # Rockafellar-Uryasev CVaR to that scenario's own value and left the
    # causal model with no influence on allocation at all.
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0,
                              min_post_obs=5, n_risk_scenarios=32)
    hist = pd.concat([pre, post.iloc[[0]]], ignore_index=True)

    result = orch.step(dict(site_states), model, hist, graph,
                       node_candidates=["power_mw"], var_names=VARS)

    assert result["n_risk_scenarios"] == 32
    assert result["causal_uncertainty"] == pytest.approx(
        edge_uncertainty(graph, VARS)
    )
```

Add `from aco.causal.uncertainty import edge_uncertainty` to that test file.

- [ ] **Step 6: Run it and watch it fail**

Run: `python -m pytest tests/optim/test_orchestrator.py::test_orchestrator_builds_its_risk_scenarios_from_the_causal_graph -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'n_risk_scenarios'`

- [ ] **Step 7: Wire it in**

In `ActiveOrchestrator.__init__`, add `n_risk_scenarios: int = 32` and `self.n_risk_scenarios = n_risk_scenarios`. In `step()`, replace the single-scenario construction:

```python
        base_risk = [site_states[sid]["risk_sample"][0] for sid in site_ids]
        risk_samples = causal_risk_scenarios(
            active_graph, var_names, base_risk,
            n_scenarios=self.n_risk_scenarios, seed=len(self.uncertainty_history),
        )
```

with `from aco.optim.causal_risk import causal_risk_scenarios` at the top. Add to the returned dict:

```python
            "causal_uncertainty": edge_uncertainty(active_graph, var_names),
            "n_risk_scenarios": len(risk_samples),
```

- [ ] **Step 8: Run the whole suite and commit**

Run: `python -m pytest -q`
Expected: PASS

```bash
git add src/aco/optim/causal_risk.py src/aco/optim/orchestrator.py tests/
git commit -m "feat(optim): derive the ambiguity set from residual causal uncertainty

Section 6.4 specifies an optimizer distributionally robust with respect to an
ambiguity set over residual causal uncertainty. solve_slot took caller-supplied
exogenous risk samples with n_samples=1, so CVaR collapsed to that scenario's
own value and no causal knowledge could reach the allocation -- which is why
all four baselines produced identical allocations.

Risk scenarios are now generated with dispersion proportional to
edge_uncertainty, so an intervention that sharpens the graph tightens the
ambiguity set and lets the optimizer serve more demand. This is the mechanism
Section 11's headline claim depends on."
```

---

## Task 5: Counterfactual queries (§6.3, audit C4)

§6.3 requires the twin to answer "what would have happened under a different allocation" alongside interventional queries. Only `do()` exists. This also unblocks §10.1's second causal metric and Task 4.2's deferred validation — with the clipping natural experiment confirmed absent, the twin currently has no completed causal-fidelity check.

**Files:**
- Modify: `src/aco/causal/world_model.py`
- Test: `tests/causal/test_world_model.py`

**Interfaces:**
- Produces: `CausalWorldModel.counterfactual(df: pd.DataFrame, interventions: dict) -> pd.DataFrame` — abduction–action–prediction. Abduction records each fitted node's residual on the observed rows; action clamps the intervened variables; prediction re-propagates in topological order, adding each node's own recorded residual back. Unlike `do()`, this preserves the specific noise realization of the observed rows, which is what makes it a counterfactual about *those* rows rather than a fresh interventional sample.

- [ ] **Step 1: Write the failing tests**

```python
def test_counterfactual_reproduces_observed_data_under_the_observed_value():
    # The consistency axiom: the counterfactual "what if X had been x" where x
    # is the value X actually took must return exactly what was observed. This
    # is what distinguishes abduction-action-prediction from do(), which
    # discards the observed noise and returns the model's mean response.
    rng = np.random.default_rng(0)
    n = 200
    irradiance = rng.normal(500, 100, n)
    df = pd.DataFrame({
        "poa_irradiance": irradiance,
        "dc_power": 0.6 * irradiance + rng.normal(0, 30, n),
    })
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.001)
    model = CausalWorldModel(graph)
    model.fit(df)

    cf = model.counterfactual(df, {"poa_irradiance": df["poa_irradiance"].to_numpy()})

    np.testing.assert_allclose(cf["dc_power"], df["dc_power"], rtol=1e-9)


def test_counterfactual_differs_from_do_by_preserving_observed_noise():
    rng = np.random.default_rng(1)
    n = 200
    irradiance = rng.normal(500, 100, n)
    df = pd.DataFrame({
        "poa_irradiance": irradiance,
        "dc_power": 0.6 * irradiance + rng.normal(0, 30, n),
    })
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.001)
    model = CausalWorldModel(graph)
    model.fit(df)

    halved = df["poa_irradiance"].to_numpy() * 0.5
    cf = model.counterfactual(df, {"poa_irradiance": halved})
    interventional = model.do(df, {"poa_irradiance": halved})

    # Both respond to the intervention...
    assert cf["dc_power"].mean() < df["dc_power"].mean()
    # ...but only the counterfactual carries the observed rows' own residuals.
    assert not np.allclose(cf["dc_power"], interventional["dc_power"])
    residual = df["dc_power"].to_numpy() - interventional["dc_power"].to_numpy()
    np.testing.assert_allclose(
        cf["dc_power"].to_numpy() - interventional["dc_power"].to_numpy(),
        residual - (residual - (cf["dc_power"].to_numpy() - interventional["dc_power"].to_numpy())),
        rtol=1e-9,
    )
```

For the second test's final assertion, prefer this simpler equivalent — it states the property directly:

```python
    # counterfactual == interventional prediction + the node's observed residual
    expected = interventional["dc_power"].to_numpy() + (
        df["dc_power"].to_numpy() - model.predict(df)["dc_power"].to_numpy()
    )
    np.testing.assert_allclose(cf["dc_power"], expected, rtol=1e-9)
```

Use the simpler form and delete the convoluted one.

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/causal/test_world_model.py -q`
Expected: FAIL — `AttributeError: 'CausalWorldModel' object has no attribute 'counterfactual'`

- [ ] **Step 3: Implement abduction–action–prediction**

In `src/aco/causal/world_model.py`:

```python
    def counterfactual(self, df: pd.DataFrame, interventions: dict) -> pd.DataFrame:
        """Answer "what would have happened to *these rows* under a different
        assignment" (proposal Section 6.3).

        Three stages, following Pearl. Abduction: recover each fitted node's
        residual on the observed data -- the part of its value its parents do
        not explain, which is this row's own noise realization. Action: clamp
        the intervened variables. Prediction: re-propagate in topological
        order, adding each node's recorded residual back to the model's
        response.

        The difference from `do()` is that third step. `do()` returns the
        model's mean response to the intervention and discards which specific
        rows were observed; `counterfactual` keeps them, so asking for the
        value a variable actually took returns exactly the observed data.
        """
        residuals = {}
        for node in nx.topological_sort(self.graph):
            if node not in self.models:
                continue
            parents, reg = self.models[node]
            residuals[node] = df[node].to_numpy() - reg.predict(df[parents])

        out = df.copy()
        for var, value in interventions.items():
            out[var] = value
        for node in nx.topological_sort(self.graph):
            if node in interventions or node not in self.models:
                continue
            parents, reg = self.models[node]
            out[node] = reg.predict(out[parents]) + residuals[node]
        return out
```

- [ ] **Step 4: Run them and watch them pass**

Run: `python -m pytest tests/causal/test_world_model.py -q`
Expected: PASS

- [ ] **Step 5: Run the whole suite and commit**

```bash
python -m pytest -q
git add src/aco/causal/world_model.py tests/causal/test_world_model.py
git commit -m "feat(world-model): counterfactual queries via abduction-action-prediction

Section 6.3 requires the twin to answer 'what would have happened under a
different allocation' alongside interventional queries; only do() existed.
Also unblocks Section 10.1's counterfactual-accuracy metric and Task 4.2's
deferred validation, which is the twin's only remaining path to a causal
fidelity claim now that the clipping natural experiment is confirmed absent."
```

---

## Task 6: Sensing and storage as priced decisions (§6.5, §10.1)

§6.5 names four knobs — sampling rate, compression level, retention period, replication — governed by the same VoI criterion. Two exist as interventions and neither has any effect: `sampling_rate_hz` is written to `SiteState` and never read. Meanwhile §10.1's operational metrics require bandwidth, storage footprint and energy, none of which have any source. One chain fixes both.

**Files:**
- Create: `src/aco/sim/telemetry.py`
- Modify: `src/aco/interventions/library.py`
- Test: `tests/sim/test_telemetry.py`, `tests/interventions/test_library.py`

**Interfaces:**
- Produces: `aco.sim.telemetry.telemetry_footprint(sampling_rate_hz: float, n_sensors: int = 8, bytes_per_sample: int = 4, compression_ratio: float = 1.0, retention_slots: int = 288, replication_factor: int = 1, slot_seconds: float = 300.0) -> dict` — returns `{"bandwidth_bytes_per_s": float, "storage_bytes": float, "samples_per_slot": float}`.
- Produces: `INTERVENTIONS` gains `"compression_change"`, `"retention_change"`, `"replication_change"` with `target_var` `"compression_ratio"`, `"retention_slots"`, `"replication_factor"` respectively, completing §6.5's four knobs.

- [ ] **Step 1: Write the failing tests**

In `tests/sim/test_telemetry.py`:

```python
import pytest

from aco.sim.telemetry import telemetry_footprint


def test_doubling_the_sampling_rate_doubles_bandwidth_and_storage():
    # This is what makes sampling_rate_hz a real decision rather than a value
    # written to SiteState and never read: raising it buys resolution and
    # costs bandwidth and storage, which Section 6.2 must weigh against the
    # information gained.
    low = telemetry_footprint(sampling_rate_hz=1.0)
    high = telemetry_footprint(sampling_rate_hz=2.0)
    assert high["bandwidth_bytes_per_s"] == pytest.approx(2 * low["bandwidth_bytes_per_s"])
    assert high["storage_bytes"] == pytest.approx(2 * low["storage_bytes"])


def test_compression_reduces_both_bandwidth_and_storage():
    plain = telemetry_footprint(sampling_rate_hz=1.0, compression_ratio=1.0)
    squeezed = telemetry_footprint(sampling_rate_hz=1.0, compression_ratio=4.0)
    assert squeezed["storage_bytes"] == pytest.approx(plain["storage_bytes"] / 4)
    assert squeezed["bandwidth_bytes_per_s"] == pytest.approx(plain["bandwidth_bytes_per_s"] / 4)


def test_retention_and_replication_affect_storage_but_not_bandwidth():
    base = telemetry_footprint(sampling_rate_hz=1.0, retention_slots=288, replication_factor=1)
    kept = telemetry_footprint(sampling_rate_hz=1.0, retention_slots=576, replication_factor=1)
    mirrored = telemetry_footprint(sampling_rate_hz=1.0, retention_slots=288, replication_factor=3)

    assert kept["storage_bytes"] == pytest.approx(2 * base["storage_bytes"])
    assert mirrored["storage_bytes"] == pytest.approx(3 * base["storage_bytes"])
    for other in (kept, mirrored):
        assert other["bandwidth_bytes_per_s"] == pytest.approx(base["bandwidth_bytes_per_s"])
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/sim/test_telemetry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aco.sim.telemetry'`

- [ ] **Step 3: Implement the footprint model**

Create `src/aco/sim/telemetry.py`:

```python
"""Sensing and storage footprint (proposal Sections 6.5 and 10.1).

Section 6.5 treats sampling rate, compression level, retention period and
replication as interventions inside the same optimization loop. That only has
teeth if those settings cost something, so this module is the chain that
prices them:

    sampling rate -> samples generated -> bandwidth -> storage -> cost

It is also the source for three of Section 10.1's operational metrics
(bandwidth, storage footprint, and the energy that follows from them), which
otherwise have no origin anywhere in the pipeline.
"""

# A PV site's standard instrument set: irradiance, three module temperatures,
# ambient temperature, DC power, AC power, inverter temperature.
DEFAULT_N_SENSORS = 8

# One 32-bit float per sensor reading.
DEFAULT_BYTES_PER_SAMPLE = 4

# One day of 5-minute slots.
DEFAULT_RETENTION_SLOTS = 288


def telemetry_footprint(
    sampling_rate_hz: float,
    n_sensors: int = DEFAULT_N_SENSORS,
    bytes_per_sample: int = DEFAULT_BYTES_PER_SAMPLE,
    compression_ratio: float = 1.0,
    retention_slots: int = DEFAULT_RETENTION_SLOTS,
    replication_factor: int = 1,
    slot_seconds: float = 300.0,
) -> dict:
    """Bandwidth and storage implied by a site's sensing/storage settings.

    `compression_ratio` is the factor by which compression shrinks the stream
    (1.0 = uncompressed), so it divides both bandwidth and storage.
    `retention_slots` and `replication_factor` multiply stored volume only --
    keeping data longer or in more copies costs storage, not link capacity.
    """
    if compression_ratio <= 0:
        raise ValueError(f"compression_ratio must be positive, got {compression_ratio}")

    bytes_per_second = sampling_rate_hz * n_sensors * bytes_per_sample / compression_ratio
    samples_per_slot = sampling_rate_hz * slot_seconds
    bytes_per_slot = bytes_per_second * slot_seconds
    return {
        "bandwidth_bytes_per_s": bytes_per_second,
        "storage_bytes": bytes_per_slot * retention_slots * replication_factor,
        "samples_per_slot": samples_per_slot,
    }
```

- [ ] **Step 4: Run them and watch them pass**

Run: `python -m pytest tests/sim/test_telemetry.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the failing test for the remaining two §6.5 knobs**

In `tests/interventions/test_library.py`:

```python
def test_library_covers_all_four_sensing_and_storage_knobs():
    # Section 6.5: "sampling rate, compression level, retention period, and
    # replication ... treated as interventions".
    targets = {spec["target_var"] for spec in INTERVENTIONS.values()}
    assert {"sampling_rate_hz", "compression_ratio",
            "retention_slots", "replication_factor"} <= targets
```

- [ ] **Step 6: Run it and watch it fail**

Run: `python -m pytest tests/interventions/test_library.py -q`
Expected: FAIL — the assertion, with `compression_ratio`, `retention_slots`, `replication_factor` missing

- [ ] **Step 7: Add the three interventions**

In `src/aco/interventions/library.py`, add the apply functions and entries. Note `test_every_intervention_declares_the_state_variable_it_manipulates` requires each `apply` to write exactly its declared `target_var`, and the baseline state in that test must gain the three new keys:

```python
def _apply_compression(state, magnitude):
    new_state = dict(state)
    new_state["compression_ratio"] = state.get("compression_ratio", 1.0) * (1 + magnitude)
    return new_state


def _apply_retention(state, magnitude):
    new_state = dict(state)
    new_state["retention_slots"] = int(state.get("retention_slots", 288) * (1 + magnitude))
    return new_state


def _apply_replication(state, magnitude):
    new_state = dict(state)
    new_state["replication_factor"] = int(state.get("replication_factor", 1) + magnitude)
    return new_state
```

and, inside `INTERVENTIONS`:

```python
    "compression_change": {
        "apply": _apply_compression, "cost_fn": lambda m: 0.3 * m,
        "max_magnitude": 3.0, "target_var": "compression_ratio", "max_duration_slots": 288,
    },
    "retention_change": {
        "apply": _apply_retention, "cost_fn": lambda m: 0.4 * m,
        "max_magnitude": 2.0, "target_var": "retention_slots", "max_duration_slots": 288,
    },
    "replication_change": {
        "apply": _apply_replication, "cost_fn": lambda m: 1.5 * m,
        "max_magnitude": 2.0, "target_var": "replication_factor", "max_duration_slots": 288,
    },
```

Update the baseline dict in `test_every_intervention_declares_the_state_variable_it_manipulates` to include `"compression_ratio": 1.0, "retention_slots": 288, "replication_factor": 1`.

- [ ] **Step 8: Run the whole suite and commit**

Run: `python -m pytest -q`
Expected: PASS

```bash
git add src/aco/sim/telemetry.py src/aco/interventions/library.py tests/
git commit -m "feat(sensing): price sampling, compression, retention and replication

Section 6.5 names four sensing/storage knobs governed by the same VoI
criterion. Two existed as interventions and neither had any effect --
sampling_rate_hz was written to SiteState and never read. telemetry_footprint
supplies the sampling -> bandwidth -> storage chain, which also gives Section
10.1's bandwidth and storage-footprint metrics their first source."
```

---

## Task 7: Rebuild the cluster join (audit B1)

Only 18 of 500 trace shards are on disk (~25 hours), so `cluster_days = [0, 1]` and `sim_day_cluster = cluster_days[sim_day_solar % 2]`. The inner merge on `(sim_day_cluster, hour_of_day)` drops every solar slot whose hour is absent from the cluster day it drew: even solar days keep 09:20–23:55 (176 slots, mean 28.7 MW), odd days keep 00:00–10:25 (126 slots, mean 10.2 MW). **52.5% of the solar year survives** and available fleet power oscillates ~3× with day parity for a purely artifactual reason.

**Files:**
- Modify: `src/aco/data/sim_clock.py`
- Test: `tests/data/test_sim_clock.py`

**Interfaces:**
- Produces: `aco.data.sim_clock.build_diurnal_cluster_profile(machine_util_df: pd.DataFrame, machine_ids) -> pd.DataFrame` — collapses the available cluster window to one mean `cpu_rate_sum` per `hour_of_day` for the given machines. Columns: `hour_of_day`, `cpu_rate_sum`.
- Modifies: `build_site_timeline` joins each site's solar rows to that profile on `hour_of_day` alone, so every solar slot is retained and `sim_day_cluster` is no longer needed as a join key.

**Honesty note for the paper:** this trades away day-to-day cluster variability, of which only ~25 hours exists on disk anyway. State it as a limitation — the cloud workload is a fixed diurnal profile, not an independently varying trace — rather than implying 29 days of trace were used.

- [ ] **Step 1: Write the failing tests**

```python
def test_diurnal_profile_has_one_row_per_hour_of_day():
    from aco.data.sim_clock import build_diurnal_cluster_profile

    mu = pd.DataFrame({
        "machine_id": ["m1", "m1", "m2", "m2"],
        "hour_of_day": [0.0, 0.5, 0.0, 0.5],
        "cpu_rate_sum": [1.0, 2.0, 3.0, 4.0],
        "wall_time": pd.to_datetime(["2011-05-13 00:00", "2011-05-13 00:30"] * 2),
    })
    profile = build_diurnal_cluster_profile(mu, ["m1", "m2"])

    assert profile["hour_of_day"].tolist() == [0.0, 0.5]
    assert profile["cpu_rate_sum"].tolist() == [4.0, 6.0]  # summed across machines


def test_site_timeline_keeps_every_solar_slot():
    # The defect this fixes: joining on (sim_day_cluster, hour_of_day) dropped
    # 47.5% of solar slots in an alternating day-parity pattern, because only
    # two cluster days exist and neither covers a full 24 hours.
    plants = pd.DataFrame([{"plant_id": "p1", "region": "AZ"}])
    hours = [h / 2 for h in range(48)]
    power = pd.DataFrame({
        "plant_id": ["p1"] * 96,
        "kind": ["Actual"] * 96,
        "timestamp": pd.to_datetime(["2006-01-01"] * 48 + ["2006-01-02"] * 48)
                     + pd.to_timedelta(hours * 2, unit="h"),
        "power_mw": [10.0] * 96,
        "hour_of_day": hours * 2,
    })
    mu = pd.DataFrame({
        "machine_id": ["m1"] * 20,
        "wall_time": pd.to_datetime(["2011-05-13 09:20"] * 20)
                     + pd.to_timedelta(range(20), unit="h"),
        "cpu_rate_sum": [5.0] * 20,
    })
    mu["hour_of_day"] = mu["wall_time"].dt.hour + mu["wall_time"].dt.minute / 60.0

    timeline = build_site_timeline(power, plants, mu, n_sites=1, seed=0)

    # Every solar row whose hour_of_day the profile covers must survive, and
    # coverage must not depend on the parity of sim_day_solar.
    per_day = timeline.groupby("sim_day_solar").size()
    assert per_day.nunique() == 1, f"coverage varies by day: {per_day.to_dict()}"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/data/test_sim_clock.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_diurnal_cluster_profile'`

- [ ] **Step 3: Implement the profile and rewire the join**

In `src/aco/data/sim_clock.py`:

```python
def build_diurnal_cluster_profile(machine_util_df, machine_ids) -> pd.DataFrame:
    """Collapse the available cluster window to one mean load per hour-of-day.

    Only ~25 hours of the Google trace are on disk (18 of 500 shards), which
    yields two partial cluster days. Cycling solar days onto those two days
    and joining on (sim_day_cluster, hour_of_day) dropped 47.5% of solar slots
    in an alternating pattern, because neither cluster day covers a full 24
    hours -- producing a ~3x swing in available fleet power that tracked day
    parity rather than anything physical.

    Collapsing to a diurnal profile keeps every solar slot. The cost is
    day-to-day cluster variability, of which barely one day exists in this
    download anyway; the cloud workload becomes a fixed diurnal profile, which
    must be disclosed as a limitation.
    """
    subset = machine_util_df[machine_util_df["machine_id"].isin(machine_ids)]
    return (
        subset.groupby("hour_of_day", as_index=False)["cpu_rate_sum"]
        .sum()
        .sort_values("hour_of_day")
        .reset_index(drop=True)
    )
```

and in `build_site_timeline`, replace the per-site block that computes `sim_day_cluster` and merges on two keys:

```python
    rows = []
    for (_, site_row), machines in zip(sites.iterrows(), machine_blocks):
        site_power = actual[actual["plant_id"] == site_row["plant_id"]].rename(
            columns={"sim_day": "sim_day_solar"}
        )
        profile = build_diurnal_cluster_profile(mu, machines)
        merged = pd.merge(site_power, profile, on="hour_of_day", how="inner")
        merged["site_id"] = site_row["plant_id"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)
```

- [ ] **Step 4: Run them and watch them pass**

Run: `python -m pytest tests/data/test_sim_clock.py -q`
Expected: PASS

- [ ] **Step 5: Regenerate the fleet timeline and verify the artifact is gone**

```bash
python -m aco.data.run_sim_clock_ingest
python -c "
import pandas as pd
t = pd.read_parquet('fleet_data/processed/site_timeline.parquet')
one = t[t.site_id == t.site_id.iloc[0]]
print('rows per site:', len(one), 'of a possible', 365*288)
ev = one[one.sim_day_solar % 2 == 0]; od = one[one.sim_day_solar % 2 == 1]
print('mean power even days:', round(ev.power_mw.mean(), 3))
print('mean power odd  days:', round(od.power_mw.mean(), 3))
"
```

Expected: rows per site close to 365 × 288, and the even/odd means within a few percent of each other rather than 28.7 vs 10.2. If they still differ by ~3×, the join was not rewired correctly — stop and fix before committing.

- [ ] **Step 6: Run the whole suite and commit**

```bash
python -m pytest -q
git add src/aco/data/sim_clock.py tests/data/test_sim_clock.py
git commit -m "fix(data): join cluster load on a diurnal profile, not two partial days

Only 18 of 500 trace shards are on disk (~25 hours), giving two partial
cluster days. Cycling solar days onto them and joining on
(sim_day_cluster, hour_of_day) dropped 47.5% of solar slots in an alternating
day-parity pattern: even days kept 09:20-23:55 at mean 28.7 MW, odd days kept
00:00-10:25 at mean 10.2 MW. Every Phase 8 series would have carried a
period-2 component that was the join, not physics.

Collapsing to one mean load per hour-of-day keeps every solar slot. The cost
is day-to-day cluster variability, of which barely one day exists in this
download; disclose the cloud workload as a fixed diurnal profile."
```

---

## Self-Review

**Spec coverage.** §6.1 representation — partially addressed (Task 1 makes the interventional refinement sound; the semantic/event layer is explicitly out of scope and needs a decision). §6.2 VoI — Task 1 prices duration into cost and risk. §6.3 world model — Task 5 adds counterfactuals; the edge–cloud infrastructure model remains unaddressed and needs its own plan. §6.4 DRO — Task 4 builds the causal-uncertainty ambiguity set; formal regret bounds remain a theory task, not a coding one. §6.5 sensing/storage — Task 6 covers all four knobs and prices them; making them *decision variables inside `solve_slot`* (§8.3) is deferred to the Phase 7–8 plan, since it changes the optimizer's shape and the baselines must be written against the final shape. §8.1 intervention library — Task 1 adds duration; per-site subsets remain open. §8.4 causal model update — closed in commit `5297c6c`, strengthened by Tasks 1 and 3. §10.1 metrics — Tasks 3 and 6 give the uncertainty-over-time, bandwidth and storage metrics their sources; latency and energy still have none. §10.2 baselines — out of scope by design.

**Known gaps this plan does not close,** carried forward to the Phase 7–8 plan: latency has no model anywhere; interventions are still broadcast fleet-wide rather than selected per site (§8.1); magnitude is still pinned at `max_magnitude / 2` rather than searched; the Lyapunov queue is still write-only (audit C3); there is no edge tier (§7); and no FDR correction on the discovered edge set (audit C5).

**Placeholder scan.** Every step contains runnable code or a concrete command with a stated expected result. Task 5 Step 1 originally contained a convoluted assertion; it is replaced inline with the simpler equivalent and the instruction to delete the first form.

**Type consistency.** `target_var` is used identically in Tasks 1 and 6. `edge_uncertainty(graph, var_names)` and `node_uncertainty(graph, node, var_names)` from Task 3 are consumed with those exact signatures in Task 4 (`causal_risk_scenarios`) and in Task 3's own orchestrator wiring. `causal_risk_scenarios(graph, var_names, base_risk, n_scenarios, seed)` matches its use in Task 4 Step 7. `max_duration_slots` introduced in Task 1 is set on the three new interventions in Task 6. `telemetry_footprint`'s keyword names match between its definition and all three of its tests.

**Ordering note.** Task 3 must precede Task 4 (`causal_risk_scenarios` imports `edge_uncertainty`). Task 1 must precede Task 6 (the new interventions need `max_duration_slots`). Tasks 2, 5 and 7 are independent and may run in any order.
