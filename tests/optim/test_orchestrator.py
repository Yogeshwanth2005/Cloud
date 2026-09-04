import networkx as nx
import numpy as np
import pandas as pd
import pytest

from aco.causal.world_model import CausalWorldModel
from aco.interventions.library import INTERVENTIONS
from aco.optim.orchestrator import ActiveOrchestrator

VARS = ["power_mw", "cpu_rate_sum"]


def _signal_frame(n, seed, noise=80.0, driver="power_mw", response="cpu_rate_sum"):
    """An autocorrelated `driver` -> `response` series.

    `driver` defaults to `power_mw`, the variable `curtailment` manipulates.
    `noise` controls how tight the relationship is; the post-intervention frame
    uses a smaller value to stand in for the sharper signal a real probe
    produces.
    """
    rng = np.random.default_rng(seed)
    phi = 0.8
    noise_std = 100 * (1 - phi ** 2) ** 0.5
    power = np.empty(n)
    power[0] = rng.normal(500, 100)
    for t in range(1, n):
        power[t] = 500 + phi * (power[t - 1] - 500) + rng.normal(0, noise_std)
    return pd.DataFrame({driver: power, response: 0.6 * power + rng.normal(0, noise, n)})


def _fit_signal_world_model():
    df = _signal_frame(300, seed=4)
    graph = nx.DiGraph()
    graph.add_edge("power_mw", "cpu_rate_sum", pval=1e-3)
    model = CausalWorldModel(graph)
    model.fit(df)
    return model, df, graph


def test_orchestrator_step_returns_allocation_and_selected_intervention():
    model, df, graph = _fit_signal_world_model()
    # min_post_obs must not exceed curtailment's 36-slot duration bound, or the
    # intervention is (correctly) rejected as unholdable for the whole window.
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=5)
    site_states = {
        "s1": {"power_mw": 10.0, "compute_demand": 6.0, "cost_per_unit": 1.0, "risk_sample": [0.1]},
        "s2": {"power_mw": 5.0, "compute_demand": 4.0, "cost_per_unit": 2.0, "risk_sample": [0.2]},
    }

    result = orch.step(site_states, model, df, graph, node_candidates=["power_mw"], var_names=VARS)

    assert "allocation" in result
    assert "queue_backlog" in result
    assert result["queue_backlog"] >= 0.0
    assert result["intervention"] is not None
    node, name, magnitude = result["intervention"]
    assert node == "power_mw"
    assert name in INTERVENTIONS


def test_orchestrator_queue_grows_after_repeated_violation():
    # solve_slot enforces the CVaR limit as a hard per-slot constraint, so a
    # feasible slot never really "violates" it -- the queue can only register
    # a violation when the slot's CVaR requirement is outright unmeetable
    # (cvar_limit below the always-available cvar=0 of allocating nothing).
    graph = nx.DiGraph()
    model = CausalWorldModel(graph)
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=-1.0)
    site_states = {"s1": {"power_mw": 10.0, "compute_demand": 6.0, "cost_per_unit": 1.0, "risk_sample": [5.0]}}

    first = orch.step(site_states, model, df, graph, node_candidates=[], var_names=[])
    second = orch.step(site_states, model, df, graph, node_candidates=[], var_names=[])

    assert second["queue_backlog"] >= first["queue_backlog"]


# --- Phase 1: the closed active-learning loop (proposal Section 8.4) ---------


class _AlwaysProbes(CausalWorldModel):
    """Test double pinning the VoI uncertainty-reduction estimate high, so an
    intervention is selected whenever one is allowed.

    Isolates the orchestrator's loop mechanics from the VoI heuristic itself,
    whose estimate is knife-edge around this fixture (a single extra row swings
    it from 1.0 to 0.02) and so cannot drive a stable test.
    """

    def estimate_uncertainty_reduction(self, *args, **kwargs):
        return 0.9


def _loop_fixture():
    pre = _signal_frame(300, seed=4)
    post = _signal_frame(200, seed=5, noise=20.0)
    graph = nx.DiGraph()
    graph.add_edge("power_mw", "cpu_rate_sum", pval=0.4, weight=0.3, lag=0)
    model = _AlwaysProbes(graph)
    model.fit(pre)
    site_states = {"s1": {"power_mw": 10.0, "compute_demand": 6.0, "cost_per_unit": 1.0, "risk_sample": [0.1]}}
    return pre, post, graph, model, site_states


SENSING_VARS = ["sampling_rate_hz", "cpu_rate_sum"]


def _sensing_loop_fixture():
    """A probe of `sampling_rate_hz`, which `high_res_sampling` manipulates.

    Needed because PCMCI+ cannot orient a contemporaneous link below roughly
    120 samples at this variable count -- measured, and not a signal-strength
    limit: even a near-deterministic relationship yields no edge at 36 or 60
    rows. A curtailment may be held for at most 36 slots (3 hours), so it can
    never supply an identifiable window; only the sensing and logging probes,
    bounded at 288 slots, can. The edge is physically real in this system:
    raising the sampling rate produces more telemetry to ingest and process,
    which raises cluster CPU load -- the same chain Section 6.5 prices.
    """
    pre = _signal_frame(300, seed=4, driver="sampling_rate_hz")
    post = _signal_frame(200, seed=5, noise=20.0, driver="sampling_rate_hz")
    graph = nx.DiGraph()
    graph.add_edge("sampling_rate_hz", "cpu_rate_sum", pval=0.4, weight=0.3, lag=0)
    model = _AlwaysProbes(graph)
    model.fit(pre)
    site_states = {"s1": {"power_mw": 10.0, "compute_demand": 6.0,
                          "cost_per_unit": 1.0, "risk_sample": [0.1]}}
    return pre, post, graph, model, site_states


# How the caller manages its observation frame. The orchestrator must behave
# identically under all three: an expanding history, a fixed-size sliding
# window (len never grows), and a frame that is rebased smaller mid-run.
FRAME_POLICIES = {
    "expanding": lambda hist, row: pd.concat([hist, row], ignore_index=True),
    "sliding": lambda hist, row: pd.concat([hist, row], ignore_index=True).tail(300).reset_index(drop=True),
    "rebased": lambda hist, row: pd.concat([hist, row], ignore_index=True).tail(250).reset_index(drop=True),
}


def _run_slots(orch, model, graph, site_states, pre, post, n_slots, advance,
               node_candidates=("power_mw",), var_names=None):
    """Drive `n_slots` slots, appending exactly one new observation per slot."""
    hist = pre
    results = []
    for t in range(n_slots):
        hist = advance(hist, post.iloc[[t]])
        results.append(
            orch.step(dict(site_states), model, hist, graph,
                      node_candidates=list(node_candidates),
                      var_names=list(var_names) if var_names else VARS)
        )
    return results


@pytest.mark.parametrize("policy", sorted(FRAME_POLICIES))
def test_loop_closes_after_exactly_min_post_obs_observations_under_any_frame_policy(policy):
    # The orchestrator must count its own post-intervention observations
    # rather than differencing len(df): a caller feeding a fixed-size sliding
    # window never grows the frame, so a length-based marker would stall
    # forever and the loop would never close.
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=5)

    results = _run_slots(orch, model, graph, site_states, pre, post, 12, FRAME_POLICIES[policy])
    closed_at = [t for t, r in enumerate(results) if r["causal_update"] is not None]

    # slot 0 fires the intervention; slots 1-5 supply its five observations.
    assert closed_at == [5, 10], f"{policy} frame closed at {closed_at}"
    assert all(results[t]["causal_update"]["n_post_obs"] == 5 for t in closed_at)


def test_orchestrator_alternates_one_intervention_with_one_full_observation_window():
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=5)

    results = _run_slots(orch, model, graph, site_states, pre, post, 12, FRAME_POLICIES["expanding"])

    events = []
    for r in results:
        if r["causal_update"] is not None:
            events.append("update")
        # The intervention is now held for every slot of its window, so a new
        # cycle is marked by its cost being charged, not by presence.
        if r["intervention_cost"] > 0:
            events.append("intervene")
    assert events == ["intervene", "update", "intervene", "update", "intervene"]


def test_causal_update_reports_the_variable_that_was_actually_manipulated():
    # The VoI layer picks a node to probe; the graph update must be told the
    # variable the chosen intervention really manipulates, since that is what
    # licenses severing its incoming edges.
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=5)

    results = _run_slots(orch, model, graph, site_states, pre, post, 6, FRAME_POLICIES["expanding"])

    _node, name, _magnitude = results[0]["intervention"]
    update = results[5]["causal_update"]
    assert update["target_var"] == INTERVENTIONS[name]["target_var"] == "power_mw"


def test_orchestrator_sharpens_the_causal_graph_after_observing_the_intervention():
    pre, post, graph, model, site_states = _sensing_loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=120)

    results = _run_slots(orch, model, graph, site_states, pre, post, 121,
                         FRAME_POLICIES["expanding"],
                         node_candidates=("sampling_rate_hz",), var_names=SENSING_VARS)
    closing = results[120]

    assert closing["causal_update"] is not None
    updated = closing["graph"]
    assert updated is not graph, "the update must not mutate the caller's graph"
    assert (updated["sampling_rate_hz"]["cpu_rate_sum"]["pval"]
            < graph["sampling_rate_hz"]["cpu_rate_sum"]["pval"])


def test_orchestrator_refits_world_model_onto_the_updated_graph():
    pre, post, graph, model, site_states = _sensing_loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=120)
    regressor_before = model.models["cpu_rate_sum"][1]

    results = _run_slots(orch, model, graph, site_states, pre, post, 121,
                         FRAME_POLICIES["expanding"],
                         node_candidates=("sampling_rate_hz",), var_names=SENSING_VARS)

    assert model.graph is results[120]["graph"], "world model must reason over the updated graph"
    assert model.models["cpu_rate_sum"][1] is not regressor_before, "world model must be refit"


def test_orchestrator_does_not_update_graph_when_no_intervention_was_applied():
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=5)

    observed = pd.concat([pre, post], ignore_index=True)
    result = orch.step(dict(site_states), model, observed, graph, node_candidates=[], var_names=VARS)

    assert result["intervention"] is None
    assert result["causal_update"] is None
    assert result["graph"] is graph


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
