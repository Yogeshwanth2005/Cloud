import networkx as nx
import numpy as np
import pandas as pd
import pytest

from aco.causal.world_model import CausalWorldModel
from aco.interventions.library import INTERVENTIONS
from aco.optim.orchestrator import ActiveOrchestrator

VARS = ["power_mw", "cpu_rate_sum"]


def _signal_frame(n, seed, noise=80.0):
    """An autocorrelated power_mw -> cpu_rate_sum series.

    `power_mw` is the variable `curtailment` actually manipulates, so it is a
    legitimate probe target; `noise` controls how tight the relationship is,
    and the post-intervention frame uses a smaller value to stand in for the
    sharper signal a real probe produces.
    """
    rng = np.random.default_rng(seed)
    phi = 0.8
    noise_std = 100 * (1 - phi ** 2) ** 0.5
    power = np.empty(n)
    power[0] = rng.normal(500, 100)
    for t in range(1, n):
        power[t] = 500 + phi * (power[t - 1] - 500) + rng.normal(0, noise_std)
    return pd.DataFrame({"power_mw": power, "cpu_rate_sum": 0.6 * power + rng.normal(0, noise, n)})


def _fit_signal_world_model():
    df = _signal_frame(300, seed=4)
    graph = nx.DiGraph()
    graph.add_edge("power_mw", "cpu_rate_sum", pval=1e-3)
    model = CausalWorldModel(graph)
    model.fit(df)
    return model, df, graph


def test_orchestrator_step_returns_allocation_and_selected_intervention():
    model, df, graph = _fit_signal_world_model()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0)
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


# How the caller manages its observation frame. The orchestrator must behave
# identically under all three: an expanding history, a fixed-size sliding
# window (len never grows), and a frame that is rebased smaller mid-run.
FRAME_POLICIES = {
    "expanding": lambda hist, row: pd.concat([hist, row], ignore_index=True),
    "sliding": lambda hist, row: pd.concat([hist, row], ignore_index=True).tail(300).reset_index(drop=True),
    "rebased": lambda hist, row: pd.concat([hist, row], ignore_index=True).tail(250).reset_index(drop=True),
}


def _run_slots(orch, model, graph, site_states, pre, post, n_slots, advance):
    """Drive `n_slots` slots, appending exactly one new observation per slot."""
    hist = pre
    results = []
    for t in range(n_slots):
        hist = advance(hist, post.iloc[[t]])
        results.append(
            orch.step(dict(site_states), model, hist, graph, node_candidates=["power_mw"], var_names=VARS)
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
        if r["intervention"] is not None:
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
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=120)

    results = _run_slots(orch, model, graph, site_states, pre, post, 121, FRAME_POLICIES["expanding"])
    closing = results[120]

    assert closing["causal_update"] is not None
    updated = closing["graph"]
    assert updated is not graph, "the update must not mutate the caller's graph"
    assert updated["power_mw"]["cpu_rate_sum"]["pval"] < graph["power_mw"]["cpu_rate_sum"]["pval"]


def test_orchestrator_refits_world_model_onto_the_updated_graph():
    pre, post, graph, model, site_states = _loop_fixture()
    orch = ActiveOrchestrator(V=1.0, cvar_alpha=0.9, cvar_limit=5.0, min_post_obs=120)
    regressor_before = model.models["cpu_rate_sum"][1]

    results = _run_slots(orch, model, graph, site_states, pre, post, 121, FRAME_POLICIES["expanding"])

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
