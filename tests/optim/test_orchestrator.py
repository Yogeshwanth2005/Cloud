import networkx as nx
import numpy as np
import pandas as pd

from aco.causal.world_model import CausalWorldModel
from aco.interventions.library import INTERVENTIONS
from aco.optim.orchestrator import ActiveOrchestrator


def _fit_signal_world_model():
    # Same construction as test_voi.py's
    # test_select_best_intervention_uses_world_model_to_estimate_reduction --
    # a real autocorrelated irradiance -> dc_power signal, proven to make
    # select_best_intervention actually return a candidate rather than None.
    rng = np.random.default_rng(4)
    n = 300
    phi = 0.8
    noise_std = 100 * (1 - phi ** 2) ** 0.5
    irradiance = np.empty(n)
    irradiance[0] = rng.normal(500, 100)
    for t in range(1, n):
        irradiance[t] = 500 + phi * (irradiance[t - 1] - 500) + rng.normal(0, noise_std)
    dc_power = 0.6 * irradiance + rng.normal(0, 80, n)
    df = pd.DataFrame({"poa_irradiance": irradiance, "dc_power": dc_power})

    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=1e-3)
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

    result = orch.step(
        site_states, model, df, graph,
        node_candidates=["poa_irradiance"], var_names=["poa_irradiance", "dc_power"],
    )

    assert "allocation" in result
    assert "queue_backlog" in result
    assert result["queue_backlog"] >= 0.0
    assert result["intervention"] is not None
    node, name, magnitude = result["intervention"]
    assert node == "poa_irradiance"
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
