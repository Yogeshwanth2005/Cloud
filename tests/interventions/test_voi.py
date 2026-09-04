import networkx as nx
import numpy as np
import pandas as pd
import pytest

from aco.causal.world_model import CausalWorldModel
from aco.interventions.voi import (
    INFO_VALUE_SCALE, RISK_SCALE, score_intervention, select_best_intervention,
)
from aco.interventions.library import INTERVENTIONS


def test_score_intervention_is_positive_when_uncertainty_reduction_is_high():
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.2)
    score = score_intervention(
        graph, "poa_irradiance", expected_uncertainty_reduction=0.15,
        name="high_res_sampling", magnitude=0.1,  # small fraction of the 4.0 safety bound
    )
    assert score > 0


def test_score_intervention_subtracts_risk_proportional_to_safety_margin_used():
    # Proposal Section 6.2 is "Value-of-Information under Risk Constraints" --
    # score_intervention must net out risk, not just cost. Risk is modeled as
    # the fraction of the intervention's own pre-registered safety bound
    # (max_magnitude) consumed, since that's a real quantity every registered
    # intervention already carries and cost_fn alone doesn't capture it (e.g.
    # high_res_logging's cost_fn is tiny relative to its 9.0 max_magnitude).
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.2)
    magnitude = 0.15
    score = score_intervention(
        graph, "poa_irradiance", expected_uncertainty_reduction=0.1,
        name="curtailment", magnitude=magnitude,
    )
    info_value = 0.1 * 1 * INFO_VALUE_SCALE
    cost = INTERVENTIONS["curtailment"]["cost_fn"](magnitude)
    risk = (magnitude / INTERVENTIONS["curtailment"]["max_magnitude"]) * RISK_SCALE
    assert score == pytest.approx(info_value - cost - risk)


def test_select_best_intervention_returns_none_when_all_negative():
    graph = nx.DiGraph()
    graph.add_edge("power_mw", "cpu_rate_sum", pval=0.01)  # already well-known -> little to gain
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "power_mw": rng.normal(500, 1, 10),
        "cpu_rate_sum": rng.normal(100, 1, 10),
    })
    model = CausalWorldModel(graph)
    model.fit(df)

    result = select_best_intervention(
        model, df, graph, node_candidates=["power_mw"], var_names=["power_mw", "cpu_rate_sum"],
    )
    assert result is None


def test_select_best_intervention_uses_world_model_to_estimate_reduction():
    # There is no path from voi.py to CausalWorldModel unless select_best_intervention
    # actually calls it -- this proves the decision criterion's key input (expected
    # uncertainty reduction) is estimated by the world model, not caller-supplied.
    rng = np.random.default_rng(4)
    n = 300
    phi = 0.8
    noise_std = 100 * (1 - phi ** 2) ** 0.5
    irradiance = np.empty(n)
    irradiance[0] = rng.normal(500, 100)
    for t in range(1, n):
        irradiance[t] = 500 + phi * (irradiance[t - 1] - 500) + rng.normal(0, noise_std)
    cpu_rate_sum = 0.6 * irradiance + rng.normal(0, 80, n)
    df = pd.DataFrame({"power_mw": irradiance, "cpu_rate_sum": cpu_rate_sum})

    graph = nx.DiGraph()
    graph.add_edge("power_mw", "cpu_rate_sum", pval=1e-3)
    model = CausalWorldModel(graph)
    model.fit(df)

    result = select_best_intervention(
        model, df, graph, node_candidates=["power_mw"], var_names=["power_mw", "cpu_rate_sum"],
    )
    assert result is not None
    node, name, magnitude = result
    assert node == "power_mw"
    assert name in INTERVENTIONS


class _HighGain(CausalWorldModel):
    """Pins the uncertainty-reduction estimate high so scoring, not the
    heuristic's own behavior, decides what gets selected."""

    def estimate_uncertainty_reduction(self, *args, **kwargs):
        return 0.9


def test_select_best_intervention_rejects_a_node_no_intervention_can_manipulate():
    # You cannot intervene on irradiance -- it is the sun. None of the four
    # registered interventions manipulate it, so there is nothing to select.
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.3)
    result = select_best_intervention(
        _HighGain(graph), None, graph,
        node_candidates=["poa_irradiance"], var_names=["poa_irradiance", "dc_power"],
    )
    assert result is None


def test_select_best_intervention_picks_the_intervention_that_manipulates_the_node():
    graph = nx.DiGraph()
    graph.add_edge("power_mw", "cpu_rate_sum", pval=0.3)
    result = select_best_intervention(
        _HighGain(graph), None, graph,
        node_candidates=["power_mw"], var_names=["power_mw", "cpu_rate_sum"],
    )
    assert result is not None
    node, name, _magnitude = result
    assert node == "power_mw"
    assert INTERVENTIONS[name]["target_var"] == "power_mw"
