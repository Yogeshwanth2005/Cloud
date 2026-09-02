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


def test_fit_observational_graph_keeps_strongest_lag_not_last_lag():
    # dc_power is driven by irradiance at both lag 0 (strong) and lag 1 (weaker
    # but still statistically significant). A DiGraph only has one edge slot
    # per (u, v) pair, so the edge-construction loop must keep the
    # lowest-pval (most significant) lag's attributes -- not whichever lag
    # happens to be processed last -- or the strong lag-0 relationship gets
    # silently clobbered by the weaker lag-1 one.
    rng = np.random.default_rng(1)
    n = 5000
    irradiance = rng.normal(500, 100, n)
    irr_lag1 = np.roll(irradiance, 1)
    irr_lag1[0] = irradiance[0]
    dc_power = 0.6 * irradiance + 0.2 * irr_lag1 + rng.normal(0, 3, n)
    df = pd.DataFrame({"poa_irradiance": irradiance, "dc_power": dc_power})

    graph = fit_observational_graph(df, var_names=["poa_irradiance", "dc_power"], tau_max=1)

    assert graph.has_edge("poa_irradiance", "dc_power")
    edge = graph["poa_irradiance"]["dc_power"]
    assert edge["lag"] == 0
    assert edge["weight"] > 0.9  # lag-0 link strength, not lag-1's ~0.3


def test_update_graph_with_intervention_sharpens_pval():
    # Simulate pre- and post-intervention data: pre has weak/noisy signal (small n),
    # post has the same relationship but much cleaner (large n). The test verifies
    # that update_graph_with_intervention replaces the weak pre-intervention estimate
    # with the sharper (lower pval) post-intervention estimate.
    from aco.causal.graph import update_graph_with_intervention

    rng = np.random.default_rng(1)
    irr_pre = rng.normal(500, 100, 30)
    pre = pd.DataFrame({
        "poa_irradiance": irr_pre,
        "dc_power": 0.2 * irr_pre + rng.normal(0, 20, 30),  # weak but real signal pre-intervention
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
