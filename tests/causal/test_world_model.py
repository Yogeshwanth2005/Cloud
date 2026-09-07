import networkx as nx
import numpy as np
import pandas as pd

from aco.causal.world_model import CausalWorldModel


def test_do_changes_downstream_prediction():
    rng = np.random.default_rng(0)
    n = 300
    irradiance = rng.normal(500, 100, n)
    dc_power = 0.2 * irradiance + rng.normal(0, 1, n)
    df = pd.DataFrame({"poa_irradiance": irradiance, "dc_power": dc_power})

    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power")

    model = CausalWorldModel(graph)
    model.fit(df)

    baseline = model.predict(df.iloc[[0]])["dc_power"].iloc[0]
    intervened = model.do(df.iloc[[0]], {"poa_irradiance": 0.0})["dc_power"].iloc[0]
    assert intervened < baseline  # dropping irradiance to 0 should predict near-zero power
    assert abs(intervened) < abs(baseline)


def test_estimate_uncertainty_reduction_is_positive_for_a_real_causal_edge():
    # poa_irradiance -> dc_power is a real, lag-1 (unambiguous) causal edge, so
    # probing poa_irradiance with a varying batch and refitting should sharpen
    # (lower) its edges' pval relative to the pre-intervention fit -- proving
    # the world model, not an arbitrary caller, is what estimates this proxy
    # for proposal Section 8.2 ("the Causal World Model estimates the expected
    # reduction in causal uncertainty").
    rng = np.random.default_rng(7)
    n = 400
    phi = 0.8
    # AR(1) irradiance: PCMCI+ needs real temporal asymmetry to orient a
    # same-lag link at all (see fit_observational_graph's docstring) -- an
    # i.i.d. series linked only at lag 0 comes back unorientable ('o-o').
    noise_std = 100 * (1 - phi ** 2) ** 0.5
    irradiance = np.empty(n)
    irradiance[0] = rng.normal(500, 100)
    for t in range(1, n):
        irradiance[t] = 500 + phi * (irradiance[t - 1] - 500) + rng.normal(0, noise_std)
    # Noisy enough that a DEFAULT_PROBE_WINDOW-row observational fit can't pin
    # the edge down on its own, which is what leaves headroom for the probe to
    # sharpen it. With a clean signal the observational fit already identifies
    # the edge, and a zero reduction is then the correct answer -- there is
    # nothing left to learn. See the companion test below.
    dc_power = 0.6 * irradiance + rng.normal(0, 2000, n)
    df = pd.DataFrame({"poa_irradiance": irradiance, "dc_power": dc_power})

    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power")
    model = CausalWorldModel(graph)
    model.fit(df)

    reduction = model.estimate_uncertainty_reduction(
        df, "poa_irradiance", magnitude=0.5, var_names=["poa_irradiance", "dc_power"], tau_max=1,
    )
    assert reduction > 0.0


def test_estimate_uncertainty_reduction_is_zero_for_unknown_node():
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power")
    model = CausalWorldModel(graph)
    model.fit(pd.DataFrame({"poa_irradiance": [1.0, 2.0], "dc_power": [0.2, 0.4]}))

    reduction = model.estimate_uncertainty_reduction(
        pd.DataFrame({"poa_irradiance": [1.0], "dc_power": [0.2]}),
        "not_a_node", magnitude=0.5, var_names=["poa_irradiance", "dc_power"],
    )
    assert reduction == 0.0


def test_default_probe_window_is_large_enough_to_identify_an_edge():
    # PCMCI+ needs roughly 120 rows to orient a contemporaneous link at this
    # variable count -- a sample-size floor, not a signal-strength one. A probe
    # window below that makes estimate_uncertainty_reduction structurally
    # blind: both the pre and post fits find nothing, so the reduction is
    # exactly zero no matter what a real probe would reveal. The default was
    # 100, which is below the floor.
    import warnings

    from aco.causal.graph import fit_observational_graph
    from aco.causal.world_model import DEFAULT_PROBE_WINDOW

    rng = np.random.default_rng(4)
    n = 600
    phi = 0.8
    power = np.empty(n)
    power[0] = rng.normal(500, 100)
    for t in range(1, n):
        power[t] = 500 + phi * (power[t - 1] - 500) + rng.normal(0, 100 * (1 - phi ** 2) ** 0.5)
    df = pd.DataFrame({"power_mw": power, "cpu_rate_sum": 0.6 * power + rng.normal(0, 20, n)})

    window = df.tail(DEFAULT_PROBE_WINDOW).reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        graph = fit_observational_graph(window, var_names=["power_mw", "cpu_rate_sum"], tau_max=1)

    assert graph.number_of_edges() > 0, (
        f"default probe window of {DEFAULT_PROBE_WINDOW} rows cannot identify "
        "even a clean edge, so the VoI estimator is blind by construction"
    )
