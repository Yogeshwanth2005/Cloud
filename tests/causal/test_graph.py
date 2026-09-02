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
