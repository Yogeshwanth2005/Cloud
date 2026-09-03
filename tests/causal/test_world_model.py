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
