import pandas as pd

from aco.causal.validate_world_model import label_clipping_events


def test_label_clipping_events_flags_saturation():
    df = pd.DataFrame({
        "ac_power": [10.0, 49.5, 50.0, 30.0],
        "dc_power": [10.5, 55.0, 60.0, 31.0],
    })
    flags = label_clipping_events(df, "ac_power", "dc_power", rated_kw=50.0)
    assert flags.tolist() == [False, True, True, False]
