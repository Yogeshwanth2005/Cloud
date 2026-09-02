import pandas as pd
from aco.sim.engine import ReplayEngine


def _toy_timeline():
    return pd.DataFrame({
        "site_id": ["s1", "s1", "s2", "s2"],
        "sim_day_solar": [0, 1, 0, 1],
        "hour_of_day": [12.0, 12.0, 12.0, 12.0],
        "power_mw": [10.0, 20.0, 5.0, 6.0],
        "cpu_rate_sum": [1.0, 1.5, 2.0, 2.5],
    })


def test_reset_returns_first_tick_for_every_site():
    engine = ReplayEngine(_toy_timeline())
    states = engine.reset()
    assert set(states.keys()) == {"s1", "s2"}
    assert states["s1"].power_mw == 10.0


def test_step_advances_and_applies_curtailment():
    engine = ReplayEngine(_toy_timeline())
    engine.reset()
    states = engine.step({"s1": {"curtailment_frac": 0.5, "sampling_rate_hz": 1.0}})
    assert states["s1"].power_mw == 10.0  # 20.0 * (1 - 0.5)
    assert states["s2"].power_mw == 6.0   # no intervention specified -> passthrough
