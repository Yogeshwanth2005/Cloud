import pandas as pd
import pytest

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

    s1 = states["s1"]
    assert s1.site_id == "s1"
    assert s1.sim_day == 0
    assert s1.hour_of_day == 12.0
    assert s1.power_mw == 10.0
    assert s1.cpu_rate_sum == 1.0
    assert s1.curtailment_frac == 0.0
    assert s1.sampling_rate_hz == 1.0


def test_step_advances_and_applies_curtailment():
    engine = ReplayEngine(_toy_timeline())
    engine.reset()
    states = engine.step({"s1": {"curtailment_frac": 0.5, "sampling_rate_hz": 2.0}})

    s1 = states["s1"]
    assert s1.site_id == "s1"
    assert s1.sim_day == 1  # advanced from sim_day_solar=0 to sim_day_solar=1
    assert s1.hour_of_day == 12.0
    assert s1.power_mw == 10.0  # 20.0 * (1 - 0.5)
    assert s1.cpu_rate_sum == 1.5
    assert s1.curtailment_frac == 0.5
    assert s1.sampling_rate_hz == 2.0

    s2 = states["s2"]
    assert s2.site_id == "s2"
    assert s2.sim_day == 1  # advanced from sim_day_solar=0 to sim_day_solar=1
    assert s2.hour_of_day == 12.0
    assert s2.power_mw == 6.0  # no intervention specified -> passthrough
    assert s2.cpu_rate_sum == 2.5
    assert s2.curtailment_frac == 0.0  # default, no intervention supplied
    assert s2.sampling_rate_hz == 1.0  # default, no intervention supplied


def test_step_rejects_curtailment_frac_above_one():
    engine = ReplayEngine(_toy_timeline())
    engine.reset()
    with pytest.raises(ValueError, match="curtailment_frac"):
        engine.step({"s1": {"curtailment_frac": 1.5}})


def test_step_rejects_negative_curtailment_frac():
    engine = ReplayEngine(_toy_timeline())
    engine.reset()
    with pytest.raises(ValueError, match="curtailment_frac"):
        engine.step({"s1": {"curtailment_frac": -0.1}})


def test_step_rejects_unknown_site_id():
    engine = ReplayEngine(_toy_timeline())
    engine.reset()
    with pytest.raises(ValueError, match="unknown site"):
        engine.step({"s1": {"curtailment_frac": 0.5}, "s3": {"curtailment_frac": 0.1}})


def test_init_rejects_mismatched_site_lengths():
    timeline = pd.DataFrame({
        "site_id": ["s1", "s1", "s2"],
        "sim_day_solar": [0, 1, 0],
        "hour_of_day": [12.0, 12.0, 12.0],
        "power_mw": [10.0, 20.0, 5.0],
        "cpu_rate_sum": [1.0, 1.5, 2.0],
    })
    with pytest.raises(ValueError, match="same number of timeline ticks"):
        ReplayEngine(timeline)


def test_n_ticks_reflects_common_timeline_length():
    engine = ReplayEngine(_toy_timeline())
    assert engine.n_ticks == 2
