import numpy as np
import pandas as pd

from aco.causal.validate_world_model import (
    efficiency_by_power_bin,
    has_clipping_plateau,
    label_clipping_events,
)


def test_label_clipping_events_flags_saturation():
    df = pd.DataFrame({
        "ac_power": [10.0, 49.5, 50.0, 30.0],
        "dc_power": [10.5, 55.0, 60.0, 31.0],
    })
    flags = label_clipping_events(df, "ac_power", "dc_power", rated_kw=50.0)
    assert flags.tolist() == [False, True, True, False]


def test_efficiency_by_power_bin_ignores_part_load_rows():
    # Efficiency is poor at low power and good at high power. Binning over the
    # upper half of the dc range must not be dragged down by the part-load rows.
    rng = np.random.default_rng(0)
    dc = np.concatenate([rng.uniform(10, 200, 5000), rng.uniform(3000, 8000, 5000)])
    ac = np.where(dc < 500, 0.5 * dc, 0.92 * dc)
    df = pd.DataFrame({"ac_power": ac, "dc_power": dc})
    eff = efficiency_by_power_bin(df, "ac_power", "dc_power", n_bins=5)
    assert (eff > 0.9).all()


def test_has_clipping_plateau_detects_a_real_cap():
    # ac saturates at a hard 5000 W cap while dc keeps rising -> efficiency
    # must fall away across the upper dc range.
    rng = np.random.default_rng(0)
    dc = rng.uniform(100, 8000, 200000)
    ac = np.minimum(0.92 * dc, 5000.0) + rng.normal(0, 5, len(dc))
    df = pd.DataFrame({"ac_power": ac, "dc_power": dc})
    assert has_clipping_plateau(df, "ac_power", "dc_power") is True


def test_has_clipping_plateau_false_when_efficiency_is_flat():
    # No cap: efficiency stays constant to the largest dc values. This is the
    # system_51 case -- the natural experiment Task 4.2 assumed is absent.
    rng = np.random.default_rng(0)
    dc = rng.uniform(100, 8000, 200000)
    ac = 0.92 * dc + rng.normal(0, 5, len(dc))
    df = pd.DataFrame({"ac_power": ac, "dc_power": dc})
    assert has_clipping_plateau(df, "ac_power", "dc_power") is False
