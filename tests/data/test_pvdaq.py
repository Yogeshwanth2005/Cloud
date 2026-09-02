import pandas as pd
from aco.data.pvdaq import clean_pvdaq_frame, canonicalize_columns, select_core_columns


def test_sentinel_values_become_nan():
    df = pd.DataFrame({
        "measured_on": ["2018-09-23 00:00:00", "2018-09-23 00:05:00"],
        "ac_power__423": [-9.14, -99999.0],
        "poa_irradiance__421": [4.9, -99999.0],
    })
    out = clean_pvdaq_frame(df)
    assert pd.isna(out.loc[1, "ac_power__423"])
    assert pd.isna(out.loc[1, "poa_irradiance__421"])
    assert out.loc[0, "ac_power__423"] == -9.14


def test_implausible_years_dropped():
    df = pd.DataFrame({
        "measured_on": ["1822-01-01 00:00:00", "2018-09-23 00:05:00"],
        "ac_power__423": [1.0, 2.0],
    })
    out = clean_pvdaq_frame(df)
    assert len(out) == 1
    assert out.iloc[0]["ac_power__423"] == 2.0


def test_known_clock_glitch_years_dropped_per_system():
    df = pd.DataFrame({
        "measured_on": ["1995-06-01 00:00:00", "2018-09-23 00:05:00", "1995-06-01 00:00:00"],
        "ac_power__423": [1.0, 2.0, 3.0],
        "system_id": [50, 50, 4],
    })
    out = clean_pvdaq_frame(df)
    # system_50's pre-2011 reading is a known clock-glitch artifact and is dropped ...
    assert len(out) == 2
    assert 50 not in out.loc[out["measured_on"] < "2011-01-01", "system_id"].tolist()
    # ... but the same pre-2011 date on an unaffected system is kept.
    assert (out["system_id"] == 4).any()


def test_hour_of_day_added():
    df = pd.DataFrame({
        "measured_on": ["2018-09-23 13:30:00"],
        "ac_power__423": [1.0],
    })
    out = clean_pvdaq_frame(df)
    assert out.iloc[0]["hour_of_day"] == 13.5


def test_canonicalize_columns_strips_sensor_suffix():
    df = pd.DataFrame({"measured_on": ["x"], "ac_power__315": [10.0], "system_id": [4]})
    out = canonicalize_columns(df)
    assert list(out.columns) == ["measured_on", "ac_power", "system_id"]


def test_canonicalize_columns_averages_duplicate_canonical_columns():
    df = pd.DataFrame({
        "measured_on": ["x"],
        "ac_power__315": [10.0],
        "ac_power__375": [20.0],
        "system_id": [4],
    })
    out = canonicalize_columns(df)
    assert out.loc[0, "ac_power"] == 15.0


def test_select_core_columns_drops_side_experiment_channels():
    df = pd.DataFrame({
        "measured_on": ["x"], "system_id": [4], "ac_power": [1.0],
        "poa_irradiance": [2.0], "hvps_1-1_current": [3.0], "module_temp_9": [4.0],
    })
    out = select_core_columns(df)
    assert set(out.columns) == {"measured_on", "system_id", "ac_power", "poa_irradiance"}
