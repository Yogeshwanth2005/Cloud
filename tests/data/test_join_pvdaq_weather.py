import pandas as pd
from aco.data.join_pvdaq_weather import join_nearest_hour


def test_join_matches_same_hour():
    pvdaq = pd.DataFrame({
        "measured_on": pd.to_datetime(["2018-09-23 00:01:00", "2018-09-23 00:04:00"]),
        "ac_power__423": [1.0, 2.0],
    })
    nsrdb = pd.DataFrame({
        "timestamp": pd.to_datetime(["2018-09-23 00:00:00", "2018-09-23 01:00:00"]),
        "ghi": [0, 100],
    })
    out = join_nearest_hour(pvdaq, nsrdb)
    assert out["ghi"].tolist() == [0, 0]


def test_join_leaves_nan_outside_coverage():
    pvdaq = pd.DataFrame({
        "measured_on": pd.to_datetime(["2000-01-01 00:00:00"]),
        "ac_power__423": [1.0],
    })
    nsrdb = pd.DataFrame({
        "timestamp": pd.to_datetime(["2018-09-23 00:00:00"]),
        "ghi": [0],
    })
    out = join_nearest_hour(pvdaq, nsrdb)
    assert pd.isna(out.iloc[0]["ghi"])
