import pandas as pd
from aco.data.sim_clock import to_sim_clock, build_site_timeline


def test_to_sim_clock_computes_day_offset():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2006-01-03 06:30:00"])})
    out = to_sim_clock(df, "timestamp", epoch=pd.Timestamp("2006-01-01"))
    assert out.iloc[0]["sim_day"] == 2
    assert out.iloc[0]["hour_of_day"] == 6.5


def test_build_site_timeline_assigns_disjoint_machines():
    plants_df = pd.DataFrame({
        "plant_id": ["p1", "p2"], "region": ["Arizona", "California"],
    })
    power_df = pd.DataFrame({
        "plant_id": ["p1", "p1", "p2", "p2"], "kind": ["Actual"] * 4,
        "timestamp": pd.to_datetime(["2006-01-01 06:00", "2006-01-01 07:00"] * 2),
        "power_mw": [1.0, 2.0, 3.0, 4.0],
    })
    machine_util_df = pd.DataFrame({
        "machine_id": [10, 10, 20, 20],
        "wall_time": pd.to_datetime(["2011-05-01 06:00", "2011-05-01 07:00"] * 2),
        "cpu_rate_sum": [0.5, 0.6, 0.7, 0.8],
    })
    out = build_site_timeline(power_df, plants_df, machine_util_df, n_sites=2, seed=0)
    assert set(out["site_id"].unique()) == {"p1", "p2"}
    assert out["power_mw"].notna().all()
    assert out["cpu_rate_sum"].notna().all()


def test_build_site_timeline_does_not_cross_join_across_cluster_days():
    plants_df = pd.DataFrame({"plant_id": ["p1"], "region": ["Arizona"]})
    power_df = pd.DataFrame({
        "plant_id": ["p1", "p1"], "kind": ["Actual", "Actual"],
        "timestamp": pd.to_datetime(["2006-01-01 06:00", "2006-01-02 06:00"]),
        "power_mw": [1.0, 2.0],
    })
    # Two distinct cluster days that both happen to have a 06:00 sample --
    # a join on hour_of_day alone would match every solar row to both
    # cluster days and duplicate every timestamp.
    machine_util_df = pd.DataFrame({
        "machine_id": [10, 10],
        "wall_time": pd.to_datetime(["2011-05-01 06:00", "2011-05-02 06:00"]),
        "cpu_rate_sum": [0.5, 0.9],
    })
    out = build_site_timeline(power_df, plants_df, machine_util_df, n_sites=1, seed=0)
    assert len(out) == len(power_df)
    assert not out.duplicated(subset=["site_id", "timestamp"]).any()
