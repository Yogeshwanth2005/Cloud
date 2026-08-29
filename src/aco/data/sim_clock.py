import numpy as np
import pandas as pd


def to_sim_clock(df: pd.DataFrame, time_col: str, epoch: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["sim_day"] = (df[time_col] - epoch).dt.days
    df["hour_of_day"] = df[time_col].dt.hour + df[time_col].dt.minute / 60.0
    return df


def build_site_timeline(power_df, plants_df, machine_util_df, n_sites: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sites = (
        plants_df.groupby("region", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, n_sites // plants_df["region"].nunique())), random_state=seed))
        .head(n_sites)
    )
    machine_ids = machine_util_df["machine_id"].unique()
    rng.shuffle(machine_ids)
    machine_blocks = np.array_split(machine_ids, len(sites))

    power_epoch = pd.Timestamp("2006-01-01")
    cluster_epoch = machine_util_df["wall_time"].min().normalize()

    actual = power_df[power_df["kind"] == "Actual"].copy()
    actual = to_sim_clock(actual, "timestamp", power_epoch)
    mu = to_sim_clock(machine_util_df, "wall_time", cluster_epoch)

    rows = []
    for (_, site_row), machines in zip(sites.iterrows(), machine_blocks):
        site_power = actual[actual["plant_id"] == site_row["plant_id"]]
        site_mu = (
            mu[mu["machine_id"].isin(machines)]
            .groupby(["sim_day", "hour_of_day"], as_index=False)["cpu_rate_sum"].sum()
        )
        merged = pd.merge(site_power, site_mu, on="hour_of_day", suffixes=("_solar", "_cluster"))
        merged["site_id"] = site_row["plant_id"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)
