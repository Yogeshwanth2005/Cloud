import os

import pandas as pd

from aco.config import load_config
from aco.data.sim_clock import build_site_timeline

N_SITES = 20


def main():
    cfg = load_config(os.path.join("configs", "base.yaml"))

    plants_df = pd.read_parquet(os.path.join("fleet_data", "processed", "plants.parquet"))
    power_df = pd.read_parquet(os.path.join("fleet_data", "processed", "power_5min.parquet"))
    machine_util_df = pd.read_parquet(
        os.path.join("google_cluster_2011", "processed", "machine_utilization_5min.parquet")
    )

    timeline_df = build_site_timeline(power_df, plants_df, machine_util_df, N_SITES, cfg["seed"])

    out_path = os.path.join("fleet_data", "processed", "site_timeline.parquet")
    timeline_df.to_parquet(out_path, index=False)
    print(f"site_timeline: {len(timeline_df):,} rows, {timeline_df['site_id'].nunique()} sites -> {out_path}")


if __name__ == "__main__":
    main()
