import os

import pandas as pd

from aco.data.join_pvdaq_weather import join_nearest_hour

SYSTEMS = ["system_50", "system_51"]
DATA_ROOT = "pvdaq_data"
NSRDB_PATH = os.path.join("nsrdb_golden", "processed", "nsrdb_golden.parquet")


def main():
    out_dir = os.path.join(DATA_ROOT, "processed")
    nsrdb_df = pd.read_parquet(NSRDB_PATH)
    for sys_name in SYSTEMS:
        pvdaq_df = pd.read_parquet(os.path.join(out_dir, f"{sys_name}.parquet"))
        joined = join_nearest_hour(pvdaq_df, nsrdb_df)
        out_path = os.path.join(out_dir, f"{sys_name}_weather.parquet")
        joined.to_parquet(out_path, index=False)
        print(f"  -> {out_path}: {len(joined):,} rows")


if __name__ == "__main__":
    main()
