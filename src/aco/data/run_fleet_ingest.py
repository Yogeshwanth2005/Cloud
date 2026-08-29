import os

from aco.data.fleet import build_fleet_tables

STATE_DIRS = {
    "Arizona": "Arizona", "California": "California",
    "Colorado": "Colarado", "Nevada": "Nevada",
}


def main():
    out_dir = os.path.join("fleet_data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    plants_df, power_df = build_fleet_tables(STATE_DIRS)
    plants_df.to_parquet(os.path.join(out_dir, "plants.parquet"), index=False)
    power_df.to_parquet(os.path.join(out_dir, "power_5min.parquet"), index=False)
    print(f"plants: {len(plants_df):,} rows, power: {len(power_df):,} rows")


if __name__ == "__main__":
    main()
