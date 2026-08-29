import os
import time

from aco.data.pvdaq import load_system

SYSTEMS = ["system_4", "system_10", "system_50", "system_51", "system_1283"]
DATA_ROOT = "pvdaq_data"


def main():
    out_dir = os.path.join(DATA_ROOT, "processed")
    os.makedirs(out_dir, exist_ok=True)
    for sys_name in SYSTEMS:
        t0 = time.time()
        print(f"Loading {sys_name} ...", flush=True)
        df = load_system(os.path.join(DATA_ROOT, sys_name))
        out_path = os.path.join(out_dir, f"{sys_name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  -> {out_path}: {len(df):,} rows ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
