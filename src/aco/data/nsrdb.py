import glob

import pandas as pd

COLUMN_MAP = {
    "GHI": "ghi", "DNI": "dni", "DHI": "dhi", "Temperature": "temperature",
    "Wind Speed": "wind_speed", "Pressure": "pressure",
    "Relative Humidity": "relative_humidity",
}


def load_nsrdb_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)
    df["timestamp"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.rename(columns=COLUMN_MAP)
    return df[["timestamp"] + list(COLUMN_MAP.values())]


def build_nsrdb_table(glob_pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"no NSRDB files match {glob_pattern}")
    return pd.concat([load_nsrdb_file(f) for f in files], ignore_index=True)
