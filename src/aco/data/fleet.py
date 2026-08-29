import glob
import os
import re

import pandas as pd

FILENAME_RE = re.compile(
    r"^(?P<kind>Actual|DA|HA4)_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)_(?P<year>\d{4})_"
    r"(?P<plant_type>UPV|DPV)_(?P<capacity>\d+(?:\.\d+)?)MW_(?P<resolution>\d+)_Min\.csv$"
)


def parse_plant_filename(filename: str) -> dict:
    m = FILENAME_RE.match(os.path.basename(filename))
    if not m:
        raise ValueError(f"filename does not match expected pattern: {filename}")
    return {
        "kind": m.group("kind"),
        "lat": float(m.group("lat")),
        "lon": float(m.group("lon")),
        "year": int(m.group("year")),
        "plant_type": m.group("plant_type"),
        "capacity_mw": float(m.group("capacity")),
        "resolution_min": int(m.group("resolution")),
    }


def build_fleet_tables(state_dirs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    plant_rows = []
    power_frames = []
    for region, directory in state_dirs.items():
        for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
            meta = parse_plant_filename(path)
            plant_id = f"{region}_{meta['lat']}_{meta['lon']}_{meta['plant_type']}_{meta['capacity_mw']}MW"
            plant_rows.append({
                "plant_id": plant_id, "region": region, "lat": meta["lat"],
                "lon": meta["lon"], "plant_type": meta["plant_type"],
                "capacity_mw": meta["capacity_mw"],
            })
            df = pd.read_csv(path)
            df.columns = ["timestamp", "power_mw"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M")
            df["power_mw"] = df["power_mw"].astype("float32")
            df["plant_id"] = plant_id
            df["kind"] = meta["kind"]
            power_frames.append(df)
    plants_df = pd.DataFrame(plant_rows).drop_duplicates(subset=["plant_id"]).reset_index(drop=True)
    power_df = pd.concat(power_frames, ignore_index=True)
    power_df["plant_id"] = power_df["plant_id"].astype("category")
    power_df["kind"] = power_df["kind"].astype("category")
    power_df["hour_of_day"] = (
        power_df["timestamp"].dt.hour + power_df["timestamp"].dt.minute / 60.0
    ).astype("float32")
    return plants_df, power_df
