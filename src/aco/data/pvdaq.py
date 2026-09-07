import glob
import os
import re

import numpy as np
import pandas as pd

SENTINEL = -99999.0

# system_50 and system_51 both boot into the same firmware-default RTC date
# (1994-05-13) before their real clock is set -- confirmed by their pre-2011
# data starting at that exact date/time 15 minutes apart, and both resuming
# continuous real logging within days of each other in April 2011. That
# pre-resync data is discarded per system rather than by a blanket year cutoff,
# since other systems (e.g. system_10, real data from 2000) don't share it.
KNOWN_CLOCK_GLITCH_CUTOFFS = {
    50: pd.Timestamp("2011-01-01"),
    51: pd.Timestamp("2011-01-01"),
}

# PVDAQ column names carry a numeric sensor/stream id suffix (e.g. "ac_power__315")
# that changes over a system's lifetime as hardware is reconfigured, and can even
# appear twice in one file when two inverters/strings are logged concurrently.
# Canonicalizing strips that suffix so the same physical measurement lands in one
# column across years instead of exploding into hundreds of near-duplicate columns.
SUFFIX_RE = re.compile(r"^(?P<canonical>.+)__\d+$")

# PVDAQ systems that ran for many years accumulate columns from unrelated side
# experiments (per-string/per-inverter breakdowns, HVPS test-rig channels, dozens
# of extra module-temperature sensors) whose logger configuration doesn't match
# the standard inverter+sensor schema used by the rest of this project. Restricting
# to this core set keeps the schema stable across years and keeps memory bounded;
# it deliberately drops those side-experiment channels rather than carrying every
# historical variant.
CORE_COLUMNS = {
    "ac_current", "ac_power", "ac_voltage", "ambient_temp", "das_temp",
    "das_battery_voltage", "dc_pos_current", "dc_pos_voltage", "dc_power",
    "inverter_temp", "module_temp_1", "module_temp_2", "module_temp_3",
    "poa_irradiance", "power_factor",
}


# PVDAQ carries several undocumented sentinels besides -99999.0: -7999,
# -5308.9, -50001.8 and -53999 all appear on disk, so filtering by explicit
# sentinel value misses variants (and would miss any not yet observed). A
# physical plausibility range catches every one, because what they have in
# common is being impossible, not being a particular number.
#
# Lower bounds on irradiance and power are slightly negative on purpose: a
# real pyranometer reads a small negative offset at night, and an inverter
# draws a little power in standby. Clamping those to zero would erase real
# measurements to remove fake ones.
PHYSICAL_RANGES = {
    "poa_irradiance": (-5.0, 1500.0),
    "ambient_temp": (-60.0, 70.0),
    "module_temp_1": (-60.0, 110.0),
    "module_temp_2": (-60.0, 110.0),
    "module_temp_3": (-60.0, 110.0),
    "inverter_temp": (-60.0, 150.0),
    "das_temp": (-60.0, 150.0),
    "dc_power": (-100.0, 1e7),
    "ac_power": (-100.0, 1e7),
    "ac_voltage": (-10.0, 1000.0),
    "dc_pos_voltage": (-10.0, 2000.0),
    "power_factor": (-1.0, 1.0),
}


def apply_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """NaN out values outside each column's physically plausible range."""
    df = df.copy()
    for col, (lo, hi) in PHYSICAL_RANGES.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col in ("measured_on", "system_id"):
            continue
        m = SUFFIX_RE.match(col)
        rename[col] = m.group("canonical") if m else col
    df = df.rename(columns=rename)

    dup_mask = df.columns.duplicated(keep=False)
    if not dup_mask.any():
        return df

    keep = df.loc[:, ~dup_mask].copy()
    for name in df.columns[dup_mask].unique():
        keep[name] = df.loc[:, df.columns == name].mean(axis=1, skipna=True)
    return keep


def select_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if c in ("measured_on", "system_id") or c in CORE_COLUMNS]
    return df[keep_cols]


def clean_pvdaq_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["measured_on"] = pd.to_datetime(df["measured_on"])
    numeric_cols = [c for c in df.columns if c not in ("measured_on", "system_id")]
    for col in numeric_cols:
        df[col] = df[col].replace(SENTINEL, np.nan)
    df = apply_physical_ranges(df)
    df = df[(df["measured_on"].dt.year >= 1990) & (df["measured_on"].dt.year <= 2024)]

    if "system_id" in df.columns:
        for sid, cutoff in KNOWN_CLOCK_GLITCH_CUTOFFS.items():
            is_glitchy_system = df["system_id"] == sid
            df = df[~is_glitchy_system | (df["measured_on"] >= cutoff)]

    df["hour_of_day"] = df["measured_on"].dt.hour + df["measured_on"].dt.minute / 60.0
    return df.reset_index(drop=True)


def _load_and_shrink(path: str) -> pd.DataFrame:
    df = select_core_columns(canonicalize_columns(pd.read_csv(path)))
    numeric_cols = [c for c in df.columns if c not in ("measured_on", "system_id")]
    for col in numeric_cols:
        df[col] = df[col].astype("float32")
    return df


def load_system(system_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(system_dir, "year=*", "month=*", "day=*", "*.csv")))
    if not files:
        raise FileNotFoundError(f"no PVDAQ csv files under {system_dir}")
    frames = [_load_and_shrink(f) for f in files]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return clean_pvdaq_frame(combined)
