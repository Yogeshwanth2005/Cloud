import pandas as pd


def join_nearest_hour(pvdaq_df: pd.DataFrame, nsrdb_df: pd.DataFrame) -> pd.DataFrame:
    left = pvdaq_df.sort_values("measured_on").reset_index(drop=True)
    right = nsrdb_df.sort_values("timestamp").reset_index(drop=True)
    return pd.merge_asof(
        left, right, left_on="measured_on", right_on="timestamp",
        direction="nearest", tolerance=pd.Timedelta("30min"),
    )
