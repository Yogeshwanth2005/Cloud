import pandas as pd


def label_clipping_events(
    df: pd.DataFrame, ac_col: str, dc_col: str, rated_kw: float, tolerance: float = 0.02
) -> pd.Series:
    """Flag rows where the inverter is saturated (clipping).

    True where ac_power has reached the inverter's rated capacity while
    dc_power still exceeds what that capacity would be at typical (~0.96)
    inverter efficiency -- i.e. dc_power kept rising with irradiance but
    ac_power plateaued because the inverter is capped, not because dc_power
    stopped growing.
    """
    ac_saturated = df[ac_col] >= rated_kw * (1 - tolerance)
    dc_available = df[dc_col] >= rated_kw * 0.96 * (1 - tolerance)
    return ac_saturated & dc_available
