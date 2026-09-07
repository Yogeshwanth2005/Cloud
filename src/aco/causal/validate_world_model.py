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


def efficiency_by_power_bin(
    df: pd.DataFrame, ac_col: str, dc_col: str, n_bins: int = 10
) -> pd.Series:
    """Mean ac/dc efficiency per dc_power bin, over the upper half of the range.

    Restricting to the upper half is what makes this diagnostic meaningful.
    A PV inverter's efficiency is genuinely poor at part load (~0.71 at 100-500 W
    on system_51, vs ~0.92 above 4 kW), so any statistic that pools high-power
    rows against *all* other rows is dominated by that part-load curve and will
    report an efficiency gap whether or not clipping exists.
    """
    d = df[[ac_col, dc_col]].dropna()
    d = d[d[dc_col] > 0]
    d = d[d[dc_col] > d[dc_col].quantile(0.5)]
    bins = pd.qcut(d[dc_col], n_bins, duplicates="drop")
    return d.groupby(bins, observed=True).apply(
        lambda g: float((g[ac_col] / g[dc_col]).mean())
    )


def has_clipping_plateau(
    df: pd.DataFrame, ac_col: str, dc_col: str, min_drop: float = 0.05
) -> bool:
    """True if ac/dc efficiency falls across the upper dc_power range.

    This is the actual signature of inverter clipping: once ac_power is capped,
    ac/dc = cap/dc and decays as dc keeps rising. A system that never saturates
    holds a flat (or mildly rising) efficiency all the way to its largest
    recorded dc_power. Compares the first and last upper-range bin against
    `min_drop` rather than asserting a threshold on the absolute level, since
    the level itself is system-specific.
    """
    eff = efficiency_by_power_bin(df, ac_col, dc_col)
    if len(eff) < 2:
        return False
    return bool(eff.iloc[0] - eff.iloc[-1] > min_drop)
