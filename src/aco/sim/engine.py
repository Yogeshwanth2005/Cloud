from dataclasses import dataclass
import pandas as pd


@dataclass
class SiteState:
    site_id: str
    sim_day: int
    hour_of_day: float
    power_mw: float
    cpu_rate_sum: float
    curtailment_frac: float = 0.0
    sampling_rate_hz: float = 1.0


class ReplayEngine:
    def __init__(self, timeline_df: pd.DataFrame):
        # site_timeline.parquet carries two independent day-indices (sim_day_solar,
        # sim_day_cluster) because the solar and cluster sources don't share a
        # calendar -- see Phase 1.5. sim_day_solar drives the tick (it spans the
        # full 365-day solar year); sim_day_cluster only ever has as many distinct
        # values as the downloaded cluster-trace window and is context, not a tick axis.
        self.timeline = timeline_df.sort_values(["site_id", "sim_day_solar", "hour_of_day"]).reset_index(drop=True)
        self._cursors = {sid: 0 for sid in self.timeline["site_id"].unique()}
        self._by_site = {sid: g.reset_index(drop=True) for sid, g in self.timeline.groupby("site_id")}

    def reset(self) -> dict:
        self._cursors = {sid: 0 for sid in self._by_site}
        return self._current_states({})

    def step(self, interventions: dict) -> dict:
        for sid in self._cursors:
            self._cursors[sid] = min(self._cursors[sid] + 1, len(self._by_site[sid]) - 1)
        return self._current_states(interventions)

    def _current_states(self, interventions: dict) -> dict:
        out = {}
        for sid, df in self._by_site.items():
            row = df.iloc[self._cursors[sid]]
            iv = interventions.get(sid, {})
            curtailment = iv.get("curtailment_frac", 0.0)
            out[sid] = SiteState(
                site_id=sid, sim_day=int(row["sim_day_solar"]), hour_of_day=float(row["hour_of_day"]),
                power_mw=float(row["power_mw"]) * (1 - curtailment),
                cpu_rate_sum=float(row["cpu_rate_sum"]),
                curtailment_frac=curtailment,
                sampling_rate_hz=iv.get("sampling_rate_hz", 1.0),
            )
        return out
