"""Task 5.2 Step 5 -- empirically validate the VoI uncertainty-reduction proxy.

Checks whether select_best_intervention's pval-based proxy points at the same
node that a real refit-on-more-data actually improved most. This is the
reproducible check behind runs/validation/voi_proxy_check.json: run it to
regenerate the report.

    python -m aco.interventions.run_voi_proxy_check
"""
import json
import warnings
from pathlib import Path

import pandas as pd

from aco.causal.graph import fit_observational_graph
from aco.causal.world_model import CausalWorldModel
from aco.interventions.voi import select_best_intervention

SEED = 0
SOURCE = Path("pvdaq_data/processed/system_51_weather.parquet")
OUT = Path("runs/validation/voi_proxy_check.json")
VAR_NAMES = ["poa_irradiance", "module_temp", "ambient_temp", "dc_power", "ac_power"]
TAU_MAX = 1
N_EARLY_DAYS = 3
N_LATE_TARGET = 8000

# Since interventions declare a `target_var` (the variable they actually
# manipulate), none of the four registered interventions can reach any Tier-1
# physical variable: you cannot curtail irradiance, module temperature or
# ambient temperature. select_best_intervention therefore returns None here by
# construction rather than by accident, and this check is vacuous until the
# library gains an intervention that manipulates a physical node. Recorded in
# the report so a reader does not misread None as "nothing was worth probing".
COMPATIBILITY_NOTE = (
    "No registered intervention declares a target_var among VAR_NAMES (the Tier-1 physical "
    "variables), so no (node, intervention) pair is admissible and voi_proxy_selected is None "
    "by construction. This check cannot discriminate proxy quality until the safe intervention "
    "library covers a variable these physical nodes actually expose."
)

CAVEAT = (
    "The late-window graph contains edges whose orientation is physically implausible "
    "(e.g. dc_power -> poa_irradiance, ac_power -> module_temp), consistent with the same "
    "downsampling-induced temporal aliasing documented in the Task 4.2 clipping report: an "
    "~8000-row stride sample over an 8-year, per-minute series jumps many hours between "
    "consecutive rows, which can distort PCMCI+'s lag-based orientation. The proxy-vs-empirical "
    "agreement reported here should therefore be read as 'the proxy points at the node whose "
    "graph connectivity changed most between the two fits' rather than 'the proxy tracks a "
    "verified physical uncertainty reduction' -- a stronger version of this check would refit on "
    "a properly time-resampled (not strided) series, which was out of scope for this pass."
)


def _avg_pval(graph, node) -> float:
    pvals = [d["pval"] for u, v, d in graph.edges(data=True) if u == node or v == node]
    return sum(pvals) / len(pvals) if pvals else 1.0


def _edge_list(graph) -> list:
    return [{"u": u, "v": v, "pval": d["pval"], "lag": d["lag"]} for u, v, d in graph.edges(data=True)]


def main() -> None:
    # system_51_weather.parquet has no single "module_temp" column -- CORE_COLUMNS
    # (aco.data.pvdaq) keeps module_temp_1/2/3 as three distinct real sensors
    # rather than canonicalizing them into one, since they aren't duplicate
    # channels of the same logical variable. NODE_SCHEMA's single "module_temp"
    # name is reconciled with that here the same way canonicalize_columns
    # averages duplicate channels elsewhere in the pipeline.
    raw_cols = ["measured_on", "module_temp_1", "module_temp_2", "module_temp_3"]
    raw_cols += [c for c in VAR_NAMES if c != "module_temp"]
    df = pd.read_parquet(SOURCE, columns=raw_cols)
    df["module_temp"] = df[["module_temp_1", "module_temp_2", "module_temp_3"]].mean(axis=1)

    df = df[df["measured_on"].dt.year >= 2015].dropna(subset=VAR_NAMES)
    df = df.sort_values("measured_on").reset_index(drop=True)

    # Narrowed to N_EARLY_DAYS (not a longer window) because an initial ~90-day
    # attempt (~4000 rows) already drove every node's pval to near machine
    # epsilon -- relationships this strong saturate fast, leaving no real
    # uncertainty gradient for the proxy to be tested against.
    early_dates = sorted(df["measured_on"].dt.date.unique())[:N_EARLY_DAYS]
    early = df[df["measured_on"].dt.date.isin(early_dates)].reset_index(drop=True)

    # A stride sample (not the literal full window) over the 8-year series --
    # loading and fitting PCMCI+ on the whole per-minute range isn't tractable
    # here, and the resulting temporal-aliasing caveat is disclosed below.
    stride = max(1, len(df) // N_LATE_TARGET)
    late = df.iloc[::stride].reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        early_graph = fit_observational_graph(early, var_names=VAR_NAMES, tau_max=TAU_MAX)
        late_graph = fit_observational_graph(late, var_names=VAR_NAMES, tau_max=TAU_MAX)

    early_uncertainty = {node: _avg_pval(early_graph, node) for node in VAR_NAMES}
    late_uncertainty = {node: _avg_pval(late_graph, node) for node in VAR_NAMES}
    pval_improvement = {node: early_uncertainty[node] - late_uncertainty[node] for node in VAR_NAMES}

    best_improvement = max(pval_improvement.values())
    empirically_best_nodes_tied = [n for n, v in pval_improvement.items() if v == best_improvement]

    model = CausalWorldModel(early_graph)
    model.fit(early)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        best = select_best_intervention(
            model, early, early_graph, node_candidates=VAR_NAMES, var_names=VAR_NAMES, tau_max=TAU_MAX,
        )

    voi_proxy_selected = list(best) if best is not None else None
    agrees = bool(best is not None and best[0] in empirically_best_nodes_tied)

    report = {
        "system": "system_51",
        "window": f"2015-2023 (expanding: early=first {N_EARLY_DAYS} days, late=full window)",
        "note_on_window_choice": (
            "An initial attempt used a 90-day early window (~4000 rows). At that size, every "
            "node's pval was already near machine epsilon (relationships this strong saturate "
            "fast), leaving no real uncertainty gradient -- select_best_intervention correctly "
            f"returned None (nothing worth probing) rather than disagreeing with a meaningful "
            f"empirical signal. Narrowed to {N_EARLY_DAYS} days specifically to get non-degenerate "
            "uncertainty."
        ),
        "var_names": VAR_NAMES,
        "n_rows_early_sample": int(len(early)),
        "n_rows_late_sample": int(len(late)),
        "early_graph_edges": _edge_list(early_graph),
        "late_graph_edges": _edge_list(late_graph),
        "early_uncertainty_by_node": early_uncertainty,
        "late_uncertainty_by_node": late_uncertainty,
        "pval_improvement_by_node": pval_improvement,
        "empirically_best_nodes_tied": empirically_best_nodes_tied,
        "voi_proxy_selected": voi_proxy_selected,
        "voi_proxy_agrees_with_empirical_best": agrees,
        "compatibility_note": COMPATIBILITY_NOTE,
        "caveat": CAVEAT,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
