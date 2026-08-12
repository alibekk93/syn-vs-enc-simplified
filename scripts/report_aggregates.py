"""
Print the manuscript's aggregate number set as markdown, straight from results/.

Why this exists: every table in tmp/manuscript/ and tmp/overleaf/main.tex used to be
typed by hand from the CSVs. That is how the cost section ended up quoting wall-clock
seconds while the figures plotted CPU seconds, and how mean-based prose ended up next
to median-based panels. Nothing in a markdown file should be a number a script can
produce; run this instead and paste, or read it and write prose around it.

Conventions, identical to src/utils.aggregate_metrics_csv and every figure:
  * point estimate  = MEDIAN of the bootstrap replicates within a (dataset, mode,
                      classifier) cell, then the MEDIAN over cells;
  * interval        = 2.5/97.5 percentile of the replicates, aggregated across cells
                      as the median of each bound (a typical width, NOT a CI);
  * paired gap      = MEDIAN of the replicate-wise differences, which is the location
                      the sign-based ASL in src/stats_tests.py actually tests. Note it
                      is not median_a - median_b: medians do not subtract;
  * cost            = CPU seconds (training_cpu_time / cpu_per_sample), the
                      contention-robust basis, never wall clock.

Usage:
    python scripts/report_aggregates.py            # everything
    python scripts/report_aggregates.py utility    # one section (see SECTIONS)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats_tests import _load_replicates
from src.utils import parse_filename_metadata
from src.visualization import (
    _load_viz_config,
    _radar_cfg,
    _radar_select_modes,
    load_simple_bootstrap,
)

# gestational_diabetes was run but its FHE leg never completed, so it is out of
# manuscript scope and appears only as a Discussion example of deployment fragility.
EXCLUDED_DATASETS = ("gestational_diabetes",)

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
MODELS = ["logistic_regression", "random_forest", "xgboost"]
GENERATORS = ["arf", "gaussian_copula", "nflow", "bayesian_network", "ctgan"]
SCALES = [100, 150, 200, 250, 300]
N_BITS = [2, 4, 6, 8, 10, 12]
# The two operating points the mode comparison is pinned to. Nothing is ever pooled
# across synthesis scales or across precisions; the sweeps are reported per level.
PRIMARY_MODES = ["standard", "fhe_8"] + [f"{g}_100" for g in GENERATORS]


def _select(df, key):
    """
    Rows for one canonical mode key ("standard", "arf_100", "fhe_8").

    load_simple_bootstrap splits the scale/precision suffix out of `mode` into its own
    column, so a plain `df["mode"] == "arf_100"` silently matches nothing. Reuse the
    one parser rather than re-deriving the split here.
    """
    meta = parse_filename_metadata(key)
    sub = df[df["mode"] == meta["mode"]]
    for column in ("synth_scale", "n_bits"):
        if meta[column] is not None:
            sub = sub[sub[column] == meta[column]]
    return sub


def _table(header, rows):
    """One GitHub pipe table."""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _cells(df, keys, column):
    """Median of `column` per cell, as a Series indexed by `keys`."""
    return df.groupby(keys)[column].median()


def _f(x, n=3):
    # ASCII only: this prints to a Windows console under cp1252, where a dash or an
    # arrow raises UnicodeEncodeError and kills the whole report mid-table.
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


# ---------------------------------------------------------------- sections


def utility(df, **_):
    """Baselines by classifier, and the mode-level table across all five metrics."""
    print("## Real-data baselines (median over the 6 datasets)\n")
    std = _select(df, "standard")
    rows = []
    for model in MODELS:
        per_dataset = _cells(std[std["model"] == model], ["dataset"], "roc_auc")
        rows.append([model, _f(per_dataset.median()),
                     f"{_f(per_dataset.min())} to {_f(per_dataset.max())}"])
    overall = _cells(std, ["dataset", "model"], "roc_auc")
    rows.append(["**all**", f"**{_f(overall.median())}**", ""])
    print(_table(["classifier", "ROC-AUC", "range over datasets"], rows), "\n")

    print("## Prediction performance by mode (median over the 18 cells)\n")
    print("Generators at the 100% synthesis scale, FHE at 8 bits. Interval columns are "
          "the median of the cell-level 2.5/97.5 percentile bounds: a typical width, "
          "not a confidence interval for the aggregate.\n")
    rows = []
    for mode in PRIMARY_MODES:
        sub = _select(df, mode)
        if sub.empty:
            continue
        cells = []
        for metric in METRICS:
            per_cell = sub.groupby(["dataset", "model"])[metric]
            cells.append(f"{_f(per_cell.median().median())} "
                         f"({_f(per_cell.quantile(0.025).median())} to "
                         f"{_f(per_cell.quantile(0.975).median())})")
        rows.append([mode] + cells)
    rows.sort(key=lambda r: float(r[-1].split()[0]), reverse=True)
    print(_table(["mode"] + METRICS, rows), "\n")

    print("## ROC-AUC by dataset and by classifier\n")
    for keys, title in ((["dataset"], "dataset"), (["model"], "classifier")):
        levels = sorted(df[keys[0]].dropna().unique())
        rows = []
        for mode in PRIMARY_MODES:
            sub = _select(df, mode)
            if sub.empty:
                continue
            per = _cells(sub, keys + (["model"] if keys == ["dataset"] else ["dataset"]),
                         "roc_auc").groupby(level=0).median()
            rows.append([mode] + [_f(per.get(v)) for v in levels])
        print(f"By {title}:\n")
        print(_table(["mode"] + levels, rows), "\n")


def sweeps(df, **_):
    """Synthesis scale and quantization precision, each level reported on its own."""
    print("## Synthesis scale (ROC-AUC, median over the 18 cells)\n")
    rows = []
    for gen in GENERATORS:
        vals = [_cells(_select(df, f"{gen}_{s}"), ["dataset", "model"],
                       "roc_auc").median() for s in SCALES]
        rows.append([gen] + [_f(v) for v in vals] + [f"{vals[-1] - vals[0]:+.3f}"])
    print(_table(["generator"] + [f"{s}%" for s in SCALES] + ["100 to 300"], rows), "\n")

    print("## Quantization precision (median over the 6 datasets)\n")
    rows = []
    for nb in N_BITS:
        sub = _select(df, f"fhe_{nb}")
        cells = []
        for model in MODELS:
            m = sub[sub["model"] == model]
            cells.append(_f(_cells(m, ["dataset"], "roc_auc").median()))
            cells.append(_f(_cells(m, ["dataset"], "inf_time_per_sample").median(), 4))
        rows.append([nb] + cells)
    plain = _select(df, "standard")
    rows.append(["plaintext"] + [
        v for model in MODELS for v in (
            _f(_cells(plain[plain["model"] == model], ["dataset"], "roc_auc").median()),
            _f(_cells(plain[plain["model"] == model], ["dataset"],
                      "inf_time_per_sample").median(), 6))
    ])
    print(_table(["n_bits"] + [f"{m} {c}" for m in MODELS for c in ("ROC-AUC", "s/sample")],
                 rows), "\n")


def cost(df, **_):
    """System cost by mode, in CPU seconds, for the two pinned operating points."""
    print("## Cost by mode (median over the 18 cells, CPU seconds)\n")
    print("One-time cost is everything paid before the first prediction: classifier "
          "training for standard, generator fit + sampling + classifier training for "
          "the synthetic modes, model fit + circuit compilation for FHE. Means are "
          "shown alongside only to expose the contention outliers; the median is the "
          "reported number.\n")
    rcfg = _radar_cfg(_load_viz_config())
    rows = []
    for key, sub in _radar_select_modes(df, rcfg):
        cells = sub.groupby(["dataset", "model"])
        one_time = cells["train_time"].median()
        rows.append([
            key,
            _f(one_time.median(), 2),
            _f(one_time.mean(), 2),
            f"{cells['inf_time_per_sample'].median().median():.3g}",
            _f(cells["mem_inf_peak"].median().median(), 0),
            _f(cells["model_size_mb"].median().median(), 2),
        ])
    print(_table(["mode", "one-time (s)", "[mean]", "per-query (s/sample)",
                  "peak inf. mem (MB)", "artifact (MB)"], rows), "\n")


def gaps(_df, replicates=None, **__):
    """Paired median differences against the real-data baseline."""
    print("## Median paired gap to real data (positive = the privacy mode is worse)\n")
    print("Median of the replicate-wise differences, per cell, then the median over "
          "cells. Not median(real) minus median(mode): medians do not subtract. The "
          "significance verdicts live in results/stats/ and are unaffected by this "
          "choice of point estimate.\n")
    cells = sorted({(d, m) for d, m, mode in replicates if mode == "standard"})
    rows = []
    for mode in PRIMARY_MODES[1:]:
        per_cell = {}
        for dataset, model in cells:
            a = replicates.get((dataset, model, "standard"))
            b = replicates.get((dataset, model, mode))
            if a is None or b is None or len(a) != len(b):
                continue
            d = a - b
            d = d[np.isfinite(d)]
            if d.size:
                per_cell[(dataset, model)] = float(np.median(d))
        if not per_cell:
            continue
        vals = pd.Series(per_cell)
        worst = vals.idxmax()
        rows.append([mode, _f(vals.median()), _f(vals.min()), _f(vals.max()),
                     f"{worst[0]}/{worst[1]}"])
    print(_table(["mode", "median gap", "best cell", "worst cell", "worst cell is"],
                 rows), "\n")


SECTIONS = {"utility": utility, "sweeps": sweeps, "cost": cost, "gaps": gaps}


def main(argv):
    wanted = argv[1:] or list(SECTIONS)
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"Unknown section(s) {unknown}; choose from {list(SECTIONS)}")

    df = load_simple_bootstrap()
    df = df[~df["dataset"].isin(EXCLUDED_DATASETS)]
    replicates = {k: v for k, v in _load_replicates("results/metrics", "roc_auc").items()
                  if k[0] not in EXCLUDED_DATASETS}

    print("# Aggregate report (medians, CPU seconds)\n")
    for name in wanted:
        SECTIONS[name](df, replicates=replicates)


if __name__ == "__main__":
    main(sys.argv)
