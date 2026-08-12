"""
Print the manuscript's LaTeX table bodies straight from results/.

Same reason report_aggregates.py exists: every table in tmp/overleaf/main.tex used to be
typed by hand from the CSVs, which is how mean-based cells ended up beside median-based
figure panels and wall-clock costs beside CPU seconds. This emits the body rows only, no
table environment and no caption, ready to paste under the existing headers.

Conventions, identical to report_aggregates.py and every figure:
  * point estimate  = MEDIAN. Cell-level values are the {metric}_median columns of
                      results/metrics_aggregated.csv; aggregates are the median over cells;
  * interval        = the cell's own 2.5/97.5 percentiles ({metric}_ci_low/_ci_high). Where
                      a table aggregates cells, the interval column is the MEDIAN of the
                      cell-level bounds: a typical width, NOT a CI for the aggregate;
  * paired gap      = median_diff in results/stats/*.csv, the median of the replicate-wise
                      differences. Not median_a - median_b: medians do not subtract;
  * cost            = CPU seconds (training_cpu_time / cpu_per_sample), never wall clock;
  * nothing is pooled across synthesis scales or across quantization precisions.

Usage:
    python scripts/render_tables.py                 # every body (see SECTIONS)
    python scripts/render_tables.py s4contrasts     # one
"""

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_aggregates import EXCLUDED_DATASETS, GENERATORS, MODELS, N_BITS
from src.utils import parse_filename_metadata
from src.visualization import _load_viz_config, _radar_cfg, _radar_synth_offsets

# _radar_synth_offsets re-globs all 762 profile JSONs per call and tab:cost needs it twelve
# times (once to sort the generators, once to render each row).
_offsets = lru_cache(_radar_synth_offsets)

AGGREGATED = "results/metrics_aggregated.csv"
STATS_DIR = "results/stats"

# main.tex orders the metric columns Accuracy, Precision, Recall, F1, ROC-AUC, which is not
# the storage order in the CSV.
TABLE_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# (full name, abbreviation) exactly as main.tex spells them. Ordered by dataset size, as
# tab:s1data and tab:s2baseline are; config/datasets.yaml is in a different order.
DATASETS = {
    "cardiotocography": ("Cardiotocography", "Cardio."),
    "maternal_health_risk": ("Maternal health risk", "Maternal"),
    "mammographic_mass": ("Mammographic mass", "Mammo."),
    "diabetes": ("Diabetes", "Diabetes"),
    "heart_disease": ("Heart disease", "Heart dis."),
    "heart_failure": ("Heart failure", "Heart fail."),
}
CLASSIFIERS = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGBoost"}
# (row label, contrast abbreviation) for the generators.
GENERATOR_LABELS = {
    "arf": ("ARF", "ARF"),
    "gaussian_copula": ("Gauss. copula", "GC"),
    "bayesian_network": ("Bayes. net", "BN"),
    "nflow": ("NFlow", "NF"),
    "ctgan": ("CTGAN", "CTGAN"),
}

# The operating points the mode comparison is pinned to, taken from the same config the
# radar figure reads so the two cannot drift apart.
FHE_BITS = int(_radar_cfg(_load_viz_config())["fhe_n_bits"])
SYNTH_SCALE = 100


# ---------------------------------------------------------------- formatting


def _ci(median, low, high, n=3):
    """`0.908 (0.880--0.934)`, the cell format used by tab:s2baseline and tab:s3modes."""
    return f"{median:.{n}f} ({low:.{n}f}--{high:.{n}f})"


def _secs(x):
    """
    CPU seconds for a LaTeX cell.

    Sub-millisecond values go to scientific notation, as the plaintext rows do; above that
    the decimal count follows magnitude, which reproduces main.tex's per-column choice
    (0.0060 for LR, 11.17 for the tree ensembles) with one rule instead of three.
    """
    if x < 1e-3:
        mantissa, exponent = f"{x:.1e}".split("e")
        return rf"${mantissa} \times 10^{{{int(exponent)}}}$"
    return f"{x:.2f}" if x >= 0.1 else f"{x:.4f}"


def _signed(x, n=3):
    """`$+$0.004` / `$-$0.019`: the sign is math so it renders as a proper minus."""
    return f"$-${abs(x):.{n}f}" if x < 0 else f"$+${x:.{n}f}"


def _p(x):
    """p below the bootstrap's 1/(B+1) floor is reported as `$<$0.001`, never as zero."""
    return r"$<$0.001" if x < 0.001 else f"{x:.3f}"


def _mb(x):
    return f"{x:,.0f}"


def _row(*cells):
    return " & ".join(str(c) for c in cells) + r" \\"


def _load():
    df = pd.read_csv(AGGREGATED)
    return df[~df["dataset"].isin(EXCLUDED_DATASETS)]


def _by_dataset(df, mode, model):
    """The six per-dataset rows of one (mode, classifier) cell column."""
    return df[(df["mode"] == mode) & (df["model"] == model)]


# ---------------------------------------------------------------- sections


def fhe(df):
    """tab:fhe: ROC-AUC and encrypted-inference time per precision, plus plaintext."""
    for nb in N_BITS:
        auc, times = [], []
        for model in MODELS:
            sub = _by_dataset(df, f"fhe_{nb}", model)
            auc.append(_ci(sub["roc_auc_median"].median(),
                           sub["roc_auc_ci_low"].median(),
                           sub["roc_auc_ci_high"].median()))
            times.append(_secs(sub["inf_time_per_sample"].median()))
        print(_row(nb, *auc, *times))
    print(r"\midrule")
    plain = [_by_dataset(df, "standard", m) for m in MODELS]
    print(_row("Plaintext",
               *(f"{s['roc_auc_median'].median():.3f}" for s in plain),
               *(_secs(s["inf_time_per_sample"].median()) for s in plain)))


def _one_time(df, mode):
    """
    Everything paid before the first prediction, per (dataset, classifier) cell.

    For the synthetic modes that is the generator fit plus scale-100 sampling plus the
    downstream classifier fit. The first two stages have no metrics file, so they are
    absent from the aggregated CSV and come from the same profile-level helper the radar
    figure uses; the generator is fit once per dataset, so its cost is a dataset-level
    constant added to every classifier cell.
    """
    sub = df[df["mode"] == mode]
    generator = parse_filename_metadata(mode)["mode"]
    offsets = _offsets()
    extra = sub["dataset"].map(lambda d: offsets.get((generator, d), 0.0))
    return sub["train_time"] + extra


def cost(df):
    """tab:cost: system cost by mode at the two pinned operating points."""
    generators = sorted(GENERATORS, key=lambda g: _one_time(df, f"{g}_{SYNTH_SCALE}").median())
    modes = ["standard"] + [f"{g}_{SYNTH_SCALE}" for g in generators] + [f"fhe_{FHE_BITS}"]
    for mode in modes:
        sub = df[df["mode"] == mode]
        print(_row(_mode_label(mode),
                   f"{_one_time(df, mode).median():.2f}",
                   _secs(sub["inf_time_per_sample"].median()),
                   _mb(sub["mem_inf_peak"].median()),
                   f"{sub['model_size_mb'].median():.2f}"))


def s2baseline(df):
    """tab:s2baseline: real-data baselines per dataset and classifier, all five metrics."""
    std = df[df["mode"] == "standard"]
    for i, (dataset, (name, _)) in enumerate(DATASETS.items()):
        if i:
            print(r"\addlinespace")
        for model, label in CLASSIFIERS.items():
            cell = std[(std["dataset"] == dataset) & (std["model"] == model)].iloc[0]
            print(_row(name, f"{label:<7}",
                       *(_ci(cell[f"{m}_median"], cell[f"{m}_ci_low"], cell[f"{m}_ci_high"])
                         for m in TABLE_METRICS)))


def s3modes(df):
    """tab:s3modes: prediction performance by mode, all five metrics, median over cells."""
    modes = ["standard", f"fhe_{FHE_BITS}"] + [f"{g}_{SYNTH_SCALE}" for g in GENERATORS]
    rows = []
    for mode in modes:
        sub = df[df["mode"] == mode]
        cells = [_ci(sub[f"{m}_median"].median(),
                     sub[f"{m}_ci_low"].median(),
                     sub[f"{m}_ci_high"].median()) for m in TABLE_METRICS]
        rows.append((sub["roc_auc_median"].median(), _row(_mode_label(mode), *cells)))
    for _, row in sorted(rows, key=lambda r: r[0], reverse=True):
        print(row)


def s4contrasts(_df):
    """tab:s4contrasts: all 648 prespecified contrasts, in results/stats/ file order."""
    for path in sorted(Path(STATS_DIR).glob("*.csv")):
        family = path.stem.split("_")[0]
        for _, r in pd.read_csv(path).iterrows():
            print(_row(family,
                       DATASETS[r["dataset"]][1],
                       CLASSIFIERS[r["model"]],
                       _contrast(family, r["mode_a"], r["mode_b"]),
                       _signed(r["median_diff"]),
                       f"{_signed(r['ci_low'])}, {_signed(r['ci_high'])}",
                       _p(r["p_value"]),
                       _p(r["p_holm"]),
                       "yes" if r["significant_holm"] else "no"))


# ---------------------------------------------------------------- labels


def _mode_label(key):
    """Row label for a canonical mode key: `Standard`, `ARF`, `FHE (8-bit)`."""
    meta = parse_filename_metadata(key)
    if meta["mode"] == "standard":
        return "Standard"
    if meta["mode"] == "fhe":
        return f"FHE ({meta['n_bits']}-bit)"
    return GENERATOR_LABELS[meta["mode"]][0]


def _contrast_label(key):
    """Contrast-column name for one mode: `Real`, `FHE 8b`, `ARF 100\\%`."""
    meta = parse_filename_metadata(key)
    if meta["mode"] == "standard":
        return "Real"
    if meta["mode"] == "fhe":
        return f"FHE {meta['n_bits']}b"
    return f"{GENERATOR_LABELS[meta['mode']][1]} {meta['synth_scale']}\\%"


def _contrast(family, mode_a, mode_b):
    # A3 contrasts a generator against its own 100% run, so the generator is named once and
    # the right-hand side is bare scale; every other family names both sides in full.
    right = (f"{parse_filename_metadata(mode_b)['synth_scale']}\\%"
             if family == "A3" else _contrast_label(mode_b))
    return rf"{_contrast_label(mode_a)} vs.\ {right}"


SECTIONS = {"fhe": fhe, "cost": cost, "s2baseline": s2baseline,
            "s3modes": s3modes, "s4contrasts": s4contrasts}


def main(argv):
    wanted = argv[1:] or list(SECTIONS)
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"Unknown section(s) {unknown}; choose from {list(SECTIONS)}")

    df = _load()
    for name in wanted:
        print(f"% ---- tab:{name} ----")
        SECTIONS[name](df)
        print()


if __name__ == "__main__":
    main(sys.argv)
