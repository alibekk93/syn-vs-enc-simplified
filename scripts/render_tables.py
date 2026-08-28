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
  * nothing is pooled across synthesis scales or across quantization precisions;
  * bold            = the best cell of a metric column within its block, in the two tables
                      that mark one (tab:s2baseline per dataset, tab:s3modes overall);
  * underline       = the runner-up of a metric column, tab:s3modes only. See _mark_best for
                      what "best" and "second" mean when the medians tie.

Usage:
    python scripts/render_tables.py                 # every body (see SECTIONS)
    python scripts/render_tables.py s4contrasts     # one
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_aggregates import EXCLUDED_DATASETS, GENERATORS, MODELS, N_BITS
from src.utils import parse_filename_metadata
from src.visualization import _load_viz_config, _radar_cfg

AGGREGATED = "results/metrics_aggregated.csv"
STATS_DIR = "results/stats"

# The tables order the metric columns Accuracy, Precision, Recall, F1, ROC-AUC, which is not
# the storage order in the CSV. tab:fhe and tab:cost are in main.tex; tab:s2baseline and
# tab:s3modes are in supplementary.tex.
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


def _cell(median, low, high, n=3):
    """
    One cell of tab:s2baseline or tab:s3modes: its text and its ranking key.

    Both come out of the same three numbers rounded the same way, so the precision a reader
    sees and the precision _bold_best compares can never drift apart.
    """
    return _ci(median, low, high, n), (round(median, n), round(low, n), round(high, n))


def _mark_best(block, underline_second=False):
    """
    Mark the best cell of every metric column of one block, given row-major _cell pairs.

    Bold is the best; `underline_second` also underlines the runner-up, which tab:s3modes
    wants and tab:s2baseline does not, three classifiers being too few for a second place to
    mean much. Returns the text alone, marked cells wrapped whole, interval included, since
    that is the unit the column is compared on. \\underline is LaTeX kernel, so neither table
    needs a package the preamble does not already load.

    The key carries the interval bounds after the median because the median alone does not
    decide three columns of tab:s2baseline, where cells are equal at full float precision
    rather than rounded into equality: accuracy on mammographic mass (RF and XGBoost, both
    0.8238), F1 on diabetes (LR and RF, both 0.5591), and accuracy on heart disease (all
    three classifiers, 0.8689). Among equal medians the higher lower bound wins, then the
    higher upper bound; all three resolve on the lower bound alone.

    Ranks are shared, so cells identical in all three numbers are marked alike rather than
    separated arbitrarily. Second place is the second distinct key, not the second row: where
    two cells tie for best, the next one down is still the runner-up.
    """
    out = [[text for text, _ in row] for row in block]
    for col in range(len(out[0])):
        ranked = sorted({row[col][1] for row in block}, reverse=True)
        marks = dict(zip(ranked, [r"\textbf"] + ([r"\underline"] if underline_second else [])))
        for i, row in enumerate(block):
            command = marks.get(row[col][1])
            if command:
                out[i][col] = rf"{command}{{{out[i][col]}}}"
    return out


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


def _cost_cells(sub):
    """
    The four cost columns of tab:cost for one selection of rows.

    one_time_cost is everything paid before the first prediction, derived per row by
    src/utils.join_stage_columns; before that existed this had to re-glob the profile
    JSONs to recover the generator fit and the synthetic draw.
    """
    return (f"{sub['one_time_cost'].median():.2f}",
            _secs(sub["inf_time_per_sample"].median()),
            _mb(sub["mem_inf_peak"].median()),
            f"{sub['model_size_mb'].median():.2f}")


def cost(df):
    """
    tab:cost: system cost by mode and classifier at the two pinned operating points.

    Each mode gets a median row over its 18 cells followed by one row per classifier.
    The per-classifier rows exist because for FHE the median is not a summary: encrypted
    inference spans 0.0060 s to 11.17 s across the three classifiers, so the mode-level
    3.00 s is simply whichever classifier sorts to the middle. The plaintext modes vary
    little by classifier, and showing that is itself the point of contrast.
    """
    generators = sorted(GENERATORS,
                        key=lambda g: df[df["mode"] == f"{g}_{SYNTH_SCALE}"]["one_time_cost"].median())
    modes = ["standard"] + [f"{g}_{SYNTH_SCALE}" for g in generators] + [f"fhe_{FHE_BITS}"]
    for i, mode in enumerate(modes):
        if i:
            print(r"\addlinespace")
        sub = df[df["mode"] == mode]
        # `\\*` forbids a page break after the row, so the longtable can only split
        # between modes and never leaves a mode's classifiers stranded on two pages.
        print(_row(_mode_label(mode), "Median", *_cost_cells(sub)) + "*")
        rows = [(label, sub[sub["model"] == model]) for model, label in CLASSIFIERS.items()]
        for j, (label, cell) in enumerate(rows):
            last = j == len(rows) - 1
            print(_row("", label, *_cost_cells(cell)) + ("" if last else "*"))


def s2baseline(df):
    """
    tab:s2baseline: real-data baselines per dataset and classifier, all five metrics.

    The dataset column is a \\multirow spanning the block's three classifier rows, so the
    name is set once; the \\multirow and the first classifier row are one LaTeX row split
    over two source lines. Each block is bolded on its own, the comparison being between
    the three classifiers on one dataset.
    """
    std = df[df["mode"] == "standard"]
    for i, (dataset, (name, _)) in enumerate(DATASETS.items()):
        if i:
            print(r"\addlinespace")
        cells = [std[(std["dataset"] == dataset) & (std["model"] == model)].iloc[0]
                 for model in CLASSIFIERS]
        block = _mark_best([[_cell(c[f"{m}_median"], c[f"{m}_ci_low"], c[f"{m}_ci_high"])
                             for m in TABLE_METRICS] for c in cells])
        print(rf"\multirow{{{len(block)}}}{{*}}{{{name}}}")
        for label, row in zip(CLASSIFIERS.values(), block):
            print(f"    & {label:<7} & " + _row(*row))


def s3modes(df):
    """
    tab:s3modes: prediction performance by mode, all five metrics, median over cells.

    The whole table is one block: bold marks the best mode per metric and underline the
    runner-up, over all seven rows. Marking runs after the sort because it has to mark rows
    in the order they are emitted, though which row wins does not depend on that order.

    The row order is the raw ROC-AUC median while the marks rank the printed one, so the two
    can disagree where a pair prints alike: ARF leads FHE by 0.0005 (0.8641 against 0.8637)
    and sorts above it, but both print 0.864 and the interval breaks the tie the other way,
    putting the ROC-AUC underline on FHE. The caption already tells the reader the row order
    asserts no ordering between those two.
    """
    modes = ["standard", f"fhe_{FHE_BITS}"] + [f"{g}_{SYNTH_SCALE}" for g in GENERATORS]
    rows = []
    for mode in modes:
        sub = df[df["mode"] == mode]
        cells = [_cell(sub[f"{m}_median"].median(),
                       sub[f"{m}_ci_low"].median(),
                       sub[f"{m}_ci_high"].median()) for m in TABLE_METRICS]
        rows.append((sub["roc_auc_median"].median(), _mode_label(mode), cells))
    rows.sort(key=lambda r: r[0], reverse=True)
    block = _mark_best([cells for _, _, cells in rows], underline_second=True)
    for (_, label, _), row in zip(rows, block):
        print(_row(label, *row))


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
