# src/stats_tests.py
"""
Paired-bootstrap significance testing across evaluation modes.

WHY A PAIRED TEST IS COMPUTABLE FROM THE STORED JSONs
-----------------------------------------------------
results/metrics/{mode}__{model}__{dataset}__test__metrics.json stores, per
metric, 1000 BOOTSTRAP REPLICATE VALUES (not per-sample predictions). Those
replicates are already index-aligned ACROSS MODES within a dataset:

  * Every mode (standard / synthetic / FHE) is evaluated on the SAME real test
    set for a given dataset — model.X_test / model.y_test, produced by a
    stratified split with random_state=42. Synthetic data is only ever used for
    TRAINING; FHE only changes the inference path. The evaluation set is
    identical.
  * Every mode calls src.bootstrap_utils.run_bootstrap(..., seed=42), which does
    rng = np.random.default_rng(42) once and then draws
    idx = rng.integers(0, n_samples, size=n_samples) per replicate.
  * n_samples is a property of the dataset, not the mode. Same seed + same
    generator + same draw sizes => the same stream of index vectors.

Therefore replicate i of mode A and replicate i of mode B were computed on the
*same* resampled test rows, and d_i = A_i - B_i is a legitimate paired
difference. No re-inference is required.

This assumption is CHECKED AT RUNTIME (see _load_replicates): a file's metric
array must have length equal to its declared n_bootstrap, and the two arrays in
a pair must have equal length. A pair failing the check is skipped with
logger.error rather than silently producing a meaningless unpaired comparison.

NOTE ON QUANTIZATION: src/bootstrap_utils.py rounds every metric to 4 decimals
before storage. Differences smaller than 1e-4 are therefore recorded as exact
ties (d_i == 0). Combined with the natural discreteness of ROC-AUC on small
test sets (n ~ 100-200), ties are common and are sometimes total. Tie handling
is deliberately conservative — see paired_bootstrap_test().
"""

import csv
import fnmatch
import itertools
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils import parse_filename_metadata

logger = logging.getLogger(__name__)

DEFAULT_METRICS_DIR = "results/metrics"
DEFAULT_OUTPUT_DIR = "results/stats"

# Below this many usable (non-NaN on both sides) replicates the p-value is not
# reported at all — a bootstrap ASL estimated from a handful of replicates is
# noise, and reporting it would invite over-interpretation.
MIN_VALID_REPLICATES = 100

SUPPORTED_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Canonical mode ordering: the real-data reference first, then synthetic
# families alphabetically, then FHE. Within a family, ascending synth_scale /
# n_bits. Makes `mode_a` stable down the table and pair generation deterministic.
_MODE_FAMILY_ORDER = {
    "standard": 0,
    "arf": 1,
    "bayesian_network": 2,
    "ctgan": 3,
    "gaussian_copula": 4,
    "nflow": 5,
    "fhe": 6,
}

_OUTPUT_FIELDS = [
    "dataset", "model", "metric",
    "mode_a", "mode_b",
    "n_valid", "n_ties",
    "mean_a", "mean_b", "mean_diff",
    "ci_low", "ci_high",
    "p_value", "p_holm", "family_size", "significant_holm",
    "better",
]


# ===========================================================
# LOADING
# ===========================================================

def _mode_sort_key(mode: str) -> tuple:
    """Sort key implementing the canonical mode ordering described above."""
    meta = parse_filename_metadata(mode)
    family = meta["mode"]
    # n_bits and synth_scale are mutually exclusive by construction; whichever
    # is set is the family's magnitude axis.
    magnitude = meta["n_bits"] or meta["synth_scale"] or 0
    return (_MODE_FAMILY_ORDER.get(family, 99), family, magnitude)


def _load_replicates(metrics_dir: str, metric: str) -> dict[tuple[str, str, str], np.ndarray]:
    """
    Load the bootstrap replicate array for `metric` from every metrics JSON.

    Returns a dict keyed by (dataset, model, mode). Files that do not match the
    `{mode}__{model}__{dataset}__test__metrics.json` shape, that lack the
    requested metric, or whose array length disagrees with the declared
    n_bootstrap are skipped with a log message.
    """
    directory = Path(metrics_dir)
    if not directory.is_dir():
        raise SystemExit(f"Metrics directory not found: {metrics_dir}")

    replicates: dict[tuple[str, str, str], np.ndarray] = {}
    for path in sorted(directory.glob("*.json")):
        parts = path.stem.split("__")
        if len(parts) != 5 or parts[3] != "test" or parts[4] != "metrics":
            logger.warning(f"Unrecognized metrics filename: {path.name}")
            continue
        mode, model, dataset = parts[0], parts[1], parts[2]

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"Could not read {path.name}: {exc}")
            continue

        values = data.get("metrics", {}).get(metric)
        if not values:
            logger.warning(f"{path.name}: no '{metric}' values — skipping")
            continue

        # Pairing check: the stored array must be the full replicate series.
        n_bootstrap = data.get("n_bootstrap")
        if n_bootstrap and len(values) != n_bootstrap:
            logger.error(
                f"{path.name}: {metric} has {len(values)} values but n_bootstrap="
                f"{n_bootstrap} — replicate alignment cannot be trusted, dropping"
            )
            continue

        replicates[(dataset, model, mode)] = np.asarray(values, dtype=float)

    logger.info(f"Loaded {len(replicates)} replicate series for metric '{metric}'")
    return replicates


def select_names(available: set[str], patterns: Optional[list[str]], label: str,
                 sort_key=None) -> list[str]:
    """
    Resolve explicit names and/or fnmatch glob patterns against what is on disk.

    `--modes standard 'fhe_*' arf_300` — exact names are just globs without
    metacharacters, so one code path handles both. Matching is against the names
    actually discovered in the metrics directory, so a typo produces a loud
    warning rather than a silently-empty comparison set.

    sort_key defaults to plain alphabetical; callers selecting modes pass
    _mode_sort_key to get the canonical family ordering instead.
    """
    if not patterns:
        return sorted(available, key=sort_key)

    selected: set[str] = set()
    for pattern in patterns:
        matched = {name for name in available if fnmatch.fnmatchcase(name, pattern)}
        if not matched:
            logger.warning(f"--{label} pattern {pattern!r} matched nothing on disk")
        selected |= matched

    return sorted(selected, key=sort_key)


# ===========================================================
# THE TEST
# ===========================================================

def paired_bootstrap_test(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Two-sided paired bootstrap test on index-aligned replicate arrays.

    Given d_i = a_i - b_i over i = 1..B, the p-value is the achieved
    significance level (ASL) read off the difference distribution:

        r = min(#{d>0}, #{d<0}) + #{d==0}
        p = min(1, 2 * (r + 1) / (B + 1))

    Why the proportion-on-the-wrong-side form rather than a shifted null: the
    bootstrap distribution of d estimates the sampling distribution of the true
    difference, and asking how much of it lies past 0 is the exact dual of the
    percentile confidence interval. Since this project already reports 95% CIs
    as np.percentile(arr, [2.5, 97.5]) (see aggregate_metrics_csv in utils.py),
    this choice guarantees p < 0.05 <=> the reported CI of the difference
    excludes 0. A shifted-null formulation would let a row show a CI excluding
    zero next to p > 0.05, which is indefensible in a paper table.

    Why (r+1)/(B+1) rather than r/B: with B=1000, r=0 would give p=0, which is
    false — it only means the test's resolution was exhausted. Adding the
    observed statistic to the reference set (Davison & Hinkley 1997; North,
    Curtis & Sham 2002) floors p at 2/1001 ~ 0.002.

    Why ties are charged to the smaller side: a tie is a replicate on which the
    two modes were indistinguishable at the stored precision. Assigning it to
    either directional count would manufacture evidence the data does not
    contain. Adding ties to the minimum can only increase r, hence only increase
    p, so quantization artifacts can never create significance. The fully
    degenerate case falls out for free — if every replicate ties then r = B and
    p = 2*(B+1)/(B+1) = 2, clipped to 1.0, which is exactly right for two modes
    that produced identical predictions.
    """
    out = {
        "n_valid": 0, "n_ties": None,
        "mean_a": None, "mean_b": None, "mean_diff": None,
        "ci_low": None, "ci_high": None, "p_value": None,
    }

    # Joint mask. Dropping NaNs independently per array would shift the two
    # series relative to each other and destroy the index alignment the whole
    # test rests on. roc_auc is NaN whenever a resample drew a single class.
    mask = np.isfinite(a) & np.isfinite(b)
    n_valid = int(mask.sum())
    out["n_valid"] = n_valid
    if n_valid < MIN_VALID_REPLICATES:
        return out

    av, bv = a[mask], b[mask]
    d = av - bv

    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    n_tie = int((d == 0).sum())

    r = min(n_pos, n_neg) + n_tie
    p = min(1.0, 2.0 * (r + 1) / (n_valid + 1))

    ci_low, ci_high = np.percentile(d, [2.5, 97.5])
    out.update(
        n_ties=n_tie,
        mean_a=round(float(av.mean()), 4),
        mean_b=round(float(bv.mean()), 4),
        mean_diff=round(float(d.mean()), 4),
        ci_low=round(float(ci_low), 4),
        ci_high=round(float(ci_high), 4),
        # 6dp: the 2/1001 floor needs the resolution to stay distinguishable.
        p_value=round(float(p), 6),
    )
    return out


def holm_bonferroni(pvals: list[Optional[float]]) -> list[Optional[float]]:
    """
    Holm-Bonferroni step-down adjusted p-values, positionally aligned to input.

    None entries are NOT members of the family: a test that could not be run
    (too few valid replicates) is neither counted in m nor adjusted. Including
    them would inflate m and unfairly penalise the tests that did run.
    """
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    adjusted: list[Optional[float]] = [None] * len(pvals)
    if m == 0:
        return adjusted

    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        # Step-down weights m, m-1, ..., 1. The running max enforces
        # monotonicity so a larger raw p can never receive a smaller adjusted p.
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# ===========================================================
# DRIVER
# ===========================================================

def run_pairwise_tests(
    metrics_dir: str = DEFAULT_METRICS_DIR,
    metric: str = "roc_auc",
    modes: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    alpha: float = 0.05,
) -> list[dict]:
    """
    All pairwise mode comparisons within each (dataset, classifier) cell.

    Holm correction is applied per cell: each cell is an independent
    sub-analysis, so its own set of pairwise tests is the natural family.
    """
    replicates = _load_replicates(metrics_dir, metric)
    if not replicates:
        logger.warning(f"No usable replicate series found in {metrics_dir}")
        return []

    all_datasets = {key[0] for key in replicates}
    all_models = {key[1] for key in replicates}
    all_modes = {key[2] for key in replicates}

    sel_datasets = select_names(all_datasets, datasets, "datasets")
    sel_models = select_names(all_models, models, "models")
    sel_modes = select_names(all_modes, modes, "modes", sort_key=_mode_sort_key)

    if len(sel_modes) < 2:
        raise SystemExit(
            f"Need at least 2 modes to compare, got {len(sel_modes)}. "
            f"Available modes: {', '.join(sorted(all_modes, key=_mode_sort_key))}"
        )
    if not sel_datasets or not sel_models:
        raise SystemExit("Dataset or model selection is empty — nothing to compare.")

    logger.info(
        f"Comparing {len(sel_modes)} modes across {len(sel_datasets)} dataset(s) "
        f"x {len(sel_models)} classifier(s): {', '.join(sel_modes)}"
    )

    rows: list[dict] = []
    for dataset in sel_datasets:
        for model in sel_models:
            # Modes present for THIS cell — a mode may be missing for one
            # classifier but not another (e.g. an FHE sweep that only ran for
            # xgboost), so pairs are built per cell rather than globally.
            present = [m for m in sel_modes if (dataset, model, m) in replicates]
            if len(present) < 2:
                logger.info(
                    f"{dataset}/{model}: only {len(present)} selected mode(s) present — skipping"
                )
                continue

            cell_rows: list[dict] = []
            for mode_a, mode_b in itertools.combinations(present, 2):
                try:
                    a = replicates[(dataset, model, mode_a)]
                    b = replicates[(dataset, model, mode_b)]
                    if len(a) != len(b):
                        logger.error(
                            f"{dataset}/{model}: {mode_a} has {len(a)} replicates but "
                            f"{mode_b} has {len(b)} — cannot pair, skipping"
                        )
                        continue

                    result = paired_bootstrap_test(a, b)
                    diff = result["mean_diff"]
                    row = {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "mode_a": mode_a,
                        "mode_b": mode_b,
                        **result,
                        # All supported metrics are higher-is-better, so the sign
                        # of the difference names the winner directly. Descriptive
                        # only — read it together with significant_holm.
                        "better": None if diff is None else (
                            "tie" if diff == 0 else (mode_a if diff > 0 else mode_b)
                        ),
                    }
                    cell_rows.append(row)
                except Exception as exc:
                    # One bad combination must not abort the sweep.
                    logger.error(f"{dataset}/{model}: {mode_a} vs {mode_b} failed: {exc}")

            if not cell_rows:
                continue

            adjusted = holm_bonferroni([r["p_value"] for r in cell_rows])
            family_size = sum(1 for r in cell_rows if r["p_value"] is not None)
            logger.info(f"{dataset}/{model}: {family_size} tests in Holm family")
            for row, p_holm in zip(cell_rows, adjusted):
                row["p_holm"] = None if p_holm is None else round(float(p_holm), 6)
                row["family_size"] = family_size
                row["significant_holm"] = p_holm is not None and p_holm < alpha

            rows.extend(cell_rows)

    rows.sort(key=lambda r: (
        r["dataset"], r["model"], _mode_sort_key(r["mode_a"]), _mode_sort_key(r["mode_b"])
    ))
    return rows


# ===========================================================
# OUTPUT
# ===========================================================

def write_stats_csv(rows: list[dict], output_path: str) -> None:
    """Write the results table as CSV (csv.DictWriter, matching aggregate_metrics_csv)."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} comparison rows to {output_path}")


def write_stats_markdown(rows: list[dict], output_path: str) -> None:
    """Write the same rows as GitHub pipe tables, one section per (dataset, classifier)."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value):
        """Render a missing statistic as an em-dash rather than the string 'None'."""
        return "—" if value is None else value

    lines: list[str] = ["# Paired bootstrap comparisons", ""]
    current_cell = None
    for row in rows:
        cell = (row["dataset"], row["model"])
        if cell != current_cell:
            current_cell = cell
            lines += [
                "",
                f"## {row['dataset']} / {row['model']}  ({row['metric']}, "
                f"Holm family n={row['family_size']})",
                "",
                "| mode_a | mode_b | mean_a | mean_b | diff | 95% CI | ties | p | p_holm | sig |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
        ci = "—" if row["ci_low"] is None else f"[{row['ci_low']}, {row['ci_high']}]"
        lines.append(
            f"| {row['mode_a']} | {row['mode_b']} | {fmt(row['mean_a'])} | "
            f"{fmt(row['mean_b'])} | {fmt(row['mean_diff'])} | {ci} | "
            f"{fmt(row['n_ties'])} | {fmt(row['p_value'])} | {fmt(row['p_holm'])} | "
            f"{'**yes**' if row['significant_holm'] else 'no'} |"
        )

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote markdown table to {output_path}")


def run(
    metrics_dir: str = DEFAULT_METRICS_DIR,
    output: Optional[str] = None,
    metric: str = "roc_auc",
    modes: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    alpha: float = 0.05,
    fmt: str = "csv",
) -> list[dict]:
    """Entry point used by `main.py paired-bootstrap-tests`."""
    rows = run_pairwise_tests(
        metrics_dir=metrics_dir, metric=metric,
        modes=modes, datasets=datasets, models=models, alpha=alpha,
    )
    if not rows:
        # Still write the header so downstream steps don't crash on a missing file.
        logger.warning("No comparisons produced — writing an empty table")

    base = Path(output) if output else Path(DEFAULT_OUTPUT_DIR) / f"paired_bootstrap__{metric}.csv"
    if fmt in ("csv", "both"):
        write_stats_csv(rows, str(base))
    if fmt in ("markdown", "both"):
        write_stats_markdown(rows, str(base.with_suffix(".md")))

    n_sig = sum(1 for r in rows if r["significant_holm"])
    logger.info(f"{n_sig}/{len(rows)} comparisons significant after Holm (alpha={alpha})")
    return rows
