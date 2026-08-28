"""
Self-check for scripts/render_tables.py: python scripts/test_render_tables.py

The point is not coverage, it is that the emitted bodies cannot silently drift from
results/. Each assert recomputes its target independently of the renderer.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_tables import STATS_DIR, SECTIONS, _load


def emit(section):
    buf = io.StringIO()
    with redirect_stdout(buf):
        SECTIONS[section](_load())
    return buf.getvalue().splitlines()


df = _load()

# 1. Every prespecified contrast is present, and its verdict is the CSV's verdict.
rows = emit("s4contrasts")
stats = pd.concat([pd.read_csv(p) for p in sorted(Path(STATS_DIR).glob("*.csv"))])
assert len(rows) == 648, f"expected 648 contrast rows, got {len(rows)}"
assert len(stats) == 648, f"results/stats holds {len(stats)} contrasts, not 648"
for row, sig in zip(rows, stats["significant_holm"]):
    verdict = row.rsplit("&", 1)[1].strip().removesuffix(r"\\").strip()
    assert verdict == ("yes" if sig else "no"), f"significance drifted on: {row}"

# 2. The baseline table is the standard-mode rows of the aggregated CSV, grouped by dataset.
#    Each dataset is a \multirow header line plus three classifier rows, which are the only
#    lines carrying data.
rows = emit("s2baseline")
data = [r for r in rows if r.startswith("    &")]
assert sum(r.startswith(r"\multirow") for r in rows) == 6
assert sum(r.startswith(r"\addlinespace") for r in rows) == 5
assert len(data) == 18, "expected 18 baseline rows, one per (dataset, classifier) cell"
cell = df[(df["mode"] == "standard") & (df["dataset"] == "cardiotocography")
          & (df["model"] == "logistic_regression")].iloc[0]
want = (f"{cell['roc_auc_median']:.3f} "
        f"({cell['roc_auc_ci_low']:.3f}--{cell['roc_auc_ci_high']:.3f})")
assert data[0].rstrip(r"\ ").endswith(want), f"{data[0]!r} does not end in {want!r}"
# One bold per metric column per dataset block. Fails if a data refresh ever produces cells
# that _bold_best cannot separate on the median or either interval bound, which would leave
# a column with two winners.
assert sum(r.count(r"\textbf") for r in data) == 30, "expected 6 datasets x 5 metric columns"

# 3. Mode-level point estimates are the median over the 18 cells, not a pooled median.
rows = emit("s3modes")
assert sum(r.count(r"\textbf") for r in rows) == 5, "expected one bold per metric column"
plain = [r.replace(r"\textbf{", "") for r in rows]  # the bold wraps the cell it marks
emitted = {r.split("&")[0].strip(): r.split("&")[-1].strip().split()[0] for r in plain}
for mode, label in [("standard", "Standard"), ("fhe_8", "FHE (8-bit)"), ("arf_100", "ARF")]:
    want = f"{df[df['mode'] == mode]['roc_auc_median'].median():.3f}"
    assert emitted[label] == want, f"{label}: emitted {emitted[label]}, expected {want}"

print("ok: 648 contrasts verified row for row, 18 baseline rows, mode medians reproduced, "
      "35 bold cells one per metric column per block")
