#!/usr/bin/env python
"""Audit every numeric token in the manuscript against the analysis outputs.

Extracts numbers from tmp/overleaf/main.tex and checks each against the union of
results/metrics_aggregated.csv, results/stats/*.csv, and tmp/manuscript/aggregates.md.
Unmatched tokens are printed with their line number and surrounding context.

The audit produces false positives by design: the text legitimately carries derived
quantities (differences, ratios, fold-changes, percentages, counts). Every flagged
number is expected to land in one of two buckets, traced to a source or justified as
derived from named sources. A number that fits neither is the defect this looks for.

Usage:  python scripts/audit_manuscript_numbers.py [--all]
        --all   also list the tokens that matched a source (verbose)
"""

import argparse
import csv
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "tmp", "overleaf", "main.tex")
METRICS = os.path.join(ROOT, "results", "metrics_aggregated.csv")
STATS_GLOB = os.path.join(ROOT, "results", "stats", "*.csv")
AGGREGATES = os.path.join(ROOT, "tmp", "manuscript", "aggregates.md")

MAX_DECIMALS = 6

# --------------------------------------------------------------------------- #
# Source universe
# --------------------------------------------------------------------------- #

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _floats_from_csv(path):
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            for cell in row:
                cell = cell.strip()
                try:
                    out.append(float(cell))
                except ValueError:
                    continue
    return out


def _floats_from_markdown(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return [float(m.group()) for m in NUM_RE.finditer(text.replace(",", ""))]


def build_source_index():
    """Return {decimals: set of rounded values} plus the raw value list."""
    values = _floats_from_csv(METRICS)
    for path in sorted(glob.glob(STATS_GLOB)):
        values += _floats_from_csv(path)
    values += _floats_from_markdown(AGGREGATES)

    index = {d: set() for d in range(MAX_DECIMALS + 1)}
    for v in values:
        for d in range(MAX_DECIMALS + 1):
            index[d].add(round(abs(v), d))
    return index, values


# --------------------------------------------------------------------------- #
# Structural exclusions
# --------------------------------------------------------------------------- #

# Blanked before tokenizing. Order matters: the scientific-notation rewrite runs
# first so that "$4.0 \times 10^{-6}$" survives as a single comparable number.
SCINOT_RE = re.compile(r"\$?([\d.]+)\s*\\times\s*10\^\{?(-?\d+)\}?\$?")

MASKS = [
    re.compile(r"\\(?:cite|ref|label|eqref|input|bibliography|bibliographystyle)\{[^}]*\}"),
    re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}"),
    re.compile(r"\\(?:setlength|setcounter|renewcommand|captionsetup|usepackage|documentclass|resizebox|arraystretch|hspace|vspace|addtolength)\s*(?:\[[^\]]*\])?(?:\{[^{}]*\})*"),
    re.compile(r"\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?"),
    re.compile(r"\{@\{\}[^}]*\}"),                 # tabular column specs
    re.compile(r"\bp\{0?\.\d+\\textwidth\}"),       # p{0.26\textwidth} column widths
    re.compile(r"[\d.]+\s*\\(?:linewidth|textwidth|columnwidth|baselineskip)"),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:pt|em|ex|mm|cm|in)\b"),
    re.compile(r"\\multicolumn\{\d+\}"),
    re.compile(r"\\arabic\{[^}]*\}"),
    re.compile(r"\b[bB]\d{1,3}\b"),                 # bare citation keys in comments
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),           # ISO dates in comments
    re.compile(r"\b(?:19|20)\d{2}\b"),              # years
    re.compile(r"\bT1\b|\butf8\b"),
    re.compile(r"10\s*\{?,?\}?,?\s*00[01]"),        # bootstrap n = 10,000 and the 1/10,001 floor
    re.compile(r"\bn\s*=\s*10"),
    re.compile(r"\\thetable|\\thefigure"),
]

# Tokens that name a section/table/figure by number rather than reporting a value.
FLOAT_NUM_RE = re.compile(
    r"(?<![\w.])"
    r"(?:\$?[-+]\$?)?"          # $-$ / $+$ LaTeX signs
    r"(\d{1,3}(?:(?:,|\{,\})\d{3})*|\d+)"   # integer part, with , or {,} separators
    r"(\.\d+)?"                 # optional fraction
    r"(?![\w])"
)


def mask_line(line):
    out = SCINOT_RE.sub(lambda m: " " + repr(float(m.group(1)) * 10 ** int(m.group(2))) + " ", line)
    for rx in MASKS:
        out = rx.sub(lambda m: " " * len(m.group()), out)
    return out


def tokens(line):
    """Yield (text, value, decimals, column) for each numeric token in a masked line."""
    for m in FLOAT_NUM_RE.finditer(line):
        raw = m.group()
        cleaned = raw.replace("$", "").replace("{,}", "").replace(",", "").replace("+", "").replace("-", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        frac = m.group(2)
        decimals = len(frac) - 1 if frac else 0
        if decimals > MAX_DECIMALS:
            decimals = MAX_DECIMALS
        yield raw, value, decimals, m.start()


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def context(line, col, width=70):
    start = max(0, col - width)
    end = min(len(line), col + width)
    return ("..." if start else "") + line[start:end].strip() + ("..." if end < len(line) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also list matched tokens")
    args = ap.parse_args()

    index, values = build_source_index()
    print(f"source universe: {len(values)} numeric values "
          f"(metrics_aggregated.csv + results/stats/*.csv + aggregates.md)\n")

    with open(TEX, encoding="utf-8") as fh:
        lines = fh.readlines()

    unmatched, matched = [], []
    for lineno, line in enumerate(lines, 1):
        masked = mask_line(line.rstrip("\n"))
        for raw, value, decimals, col in tokens(masked):
            hit = round(value, decimals) in index[decimals]
            (matched if hit else unmatched).append((lineno, raw, value, decimals, col, line.rstrip("\n")))

    total = len(matched) + len(unmatched)
    print(f"{total} numeric tokens; {len(matched)} matched a source, {len(unmatched)} unmatched\n")

    decimal_hits = [u for u in unmatched if u[3] > 0]
    integer_hits = [u for u in unmatched if u[3] == 0]

    print("=" * 78)
    print(f"UNMATCHED, NON-INTEGER  ({len(decimal_hits)})")
    print("=" * 78)
    for lineno, raw, value, decimals, col, line in decimal_hits:
        print(f"main.tex:{lineno}  {raw}")
        print(f"    {context(line, col)}")

    print()
    print("=" * 78)
    print(f"UNMATCHED, INTEGER  ({len(integer_hits)})  - counts, design constants, derived")
    print("=" * 78)
    seen = {}
    for lineno, raw, value, decimals, col, line in integer_hits:
        seen.setdefault(raw, []).append(lineno)
    for raw in sorted(seen, key=lambda r: float(r.replace(",", "").replace("$", "").replace("{,}", ""))):
        locs = seen[raw]
        shown = ", ".join(str(x) for x in locs[:12]) + (" ..." if len(locs) > 12 else "")
        print(f"  {raw:>10s}  x{len(locs):<4d} lines {shown}")

    if args.all:
        print()
        print("=" * 78)
        print(f"MATCHED  ({len(matched)})")
        print("=" * 78)
        for lineno, raw, value, decimals, col, line in matched:
            print(f"main.tex:{lineno}  {raw}")

    return 1 if decimal_hits else 0


if __name__ == "__main__":
    sys.exit(main())
