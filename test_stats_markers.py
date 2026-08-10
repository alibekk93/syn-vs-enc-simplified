"""Check that violin significance markers select the right modes.

The mapping from a stats row to a violin is the part that fails silently: a
wrong mode_b -> mode_key translation marks the wrong violin, and a missing
filter marks a violin from an analysis that was never about Real.

Run: python test_stats_markers.py
"""

import csv
import tempfile
from pathlib import Path

from src.visualization import _significant_vs_real

COLUMNS = ["dataset", "model", "metric", "comparison", "mode_a", "mode_b",
           "significant_holm"]

ROWS = [
    # A1: generator vs real, significant -> marks the "arf" violin (scale-100 view)
    ("diabetes", "xgboost", "roc_auc", "reference", "standard", "arf_100", True),
    # A1: not significant -> no marker
    ("diabetes", "xgboost", "roc_auc", "reference", "standard", "ctgan_100", False),
    # A2: FHE vs real, significant -> marks "fhe_2", keeping the bit width
    ("diabetes", "xgboost", "roc_auc", "reference", "standard", "fhe_2", True),
    # A3: reference is a generator, not Real -> excluded by the mode_a filter
    ("diabetes", "xgboost", "roc_auc", "reference", "arf_100", "arf_300", True),
    # A4: adjacent design -> excluded by the comparison filter
    ("diabetes", "xgboost", "roc_auc", "adjacent", "fhe_2", "fhe_4", True),
    # Another metric -> excluded
    ("diabetes", "xgboost", "accuracy", "reference", "standard", "nflow_100", True),
    # A non-100 scale vs Real must not mark the scale-100 violin
    ("diabetes", "xgboost", "roc_auc", "reference", "standard", "nflow_300", True),
    # A different cell keeps its own markers
    ("heart_disease", "logistic_regression", "roc_auc", "reference", "standard",
     "fhe_12", True),
]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "contrasts.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(ROWS)

        got = _significant_vs_real("roc_auc", str(tmp))

    assert got == {
        ("diabetes", "xgboost"): frozenset({"arf", "fhe_2"}),
        ("heart_disease", "logistic_regression"): frozenset({"fhe_12"}),
    }, got

    # No stats run yet must degrade to unannotated panels, not an exception.
    with tempfile.TemporaryDirectory() as empty:
        assert _significant_vs_real("roc_auc", empty) == {}

    print("ok")


if __name__ == "__main__":
    main()
