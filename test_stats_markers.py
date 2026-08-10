"""Check that violin significance markers select the right modes and tiers.

Two things fail silently here: a wrong mode_b -> mode_key translation marks the
wrong violin, and a wrong tier boundary overstates a result in print.

Run: python test_stats_markers.py
"""

import csv
import tempfile
from pathlib import Path

from src.visualization import (
    _DEFAULT_SIG_LEVELS,
    _significance_marker,
    _significant_vs_real,
)

COLUMNS = ["dataset", "model", "metric", "comparison", "mode_a", "mode_b",
           "significant_holm", "p_holm"]

ROWS = [
    # A1: generator vs real, significant -> marks the "arf" violin (scale-100 view)
    ("diabetes", "xgboost", "roc_auc", "reference", "standard", "arf_100", True, 0.0005),
    # A1: not significant -> no marker regardless of p
    ("diabetes", "xgboost", "roc_auc", "reference", "standard", "ctgan_100", False, 0.9),
    # A2: FHE vs real, significant -> marks "fhe_2", keeping the bit width
    ("diabetes", "xgboost", "roc_auc", "reference", "standard", "fhe_2", True, 0.03),
    # A3: reference is a generator, not Real -> excluded by the mode_a filter
    ("diabetes", "xgboost", "roc_auc", "reference", "arf_100", "arf_300", True, 0.001),
    # A4: adjacent design -> excluded by the comparison filter
    ("diabetes", "xgboost", "roc_auc", "adjacent", "fhe_2", "fhe_4", True, 0.001),
    # Another metric -> excluded
    ("diabetes", "xgboost", "accuracy", "reference", "standard", "nflow_100", True, 0.001),
    # A non-100 scale vs Real must not mark the scale-100 violin
    ("diabetes", "xgboost", "roc_auc", "reference", "standard", "nflow_300", True, 0.001),
    # A different cell keeps its own markers
    ("heart_disease", "logistic_regression", "roc_auc", "reference", "standard",
     "fhe_12", True, 0.005),
]


def test_selection():
    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp) / "contrasts.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(ROWS)

        got = _significant_vs_real("roc_auc", str(tmp))

    assert got == {
        ("diabetes", "xgboost"): {"arf": 0.0005, "fhe_2": 0.03},
        ("heart_disease", "logistic_regression"): {"fhe_12": 0.005},
    }, got

    # No stats run yet must degrade to unannotated panels, not an exception.
    with tempfile.TemporaryDirectory() as empty:
        assert _significant_vs_real("roc_auc", empty) == {}


def test_tiers():
    m = lambda p: _significance_marker(p, _DEFAULT_SIG_LEVELS)

    assert m(0.0005) == "***"      # reachable only once B is large enough
    assert m(0.005) == "**"        # the B=1000 floor landed here
    assert m(0.03) == "*"
    assert m(0.2) is None

    # Boundaries are exclusive: p exactly at a bound falls to the weaker tier.
    assert m(0.001) == "**"
    assert m(0.01) == "*"
    assert m(0.05) is None


if __name__ == "__main__":
    test_selection()
    test_tiers()
    print("ok")
