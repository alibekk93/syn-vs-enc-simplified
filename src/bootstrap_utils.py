"""Shared utilities for the simple bootstrap evaluation."""

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.models import SUPPORTED_METRICS

logger = logging.getLogger(__name__)


def run_bootstrap(
    predict_fn: Callable,
    predict_proba_fn: Callable,
    X_test,
    y_test,
    n: int,
    seed: int,
    metric_names: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    Draw N bootstrap samples from the test set and evaluate each one.

    Accepts both pandas DataFrame/Series (Model) and numpy arrays (FHEModel)
    for X_test/y_test — the original type is preserved when passed to
    predict_fn so that model-internal type handling (e.g. GPU dispatch) works.

    Args:
        predict_fn:       Callable that accepts X and returns y_pred array.
        predict_proba_fn: Callable that accepts X and returns y_proba or None.
        X_test:           Test features (DataFrame or ndarray).
        y_test:           Test labels (Series or ndarray).
        n:                Number of bootstrap iterations.
        seed:             Random seed for reproducibility.
        metric_names:     Ordered list of metric names to compute.

    Returns:
        (iter_metrics, iter_times) — two lists of length N.
        iter_metrics[i]: {metric_name: float} for iteration i.
        iter_times[i]:   {"total": float, "per_sample": float} for iteration i.
    """
    is_pandas_X = isinstance(X_test, pd.DataFrame)
    is_pandas_y = isinstance(y_test, pd.Series)

    if is_pandas_X:
        X_base = X_test.reset_index(drop=True)
    else:
        X_base = np.asarray(X_test)

    if is_pandas_y:
        y_base = y_test.reset_index(drop=True)
    else:
        y_base = np.asarray(y_test)

    n_samples = len(X_base)
    rng = np.random.default_rng(seed)

    iter_metrics: list[dict] = []
    iter_times: list[dict]   = []

    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)

        X_boot = X_base.iloc[idx] if is_pandas_X else X_base[idx]
        y_boot = y_base.iloc[idx].values if is_pandas_y else y_base[idx]

        t0      = time.perf_counter()
        y_pred  = predict_fn(X_boot)
        y_proba = predict_proba_fn(X_boot)
        elapsed = time.perf_counter() - t0

        metrics: dict[str, float] = {}
        for metric in metric_names:
            if metric not in SUPPORTED_METRICS:
                continue
            try:
                metrics[metric] = round(SUPPORTED_METRICS[metric](y_boot, y_pred, y_proba), 4)
            except Exception:
                metrics[metric] = float("nan")

        iter_metrics.append(metrics)
        iter_times.append({
            "total":      round(elapsed, 6),
            "per_sample": round(elapsed / n_samples, 9),
        })

    return iter_metrics, iter_times


def to_metric_lists(iter_metrics: list[dict]) -> dict[str, list[float]]:
    """Transpose list-of-dicts to dict-of-lists."""
    if not iter_metrics:
        return {}
    return {
        metric: [m[metric] for m in iter_metrics]
        for metric in iter_metrics[0]
    }


def overwrite_metrics_with_bootstrap(
    path: Path,
    metric_lists: dict[str, list[float]],
    n_bootstrap: int,
) -> None:
    """
    Read the metrics JSON that model.evaluate() just wrote and replace each
    scalar metric value with the corresponding bootstrap list. Adds n_bootstrap
    field. Overwrites the file in place.
    """
    with open(path) as f:
        data = json.load(f)

    data["n_bootstrap"] = n_bootstrap
    for metric, values in metric_lists.items():
        data["metrics"][metric] = values

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.debug(f"Bootstrap metrics written → {path}")
