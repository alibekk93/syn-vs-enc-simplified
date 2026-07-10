"""Shared utilities for the simple bootstrap evaluation."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.models import SUPPORTED_METRICS

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    metric_names: list[str],
) -> dict[str, float]:
    """Compute point metrics from prediction arrays."""
    results = {}
    for metric in metric_names:
        if metric not in SUPPORTED_METRICS:
            continue
        try:
            results[metric] = round(SUPPORTED_METRICS[metric](y_true, y_pred, y_proba), 4)
        except Exception:
            results[metric] = float("nan")
    return results


def run_bootstrap(
    y_true,
    y_pred,
    y_proba,
    n: int,
    seed: int,
    metric_names: list[str],
) -> list[dict]:
    """
    Draw N bootstrap samples from already-computed (y_true, y_pred, y_proba)
    and compute metrics on each sample. No model inference — resamples arrays.

    Args:
        y_true:       True labels (array-like).
        y_pred:       Predicted labels from a single inference pass (array-like).
        y_proba:      Predicted probabilities from a single inference pass, or None.
        n:            Number of bootstrap iterations.
        seed:         Random seed for reproducibility.
        metric_names: Ordered list of metric names to compute.

    Returns:
        List of N metric dicts, one per bootstrap iteration.
    """
    y_true_arr  = np.asarray(y_true)
    y_pred_arr  = np.asarray(y_pred)
    y_proba_arr = np.asarray(y_proba) if y_proba is not None else None

    n_samples = len(y_true_arr)
    rng = np.random.default_rng(seed)
    iter_metrics: list[dict] = []

    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt  = y_true_arr[idx]
        yp  = y_pred_arr[idx]
        ypr = y_proba_arr[idx] if y_proba_arr is not None else None
        iter_metrics.append(compute_metrics(yt, yp, ypr, metric_names))

    return iter_metrics


def to_metric_lists(iter_metrics: list[dict]) -> dict[str, list[float]]:
    """Transpose list-of-dicts to dict-of-lists."""
    if not iter_metrics:
        return {}
    return {
        metric: [m[metric] for m in iter_metrics]
        for metric in iter_metrics[0]
    }


def save_metrics_json(
    path: Path,
    mode: str,
    model_name: str,
    dataset_name: str,
    split: str,
    metrics,
    n_bootstrap: int = 0,
    extra_fields: Optional[dict] = None,
) -> None:
    """
    Write a metrics JSON file matching the format of model._save_results().

    extra_fields are inserted after split (used by FHE for fhe and n_bits).
    n_bootstrap is included when > 0.
    metrics is either a dict of scalars (no bootstrap) or dict of lists (bootstrap).
    """
    data: dict = {"mode": mode, "model": model_name, "dataset": dataset_name, "split": split}
    if extra_fields:
        data.update(extra_fields)
    if n_bootstrap > 0:
        data["n_bootstrap"] = n_bootstrap
    data["metrics"] = metrics

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Metrics saved → {path}")
