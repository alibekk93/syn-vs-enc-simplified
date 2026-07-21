"""Shared utilities for the simple bootstrap evaluation."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.models import SUPPORTED_METRICS

logger = logging.getLogger(__name__)


# Probability above which the positive class is predicted. Strict `>` mirrors
# sklearn's own tie-breaking: predict() is classes_[argmax(predict_proba)], and
# np.argmax returns the first maximum, so an exact 0.5/0.5 split resolves to the
# negative class. Using `>=` here would silently disagree with predict() on that
# boundary.
POSITIVE_THRESHOLD = 0.5


def labels_from_proba(
    y_proba: np.ndarray,
    classes: Optional[np.ndarray] = None,
    threshold: float = POSITIVE_THRESHOLD,
) -> np.ndarray:
    """Derive binary class labels from positive-class probabilities."""
    idx = (np.asarray(y_proba) > threshold).astype(int)
    if classes is None:
        return idx
    return np.asarray(classes)[idx]


def predict_once(model, X, **predict_kwargs) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Produce (y_pred, y_proba) from a single inference pass.

    Calling predict() and predict_proba() separately runs inference over X
    twice. Under FHE that means evaluating the encrypted circuit twice for no
    added information — predict() is derived from predict_proba() anyway — which
    both doubles runtime and, because the profiler divides elapsed time by one
    test set's worth of samples, doubles the reported per-sample cost.

    Deriving the labels from the probabilities instead keeps a single pass.
    Models without predict_proba fall back to predict() and return None for the
    probabilities, matching the previous behaviour.

    Extra kwargs are forwarded to the model (the FHE wrapper takes `fhe=...`).
    """
    y_proba = model.predict_proba(X, **predict_kwargs)
    if y_proba is None:
        return model.predict(X, **predict_kwargs), None

    # Both Model and FHEModel hold the fitted estimator as `.model`.
    classes = getattr(getattr(model, "model", None), "classes_", None)
    return labels_from_proba(y_proba, classes=classes), y_proba


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


def save_predictions_json(
    path: Path,
    mode: str,
    model_name: str,
    dataset_name: str,
    split: str,
    y_true,
    y_proba,
    y_pred,
    threshold: float = POSITIVE_THRESHOLD,
    extra_fields: Optional[dict] = None,
) -> None:
    """
    Persist the raw per-sample predictions behind a run's metrics.

    Kept in a directory of its own rather than inside the metrics JSON for two
    reasons. First, load_simple_bootstrap() infers the bootstrap iteration count
    from `max(len(v) for v in metrics.values() if isinstance(v, list))`, so a
    per-sample array sitting in `metrics` would be mistaken for a metric and
    corrupt that count. Second, aggregate_internal_validation_bootstrap()
    splats every top-level key of each metrics file into one combined JSON, so
    embedding per-sample arrays would inflate aggregated.json across all seeds.

    Filenames follow the same `{mode}__{model}__{dataset}__...` convention as
    metrics and resource profiles, so parse_filename_metadata() recovers
    mode / n_bits / synth_scale / model / dataset from them unchanged.

    Probabilities are written at full precision: ROC curves depend only on the
    ordering of scores, and rounding could introduce ties that alter the curve.
    """
    y_true_list = np.asarray(y_true).tolist()
    y_pred_list = np.asarray(y_pred).tolist()
    y_proba_list = np.asarray(y_proba).tolist() if y_proba is not None else None

    data: dict = {"mode": mode, "model": model_name, "dataset": dataset_name, "split": split}
    if extra_fields:
        data.update(extra_fields)

    data["threshold"] = threshold
    data["n_samples"] = len(y_true_list)
    data["y_true"] = y_true_list
    data["y_proba"] = y_proba_list
    data["y_pred"] = y_pred_list

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Predictions saved → {path}")
