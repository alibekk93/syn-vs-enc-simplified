# src/utils.py
"""Utility functions."""

import copy
import csv
import json
import logging
import random
import re
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Result-filename parsing. Lives here rather than in visualization.py
# so that numpy-only consumers (src/stats_tests.py) can decompose mode
# strings without pulling in matplotlib/seaborn/pandas.
# ------------------------------------------------------------------

def parse_filename_metadata(filename):
    """
    Extract mode / model / dataset / n_bits / synth_scale from a filename or raw mode key.

    Examples:
        fhe_4__logistic_regression__heart_disease.json  → mode=fhe, n_bits=4
        ctgan_100__rf__diabetes.json                    → mode=ctgan, synth_scale=100
        standard__rf__diabetes.json                     → mode=standard
        ctgan                                           → mode=ctgan  (bare JSON mode key)
    """
    name = Path(filename).stem
    parts = name.split("__")

    meta = {
        "raw_name": name,
        "mode": None,
        "model": None,
        "dataset": None,
        "n_bits": None,
        "synth_scale": None,
    }

    # FHE: fhe_N
    fhe_match = re.match(r"fhe_(\d+)", parts[0])
    if fhe_match:
        meta["mode"] = "fhe"
        meta["n_bits"] = int(fhe_match.group(1))
        parts[0] = "fhe"
    else:
        # Synthetic mode with synth_scale suffix: e.g. ctgan_100, gaussian_copula_150
        synth_match = re.match(r"^(.+)_(\d+)$", parts[0])
        if synth_match:
            meta["mode"] = synth_match.group(1)
            meta["synth_scale"] = int(synth_match.group(2))
        else:
            meta["mode"] = parts[0]

    if len(parts) >= 3:
        meta["model"] = parts[1]
        meta["dataset"] = parts[2]

    return meta


# ------------------------------------------------------------------
# FHE n_bits config helpers (pure config logic, kept dependency-free
# so callers — e.g. `main.py list-n-bits` — don't need concrete-ml
# installed just to expand/inject n_bits values).
# ------------------------------------------------------------------

def expand_n_bits(cfg: dict) -> list:
    """
    Returns a list of n_bits values described by an fhe.yaml-style `sweep` block.

    Behavior:
        - If sweep not defined -> single run ([None])
        - If list provided     -> return list
        - If start/end/step    -> expand range
    """
    sweep_cfg = cfg.get("sweep", {})

    if not sweep_cfg or not sweep_cfg.get("enabled", False):
        return [None]

    nb = sweep_cfg.get("n_bits")

    if nb is None:
        return [None]

    if isinstance(nb, list):
        return nb

    start = nb.get("start")
    end   = nb.get("end")
    step  = nb.get("step", 1)

    if start is None or end is None:
        return [None]

    return list(range(start, end + 1, step))


def expand_synth_scales(cfg: dict) -> list[int]:
    """Returns the list of synth_scale values from a synthesizers.yaml-style config."""
    return list(cfg.get("synth_scale", {}).get("values", [100]))


def inject_n_bits(fhe_cfg: dict, n_bits) -> dict:
    """Returns a copy of fhe_cfg with n_bits injected into every model config."""
    if n_bits is None:
        return fhe_cfg

    new_cfg = copy.deepcopy(fhe_cfg)

    for model_name in new_cfg.get("models", {}):
        new_cfg["models"][model_name]["n_bits"] = n_bits

    return new_cfg


def model_n_bits(fhe_cfg: dict, model_name: str):
    """Looks up the configured n_bits for a single model."""
    return fhe_cfg.get("models", {}).get(model_name, {}).get("n_bits")


# ------------------------------------------------------------------
# Device (GPU) helpers shared by the standard/synthetic pipelines.
# FHE's device check lives in pipelines/fhe.py since it's concrete-ml
# specific (checks the compiler, not torch/CUDA).
# ------------------------------------------------------------------

def check_cuda_available() -> bool:
    """Whether a CUDA-capable GPU is visible to this process via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def require_device(device: str) -> None:
    """Fails fast if `device='cuda'` is requested but unavailable, instead of
    failing deep inside a model/synthesizer's fit call (wasting a GPU job
    allocation)."""
    if device == "cuda" and not check_cuda_available():
        raise RuntimeError(
            "device='cuda' requested, but no CUDA-capable GPU is visible to this "
            "process (or PyTorch isn't installed with CUDA support)."
        )


def generate_seeds(seed: int, length: int):
    """Generate a list of random seeds and save to file."""
    random.seed(seed)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(length)]
    with open("internal_validation_bootstrap_seeds.txt", "w") as f:
        for s in seeds:
            f.write(f"{s}\n")
    logger.info(f"Generated {length} seeds and saved to internal_validation_bootstrap_seeds.txt")


def aggregate_internal_validation_bootstrap(results_dir: str = "results/internal_validation_bootstrap", output_path: str = "results/internal_validation_bootstrap/aggregated.json"):
    """Concatenate all internal validation bootstrap results into a single hierarchical JSON file.

    Output structure:
        {metrics|resource_profiles} -> mode -> model -> dataset -> [per-seed records]

    Resource profile filename conventions handled:
        preprocessing__{dataset}                           -> mode=preprocessing, model=_
        {synthesizer}__{dataset}__synthesis                -> mode=<synthesizer>, model=_synthesis
        {mode}__{model}__{dataset}                         -> standard / fhe_N model files
        synthetic__{synthesizer}__{model}__{dataset}       -> mode=<synthesizer> (aligns with metrics)
    """
    results_path = Path(results_dir)
    output: dict = {"metrics": {}, "resource_profiles": {}}

    def get_leaf(root: dict, mode: str, model: str, dataset: str) -> list:
        return (
            root
            .setdefault(mode, {})
            .setdefault(model, {})
            .setdefault(dataset, [])
        )

    seed_dirs = sorted(
        (d for d in results_path.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not seed_dirs:
        logger.warning(f"No seed directories found in {results_dir}")
        return

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name)

        metrics_dir = seed_dir / "metrics"
        if metrics_dir.exists():
            for f in sorted(metrics_dir.glob("*.json")):
                parts = f.stem.split("__")
                if len(parts) == 5 and parts[3] == "test" and parts[4] == "metrics":
                    mode, model, dataset = parts[0], parts[1], parts[2]
                    data = json.loads(f.read_text(encoding="utf-8"))
                    get_leaf(output["metrics"], mode, model, dataset).append(
                        {"seed": seed, **data}
                    )
                else:
                    logger.warning(f"Unrecognized metrics filename: {f.name}")

        resource_dir = seed_dir / "resource_profiles"
        if resource_dir.exists():
            for f in sorted(resource_dir.glob("*.json")):
                parts = f.stem.split("__")
                data = json.loads(f.read_text(encoding="utf-8"))

                if len(parts) == 2:
                    mode, model, dataset = parts[0], "_", parts[1]
                elif len(parts) == 3 and parts[2] == "synthesis":
                    mode, model, dataset = parts[0], "_synthesis", parts[1]
                elif len(parts) == 3:
                    mode, model, dataset = parts[0], parts[1], parts[2]
                elif len(parts) == 4 and parts[0] == "synthetic":
                    mode, model, dataset = parts[1], parts[2], parts[3]
                else:
                    logger.warning(f"Unrecognized resource profile filename: {f.name}")
                    continue

                get_leaf(output["resource_profiles"], mode, model, dataset).append(
                    {"seed": seed, **data}
                )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Aggregated internal validation bootstrap results saved to {output_path}")


_METRICS_CSV_METRIC_NAMES = ["accuracy", "f1", "roc_auc", "precision", "recall"]


def aggregate_metrics_csv(
    metrics_dir: str = "results/metrics",
    output_path: str = "results/metrics_aggregated.csv",
):
    """Aggregate every `{mode}__{model}__{dataset}__test__metrics.json` file in
    metrics_dir into a single CSV with one row per (mode, dataset, model).

    `mode` is taken verbatim from the filename's leading `__`-delimited segment,
    so synthetic-data scale suffixes (arf_100, ctgan_300, ...) and FHE bit-widths
    (fhe_2, fhe_12, ...) are kept as distinct mode values rather than collapsed
    into a shared method name.

    Each metric gets three columns — `{metric}_mean`, `{metric}_ci_low`,
    `{metric}_ci_high` — computed from that file's n=1000 bootstrap distribution:
    the mean, and the 95% CI via the 2.5th/97.5th percentiles.
    """
    fieldnames = ["mode", "dataset", "model"]
    for metric in _METRICS_CSV_METRIC_NAMES:
        fieldnames += [f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"]

    rows = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        parts = path.stem.split("__")
        if len(parts) != 5 or parts[3] != "test" or parts[4] != "metrics":
            logger.warning(f"Unrecognized metrics filename: {path.name}")
            continue
        mode, model, dataset = parts[0], parts[1], parts[2]

        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})

        row = {"mode": mode, "dataset": dataset, "model": model}
        for metric in _METRICS_CSV_METRIC_NAMES:
            values = metrics.get(metric)
            if not values:
                row[f"{metric}_mean"] = None
                row[f"{metric}_ci_low"] = None
                row[f"{metric}_ci_high"] = None
                continue

            arr = np.asarray(values, dtype=float)
            ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
            row[f"{metric}_mean"] = round(float(arr.mean()), 4)
            row[f"{metric}_ci_low"] = round(float(ci_low), 4)
            row[f"{metric}_ci_high"] = round(float(ci_high), 4)

        rows.append(row)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Aggregated {len(rows)} mode/dataset/model rows to {output_path}")