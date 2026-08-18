# src/utils.py
"""Utility functions."""

import copy
import csv
import json
import logging
import os
import random
import re
import time
import uuid
import numpy as np
import yaml
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class NegligibleProbabilityDriftFilter(logging.Filter):
    """
    Collapse pgmpy's float-epsilon "probabilities don't sum to 1" warnings.

    Sampling a Bayesian network renormalises each CPD, and floating-point
    rounding leaves the row sums off by ~1e-16. pgmpy warns every time, which
    produced 510 identical lines in a single sweep and buried the handful of
    real errors in the same logs.

    Rather than silence the message outright, this drops it only when the
    reported drift is within `tolerance` — genuine normalisation problems
    (a drift big enough to distort sampling) still come through. The first
    negligible occurrence is kept, annotated with a note that the rest are
    being suppressed, so the log records that it happened.

    Attach to the root handlers via `install_log_filters()`.
    """

    _PATTERN = re.compile(
        r"Probability values don't exactly sum to 1.*?"
        r"Differ by:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    def __init__(self, tolerance: float = 1e-9):
        super().__init__()
        self.tolerance = tolerance
        self.suppressed = 0
        self._announced = False

    def filter(self, record: logging.LogRecord) -> bool:
        match = self._PATTERN.search(record.getMessage())
        if not match:
            return True
        try:
            drift = abs(float(match.group(1)))
        except (TypeError, ValueError):
            return True

        # Large drift is a real problem — never hide it.
        if drift > self.tolerance:
            return True

        if not self._announced:
            self._announced = True
            record.msg = (
                f"{record.getMessage()} "
                f"[further warnings within {self.tolerance:g} suppressed]"
            )
            record.args = ()
            return True

        self.suppressed += 1
        return False


def install_log_filters(tolerance: float = 1e-9) -> NegligibleProbabilityDriftFilter:
    """
    Attach the noise filters to the root logger's handlers.

    Handler-level rather than logger-level: these records originate inside
    third-party libraries under logger names we do not control, and a filter on
    a Logger does not apply to records propagated up from its children. Every
    record still has to pass the root handler to be emitted, so filtering there
    catches them regardless of which library logged them.
    """
    drift_filter = NegligibleProbabilityDriftFilter(tolerance=tolerance)
    for handler in logging.getLogger().handlers:
        handler.addFilter(drift_filter)
    return drift_filter


@contextmanager
def atomic_path(path):
    """
    Yield a temporary path to write to, then atomically move it into place.

    Several output paths are shared by concurrently running jobs: every job
    rewrites data/processed/{dataset}.csv during preprocessing, and the three
    per-model jobs for a given (dataset, synthesizer) pair all regenerate the
    same data/processed/{synth}_{scale}__{dataset}.csv. A plain to_csv() opens
    the destination in truncate mode, so a concurrent reader can observe the
    file as zero-length or half-written — surfacing as "No columns to parse
    from file" or as NaNs where a final row was cut mid-line.

    Writing to a unique temporary file and renaming closes that window:
    os.replace is atomic, so a reader sees either the previous complete file
    or the new complete one, never a partial state. The temp name carries the
    pid and a random suffix so concurrent writers cannot collide on it either.

    Note this guarantees integrity, not agreement: the synthesizers are not
    seeded, so concurrent jobs write *different* valid datasets and the last
    writer wins. See the note in config/synthesizers.yaml.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        yield tmp
        _replace_with_retry(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _replace_with_retry(tmp: Path, path: Path, attempts: int = 10, delay: float = 0.05) -> None:
    """
    os.replace(tmp, path), retrying briefly on Windows.

    On POSIX — which is where the experiments run — replacing a file that a
    reader currently has open always succeeds: the reader keeps its handle on
    the old inode and the rename is atomic. Windows instead refuses with
    PermissionError while any handle is open, so a concurrent read during a
    local run would otherwise crash the writer. Retrying covers that window;
    on POSIX the first attempt always succeeds and this costs nothing.
    """
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


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
        preprocessing__diabetes.json                    → mode=preprocessing, dataset=diabetes
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
    elif len(parts) == 2:
        # Stage profiles with no model slot, i.e. preprocessing__{dataset}. The
        # dataset used to come back None here, which is why preprocessing costs
        # could not be joined onto anything.
        meta["dataset"] = parts[1]

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

# Resource-profiling columns appended to each metrics row, in output order.
# Mirrors the field set that src/visualization.py extracts from a profile, so
# the CSV and the figures draw the same numbers from the same JSON.
_METRICS_CSV_RESOURCE_COLUMNS = [
    "train_time",
    "synth_fit_time",
    "fhe_fit_time",
    "fhe_compile_time",
    "inf_time_total",
    "inf_time_per_sample",
    "mem_train_avg",
    "mem_train_peak",
    "mem_inf_avg",
    "mem_inf_peak",
    "model_size_mb",
    "data_size_mb",
    "circuit_complexity",
]

# Stage columns, joined from the profiles that have no metrics file of their own.
# Appended after the per-run columns above so the existing column order is intact.
_METRICS_CSV_STAGE_COLUMNS = [
    "synth_stage_time",
    "synth_sample_time",
    "mem_synth_peak",
    "mem_sample_peak",
    "preprocess_time",
    "mem_preprocess_peak",
    "one_time_cost",
]

_METRICS_CSV_ALL_RESOURCE_COLUMNS = _METRICS_CSV_RESOURCE_COLUMNS + _METRICS_CSV_STAGE_COLUMNS


def _extract_resource_columns(profile: dict) -> dict:
    """Flatten a resource-profile JSON (as written by ResourceProfiler.save) into
    the scalar `_METRICS_CSV_RESOURCE_COLUMNS`. Missing sub-keys yield None."""
    # Timing columns are sourced from CPU time (training_cpu_time / the
    # inference_time.cpu_* fields), not wall clock: on the shared cluster
    # wall-clock timings absorb node-contention noise, so CPU seconds are the
    # contention-robust basis for comparing modes.
    training_cpu_time = profile.get("training_cpu_time", {}) or {}
    inference_time    = profile.get("inference_time", {}) or {}
    memory            = profile.get("memory", {}) or {}
    storage           = profile.get("storage", {}) or {}
    fhe               = profile.get("fhe", {}) or {}
    mem_train = memory.get("training", {}) or {}
    mem_inf   = memory.get("inference", {}) or {}

    return {
        # train_time sums every timed training block (fit, compile, ...) in CPU
        # seconds, matching the aggregation in src/visualization.py.
        "train_time":          round(sum(training_cpu_time.values()), 4) if training_cpu_time else None,
        # Left None here on purpose: synthesis_fit is timed in the generator's own
        # profile, never in a classifier's, so this read was always returning None
        # and the column sat empty on every row. join_stage_columns fills it.
        "synth_fit_time":      None,
        "fhe_fit_time":        training_cpu_time.get("training_fit"),
        "fhe_compile_time":    training_cpu_time.get("training_compile"),
        "inf_time_total":      inference_time.get("cpu_total"),
        "inf_time_per_sample": inference_time.get("cpu_per_sample"),
        "mem_train_avg":       mem_train.get("average_mb"),
        "mem_train_peak":      mem_train.get("peak_mb"),
        "mem_inf_avg":         mem_inf.get("average_mb"),
        "mem_inf_peak":        mem_inf.get("peak_mb"),
        "model_size_mb":       storage.get("model_size_mb"),
        "data_size_mb":        storage.get("data_size_mb"),
        "circuit_complexity":  fhe.get("circuit_complexity"),
    }


def _load_profile_json(path):
    """Load the first complete JSON object in a profile file, ignoring trailing bytes."""
    content = Path(path).read_text(encoding="utf-8").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(content)
        return obj


def load_stage_profiles(profiles_dir: str = "results/resource_profiles") -> dict:
    """
    Costs of the pipeline stages that produce no metrics file of their own.

    Both readers of the profile directory look a profile up by the exact name of a
    (mode, model, dataset) run, so the stages that are not a classifier run were
    dropped on the floor: the generator fit (`{gen}__synthesis__{ds}.json`), the draw
    of the synthetic training set (`{gen}_{scale}__sampling__{ds}.json`), and
    preprocessing (`preprocessing__{ds}.json`). That is why `synth_fit_time` was empty
    on all 576 rows, and why the one-time cost of a synthetic mode had to be
    reconstructed by re-globbing this directory at figure/table render time.

    Timing is CPU seconds, matching _extract_resource_columns: the shared cluster's
    wall clock absorbs node contention.

    Returns {"synthesis": {(generator, dataset): {...}},
             "sampling":  {(generator, scale, dataset): {...}},
             "preprocessing": {dataset: {...}}}
    """
    stages = {"synthesis": {}, "sampling": {}, "preprocessing": {}}
    profiles_path = Path(profiles_dir)
    if not profiles_path.is_dir():
        return stages

    for path in sorted(profiles_path.glob("*.json")):
        meta = parse_filename_metadata(path.name)
        model, dataset = meta["model"], meta["dataset"]
        if dataset is None:
            continue
        if model == "synthesis":
            key, bucket, phase = (meta["mode"], dataset), "synthesis", "synthesis"
        elif model == "sampling":
            key, bucket, phase = (meta["mode"], meta["synth_scale"], dataset), "sampling", "sampling"
        elif meta["mode"] == "preprocessing" and model is None:
            key, bucket, phase = dataset, "preprocessing", "processing"
        else:
            continue

        profile = _load_profile_json(path)
        cpu_blocks = profile.get("training_cpu_time", {}) or {}
        memory = (profile.get("memory", {}) or {}).get(phase, {}) or {}
        stages[bucket][key] = {
            # Every timed block of the stage, not just the headline one: loading the
            # real data and persisting the fitted generator are also paid before the
            # first prediction. This is the sum the radar figure has always used.
            "cpu_time": round(sum(cpu_blocks.values()), 4) if cpu_blocks else None,
            "fit_time": cpu_blocks.get("synthesis_fit"),
            "mem_peak": memory.get("peak_mb"),
        }
    return stages


def join_stage_columns(meta: dict, train_time, stages: dict) -> dict:
    """
    The `_METRICS_CSV_STAGE_COLUMNS` for one classifier row.

    `meta` is that row's parsed filename metadata and `train_time` its own summed
    training blocks. The generator is fit once per dataset and sampled once per
    (dataset, scale), so each is a dataset-level constant charged to all three
    downstream classifier cells -- the convention the radar figure established.

    Sampling is matched on the row's OWN synth_scale rather than pinned to 100, so a
    scale-300 row is charged the scale-300 draw it actually used.

    `one_time_cost` is everything paid before the first prediction, one canonical
    definition for every mode: classifier fit for standard, generator fit + draw +
    classifier fit for the synthetic modes, and model fit + circuit compilation for
    FHE (whose train_time already sums both, since they are timed in its own profile).
    """
    mode, dataset = meta["mode"], meta["dataset"]
    synthesis = stages.get("synthesis", {}).get((mode, dataset), {})
    sampling = stages.get("sampling", {}).get((mode, meta["synth_scale"], dataset), {})
    preprocessing = stages.get("preprocessing", {}).get(dataset, {})

    synth_stage_time = synthesis.get("cpu_time")
    synth_sample_time = sampling.get("cpu_time")
    one_time = None if train_time is None else float(train_time)
    if one_time is not None:
        one_time = round(one_time + (synth_stage_time or 0.0) + (synth_sample_time or 0.0), 4)

    return {
        "synth_fit_time": synthesis.get("fit_time"),
        "synth_stage_time": synth_stage_time,
        "synth_sample_time": synth_sample_time,
        "mem_synth_peak": synthesis.get("mem_peak"),
        "mem_sample_peak": sampling.get("mem_peak"),
        "preprocess_time": preprocessing.get("cpu_time"),
        "mem_preprocess_peak": preprocessing.get("mem_peak"),
        "one_time_cost": one_time,
    }


def aggregate_metrics_csv(
    metrics_dir: str = "results/metrics",
    output_path: str = "results/metrics_aggregated.csv",
    profiles_dir: str = "results/resource_profiles",
):
    """Aggregate every `{mode}__{model}__{dataset}__test__metrics.json` file in
    metrics_dir into a single CSV with one row per (mode, dataset, model).

    `mode` is taken verbatim from the filename's leading `__`-delimited segment,
    so synthetic-data scale suffixes (arf_100, ctgan_300, ...) and FHE bit-widths
    (fhe_2, fhe_12, ...) are kept as distinct mode values rather than collapsed
    into a shared method name.

    Each metric gets three columns — `{metric}_median`, `{metric}_ci_low`,
    `{metric}_ci_high` — computed from that file's n=1000 bootstrap distribution:
    the median, and the 95% CI via the 2.5th/97.5th percentiles.

    The point estimate is the MEDIAN, project-wide: every figure, table, and
    manuscript sentence aggregates with medians (see tmp/manuscript/results_summary.md).
    Resource costs are single measurements with a heavy right tail from cluster
    contention, and bootstrap replicate distributions are mildly left-skewed, so a
    mean summarises neither well. The percentile CI is unchanged and pairs naturally
    with a median.

    The matching resource profile in `profiles_dir` (`{mode}__{model}__{dataset}.json`,
    as written by ResourceProfiler.save) contributes the scalar
    `_METRICS_CSV_RESOURCE_COLUMNS` on the same row. Rows with no profile file get
    None for every resource column.

    The stages that write a profile but no metrics file -- the generator fit, the
    synthetic draw, preprocessing -- are joined on as `_METRICS_CSV_STAGE_COLUMNS` by
    load_stage_profiles/join_stage_columns, as extra COLUMNS on the classifier rows
    that inherit those costs rather than as extra rows. Extra rows would land inside
    the `df[df["mode"] == m]` filters that scripts/render_tables.py medians over and
    silently shift the published numbers; the row grain stays one per
    (mode, dataset, classifier).
    """
    fieldnames = ["mode", "dataset", "model"]
    for metric in _METRICS_CSV_METRIC_NAMES:
        fieldnames += [f"{metric}_median", f"{metric}_ci_low", f"{metric}_ci_high"]
    fieldnames += _METRICS_CSV_ALL_RESOURCE_COLUMNS

    profiles_path = Path(profiles_dir)
    stages = load_stage_profiles(profiles_dir)
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
                row[f"{metric}_median"] = None
                row[f"{metric}_ci_low"] = None
                row[f"{metric}_ci_high"] = None
                continue

            arr = np.asarray(values, dtype=float)
            ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
            row[f"{metric}_median"] = round(float(np.median(arr)), 4)
            row[f"{metric}_ci_low"] = round(float(ci_low), 4)
            row[f"{metric}_ci_high"] = round(float(ci_high), 4)

        profile_path = profiles_path / f"{mode}__{model}__{dataset}.json"
        if profile_path.exists():
            profile = _load_profile_json(profile_path)
            row.update(_extract_resource_columns(profile))
        else:
            logger.warning(f"No resource profile for {mode}/{model}/{dataset}: {profile_path.name}")
            row.update({col: None for col in _METRICS_CSV_RESOURCE_COLUMNS})

        # After the per-run columns: this fills synth_fit_time, which no classifier
        # profile can supply, and derives one_time_cost from that row's train_time.
        row.update(join_stage_columns(parse_filename_metadata(path.name),
                                      row.get("train_time"), stages))

        rows.append(row)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Aggregated {len(rows)} mode/dataset/model rows to {output_path}")