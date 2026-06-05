"""Bootstrap evaluation pipeline — evaluates trained models on bootstrapped datasets."""
import logging
import json
import pandas as pd
from pathlib import Path

from src.utils import load_config
from src.models import Model
from src.fhe_models import FHEModel
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG = "config/datasets.yaml"
MODELS_CFG   = "config/models.yaml"
RESOURCE_CFG = "config/resource_profiling.yaml"

MODELS_DIR   = Path("models")
DATA_DIR     = Path("data/processed")

BOOT_METRICS_DIR = Path("results/bootstrapped/metrics")
BOOT_PROFILE_DIR = Path("results/bootstrapped/resource_profiles")


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def _find_model_paths(model_name: str, dataset_name: str):
    """
    Find all saved model files for a given model/dataset pair.

    Supports:
        - standard / synthetic (.joblib)
        - fhe (.json)

    Returns:
        list of (mode, path, type)
            type ∈ {"standard", "fhe"}
    """
    paths = []

    # Standard / synthetic
    for p in MODELS_DIR.glob(f"*__{model_name}__{dataset_name}.joblib"):
        parts = p.stem.split("__")
        if len(parts) >= 3:
            mode = parts[0]
            paths.append((mode, p, "standard"))

    # FHE
    for p in MODELS_DIR.glob(f"*__{model_name}__{dataset_name}.json"):
        parts = p.stem.split("__")
        if len(parts) >= 3:
            mode = parts[0]  # e.g. fhe or fhe_4
            paths.append((mode, p, "fhe"))

    return paths


def _bootstrap_dataframe(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Resample dataframe with replacement (same size)."""
    return df.sample(n=len(df), replace=True, random_state=seed).reset_index(drop=True)


def _save_json(path: Path, content: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f, indent=2)


def _load_fhe_model(model_name: str, path: Path):
    """
    Properly load FHE model from JSON using Concrete ML loader.
    """
    from concrete.ml.common.serialization.loaders import load as cml_load

    with open(path, "r") as f:
        model_obj = cml_load(f)

    # Wrap in FHEModel instance
    instance = FHEModel(model_name)  # initialize properly
    instance.model = model_obj
    return instance


# --------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------
def run(
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    seed: int = 42,
    evaluation: dict | None = None,
) -> dict:
    """
    Bootstrap evaluation pipeline with FHE support.

    Steps:
        - Identify datasets/models
        - Load trained models (standard + FHE)
        - Bootstrap datasets
        - Evaluate on bootstrapped data
        - Store metrics + resource profiles

    Returns:
        dict: {dataset: {model: {mode: results}}}
    """
    targets_datasets = datasets or list(load_config(DATASETS_CFG).keys())
    targets_models   = models   or [m["name"] for m in load_config(MODELS_CFG).get("models", [])]

    logger.info(
        f"Bootstrap pipeline started — datasets: {targets_datasets}, "
        f"models: {targets_models}, seed: {seed}"
    )

    results = {}

    for dataset_name in targets_datasets:
        results[dataset_name] = {}

        data_path = DATA_DIR / f"{dataset_name}.csv"
        df_original = pd.read_csv(data_path)

        for model_name in targets_models:
            results[dataset_name][model_name] = {}

            model_entries = _find_model_paths(model_name, dataset_name)

            if not model_entries:
                logger.warning(f"No saved models found for {model_name} on {dataset_name}")
                continue

            for mode, model_path, mtype in model_entries:
                logger.info(f"--- Bootstrap {mode}::{model_name} on {dataset_name} ---")

                profiler = ResourceProfiler(load_config(RESOURCE_CFG))

                try:
                    # --------------------------------------------------
                    # Load model
                    # --------------------------------------------------
                    if mtype == "standard":
                        model = Model.load(str(model_path))
                        is_fhe = False
                    else:
                        model = _load_fhe_model(model_name, model_path)
                        is_fhe = True

                    # --------------------------------------------------
                    # Bootstrap dataset
                    # --------------------------------------------------
                    df_boot = _bootstrap_dataframe(df_original, seed)

                    # Inject data
                    model.df = df_boot
                    model.dataset_name = dataset_name

                    # Resolve target
                    ds_cfg = load_config(DATASETS_CFG)
                    model.target = ds_cfg[dataset_name]["target"]

                    # --------------------------------------------------
                    # Split
                    # --------------------------------------------------
                    with profiler.time_block("data_split"):
                        model.split()

                    # --------------------------------------------------
                    # FHE compile (required before inference)
                    # --------------------------------------------------
                    if is_fhe:
                        model.compile()

                    # --------------------------------------------------
                    # Evaluate
                    # --------------------------------------------------
                    profiler.start_memory_sampling(phase="inference")

                    import time as _time
                    start = _time.time()

                    if is_fhe:
                        metrics = model.evaluate(fhe="simulate")
                    else:
                        metrics = model.evaluate()

                    end = _time.time()

                    profiler.log_inference(end - start, len(model.X_test))
                    profiler.stop_memory_sampling()

                    # FHE metadata
                    if is_fhe:
                        profiler.log_fhe(
                            complexity=getattr(model.model, "circuit_complexity", None)
                        )

                    profiler.log_storage(
                        model_path=str(model_path),
                        data_path=str(data_path),
                    )

                    # --------------------------------------------------
                    # Save results
                    # --------------------------------------------------
                    metrics_obj = {
                        "mode": mode,
                        "model": model_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "metrics": metrics,
                    }

                    metrics_path = (
                        BOOT_METRICS_DIR /
                        f"{mode}__{model_name}__{dataset_name}__{seed}.json"
                    )
                    _save_json(metrics_path, metrics_obj)

                    profile_path = (
                        BOOT_PROFILE_DIR /
                        f"{mode}__{model_name}__{dataset_name}__{seed}.json"
                    )
                    _save_json(profile_path, profiler.export())

                    results[dataset_name][model_name][mode] = {
                        "metrics": metrics,
                        "profiling": profiler.export(),
                    }

                    profiler.reset()

                except Exception as e:
                    logger.error(
                        f"Failed: bootstrap {mode}::{model_name} on {dataset_name}: {e}"
                    )
                    results[dataset_name][model_name][mode] = {
                        "error": str(e),
                        "profiling": profiler.export(),
                    }
                    profiler.reset()

    logger.info("Bootstrap pipeline complete.")
    return results