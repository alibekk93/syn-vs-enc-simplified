"""Bootstrap evaluation pipeline — resamples data and runs full experiment pipeline on each sample."""
import gc
import ctypes
import logging
import json
import pandas as pd
from pathlib import Path
import shutil
import yaml
import tempfile
import os

from src.utils import load_config

logger = logging.getLogger(__name__)


def _malloc_trim() -> None:
    """Ask glibc to return freed arenas to the OS. No-op on non-glibc platforms."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


DATASETS_CFG = "config/datasets.yaml"
MODELS_CFG   = "config/models.yaml"
RESOURCE_CFG = "config/resource_profiling.yaml"
SYNTH_CFG    = "config/synthesizers.yaml"


def _bootstrap_dataframe(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Resample dataframe with replacement (same size)."""
    return df.sample(n=len(df), replace=True, random_state=seed).reset_index(drop=True)


def _create_bootstrap_configs(seed: int, datasets: list[str]) -> dict:
    """
    Create bootstrap-specific configs for datasets, models, resource, synthesizers.

    Returns:
        dict with keys: datasets, models, resource, synthesizers -> Path
    """
    # Create temporary directory for configs
    tmp_dir = Path(f"tmp/bootstrap_configs/{seed}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Load base configs
    base_datasets = load_config(DATASETS_CFG)
    base_models   = load_config(MODELS_CFG)
    base_resource = load_config(RESOURCE_CFG)
    base_synth    = load_config(SYNTH_CFG)

    # Modify datasets config: add raw_path and processed_path for each dataset.
    # Both paths are seed-namespaced so concurrent SLURM array tasks (different seeds,
    # same dataset) do not clobber each other's files.
    bootstrap_raw_dir       = Path(f"data/bootstrap/{seed}")
    bootstrap_processed_dir = Path(f"data/processed/{seed}")
    bootstrap_raw_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_processed_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        if ds_name in base_datasets:
            base_datasets[ds_name]["raw_path"]       = str(bootstrap_raw_dir       / f"{ds_name}.csv")
            base_datasets[ds_name]["processed_path"] = str(bootstrap_processed_dir / f"{ds_name}.csv")
        else:
            logger.warning(f"Dataset '{ds_name}' not found in base config; skipping raw_path override.")

    # Write modified datasets config
    dataset_config_path = tmp_dir / "datasets.yaml"
    with open(dataset_config_path, 'w') as f:
        yaml.dump(base_datasets, f, default_flow_style=False)

    # Modify models config: set output directories under bootstrap seed
    if "output" not in base_models:
        base_models["output"] = {}
    base_models["output"]["results_dir"] = f"results/bootstrap/{seed}/metrics"
    base_models["output"]["models_dir"]  = f"models/bootstrap/{seed}"

    model_config_path = tmp_dir / "models.yaml"
    with open(model_config_path, 'w') as f:
        yaml.dump(base_models, f, default_flow_style=False)

    # Modify resource config: set output_dir to bootstrap seed location
    if "logging" not in base_resource:
        base_resource["logging"] = {}
    base_resource["logging"]["output_dir"] = f"results/bootstrap/{seed}/resource_profiles"

    resource_config_path = tmp_dir / "resource_profiling.yaml"
    with open(resource_config_path, 'w') as f:
        yaml.dump(base_resource, f, default_flow_style=False)

    # Set synthetic_dir to the seed-namespaced processed directory so synthetic
    # CSVs land alongside the preprocessed real data for this seed.  Model.load_data()
    # for synthetic datasets (e.g. "ctgan__diabetes") resolves the path as
    # Path(processed_path).parent / f"{synth}__{dataset}.csv", which matches.
    if "output" not in base_synth:
        base_synth["output"] = {}
    base_synth["output"]["synthetic_dir"] = str(bootstrap_processed_dir)

    synth_config_path = tmp_dir / "synthesizers.yaml"
    with open(synth_config_path, 'w') as f:
        yaml.dump(base_synth, f, default_flow_style=False)

    return {
        "datasets": dataset_config_path,
        "models":   model_config_path,
        "resource": resource_config_path,
        "synthesizers": synth_config_path,
    }


def run(
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    synthesizers: list[str] | None = None,
    seed: int = 42,
    evaluation: dict | None = None,
    fhe_mode: str = "simulate",
    n_bits: int | None = None,
    device: str | None = None,
    pipelines_cfg: dict | None = None,
) -> dict:
    """
    Bootstrap evaluation pipeline that resamples data and runs full experiment pipeline.

    Steps for each bootstrap iteration:
        - Resample raw data with replacement (bootstrap sample)
        - Save bootstrap sample to data/bootstrap/{seed}/{dataset}.csv
        - Create dataset config pointing to bootstrap samples
        - Create model config with output dirs under results/bootstrap/{seed} and models/bootstrap/{seed}
        - Create resource config with output dir under results/bootstrap/{seed}/resource_profiles
        - Run preprocessing on bootstrap sample
        - Run standard modeling on preprocessed bootstrap sample
        - Run synthetic modeling on preprocessed bootstrap sample
        - Run FHE modeling on preprocessed bootstrap sample
        - Collect and store metrics

    Args:
        datasets: List of dataset names to process. If None, all datasets are processed.
        models:   List of model names to use. If None, all models from config are used.
        synthesizers: List of synthesizer names to use. If None, all synthesizers
            from config are used.
        seed:     Random seed for bootstrap sampling.
        evaluation: Optional evaluation configuration (not currently used).
        n_bits: Override n_bits for every FHE model (passed straight through to
            `fhe.run`). Lets the FHE stage of a given seed be parallelized across
            n_bits values as separate processes — see `main.py`'s
            `run-single-bootstrap --n-bits N` and `list-n-bits`.
        device: Override the compute device ("cpu" or "cuda"), passed straight
            through to `standard.run`, `synthetic.run`, and `fhe.run`. Each
            pipeline applies it only to the models/synthesizers that actually
            have a GPU path. See the `device` key in config/models.yaml,
            config/synthesizers.yaml, and config/fhe.yaml.
        pipelines_cfg: Which stages to run, e.g. {"preprocessing": True, "raw": True,
            "synthetic": True, "fhe": True}. Stages are imported lazily so a stage's
            heavy dependencies (e.g. concrete.ml for fhe, sdv/synthcity for synthetic)
            are only required when that stage is enabled. If None, all stages run.

    Returns:
        dict: {dataset: {seed: {pipeline: {model: results}}}}
    """
    if pipelines_cfg is None:
        pipelines_cfg = {"preprocessing": True, "raw": True, "synthetic": True, "fhe": True}

    targets_datasets = datasets or list(load_config(DATASETS_CFG).keys())
    targets_models   = models   or [m["name"] for m in load_config(MODELS_CFG).get("models", [])]

    logger.info(
        f"Bootstrap pipeline started — datasets: {targets_datasets}, "
        f"models: {targets_models}, seed: {seed}"
    )

    # Create bootstrap-specific configs
    configs = _create_bootstrap_configs(seed, targets_datasets)
    logger.info(f"Using dataset config: {configs['datasets']}")
    logger.info(f"Using models config: {configs['models']}")
    logger.info(f"Using resource config: {configs['resource']}")
    logger.info(f"Using synthesizers config: {configs['synthesizers']}")

    results = {}

    for dataset_name in targets_datasets:
        results[dataset_name] = {}

        # Load original raw data
        raw_path = Path("data/raw") / f"{dataset_name}.csv"
        if not raw_path.exists():
            logger.error(f"Raw data not found for dataset {dataset_name} at {raw_path}")
            results[dataset_name]["error"] = f"Raw data not found: {raw_path}"
            continue

        df_original = pd.read_csv(raw_path)
        logger.info(f"Loaded original raw data for {dataset_name}: {df_original.shape}")

        # Create bootstrap sample
        df_boot = _bootstrap_dataframe(df_original, seed)
        logger.info(f"Created bootstrap sample for {dataset_name}: {df_boot.shape}")

        # Save bootstrap sample to the designated location
        bootstrap_raw_dir = Path(f"data/bootstrap/{seed}")
        bootstrap_raw_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_raw_path = bootstrap_raw_dir / f"{dataset_name}.csv"
        df_boot.to_csv(bootstrap_raw_path, index=False)
        logger.info(f"Saved bootstrap sample to {bootstrap_raw_path}")

        try:
            prep_results, std_results, synth_results, fhe_results = {}, {}, {}, {}

            if pipelines_cfg.get("preprocessing"):
                logger.info(f"Running preprocessing on bootstrap sample for {dataset_name}")
                from pipelines import preprocessing
                prep_results = preprocessing.run(
                    datasets=[dataset_name],
                    datasets_config=str(configs["datasets"]),
                    resource_config=str(configs["resource"])
                )

            if pipelines_cfg.get("raw"):
                logger.info(f"Running standard modeling on bootstrap sample for {dataset_name}")
                from pipelines import standard
                std_results = standard.run(
                    datasets=[dataset_name],
                    models=targets_models,
                    device=device,
                    datasets_config=str(configs["datasets"]),
                    resource_config=str(configs["resource"]),
                    models_config=str(configs["models"])
                )

            if pipelines_cfg.get("synthetic"):
                logger.info(f"Running synthetic modeling on bootstrap sample for {dataset_name}")
                from pipelines import synthetic
                synth_results = synthetic.run(
                    datasets=[dataset_name],
                    synthesizers=synthesizers,
                    models=targets_models,
                    device=device,
                    datasets_config=str(configs["datasets"]),
                    resource_config=str(configs["resource"]),
                    models_config=str(configs["models"]),
                    synthesizers_config=str(configs["synthesizers"])
                )

            if pipelines_cfg.get("fhe"):
                logger.info(f"Running FHE modeling on bootstrap sample for {dataset_name}")
                from pipelines import fhe
                fhe_results = fhe.run(
                    datasets=[dataset_name],
                    models=targets_models,
                    datasets_config=str(configs["datasets"]),
                    resource_config=str(configs["resource"]),
                    models_config=str(configs["models"]),
                    fhe_mode=fhe_mode,
                    n_bits=n_bits,
                    device=device,
                )
                # Dataset boundary: fhe.run trims per-model inside its own finally
                # blocks, but any remaining glibc arena pages are released here so
                # the next dataset starts with a clean RSS baseline.
                gc.collect()
                _malloc_trim()

            # Collect results
            results[dataset_name][seed] = {
                "preprocessing": prep_results.get(dataset_name, {}),
                "standard": std_results.get(dataset_name, {}),
                "synthetic": synth_results.get(dataset_name, {}),
                "fhe": fhe_results.get(dataset_name, {}),
            }

            logger.info(f"Completed bootstrap iteration for {dataset_name} with seed {seed}")

        except Exception as e:
            logger.error(f"Failed running experiment pipeline on bootstrap sample for {dataset_name}: {e}")
            results[dataset_name][seed] = {
                "error": str(e),
                "preprocessing": {},
                "standard": {},
                "synthetic": {},
                "fhe": {},
            }

        # Optionally remove the bootstrap sample after processing to save space?
        # Keep it for potential inspection; but we can comment out removal.
        # bootstrap_raw_path.unlink(missing_ok=True)

    logger.info("Bootstrap pipeline complete.")
    return results