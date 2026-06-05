"""Bootstrap evaluation pipeline — resamples data and runs full experiment pipeline on each sample."""
import logging
import json
import pandas as pd
from pathlib import Path
import shutil

from src.utils import load_config
from pipelines import preprocessing, standard, synthetic, fhe

logger = logging.getLogger(__name__)

DATASETS_CFG = "config/datasets.yaml"
MODELS_CFG   = "config/models.yaml"
RESOURCE_CFG = "config/resource_profiling.yaml"


def _bootstrap_dataframe(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Resample dataframe with replacement (same size)."""
    return df.sample(n=len(df), replace=True, random_state=seed).reset_index(drop=True)


def run(
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    seed: int = 42,
    evaluation: dict | None = None,
) -> dict:
    """
    Bootstrap evaluation pipeline that resamples data and runs full experiment pipeline.

    Steps for each bootstrap iteration:
        - Resample raw data with replacement (bootstrap sample)
        - Run preprocessing on bootstrap sample
        - Run standard modeling on preprocessed bootstrap sample
        - Run synthetic modeling on preprocessed bootstrap sample
        - Run FHE modeling on preprocessed bootstrap sample
        - Collect and store metrics

    Args:
        datasets: List of dataset names to process. If None, all datasets are processed.
        models:   List of model names to use. If None, all models from config are used.
        seed:     Random seed for bootstrap sampling.
        evaluation: Optional evaluation configuration (not currently used).

    Returns:
        dict: {dataset: {seed: {pipeline: {model: results}}}}
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

        # Backup original raw file
        backup_raw_path = Path("data/raw") / f"{dataset_name}.csv.backup"
        try:
            # Move original raw file to backup
            shutil.move(str(raw_path), str(backup_raw_path))

            # Save bootstrap sample as the current raw data file
            df_boot.to_csv(raw_path, index=False)
            logger.info(f"Temporarily replaced raw data for {dataset_name} with bootstrap sample")

            # Now run the full experiment pipeline on this bootstrap sample
            # Using the original dataset name so it matches config entries
            try:
                # Run preprocessing
                logger.info(f"Running preprocessing on bootstrap sample for {dataset_name}")
                prep_results = preprocessing.run(datasets=[dataset_name])

                # Run standard modeling
                logger.info(f"Running standard modeling on bootstrap sample for {dataset_name}")
                std_results = standard.run(
                    datasets=[dataset_name],
                    models=targets_models
                )

                # Run synthetic modeling
                logger.info(f"Running synthetic modeling on bootstrap sample for {dataset_name}")
                synth_results = synthetic.run(
                    datasets=[dataset_name],
                    models=targets_models
                )

                # Run FHE modeling
                logger.info(f"Running FHE modeling on bootstrap sample for {dataset_name}")
                fhe_results = fhe.run(
                    datasets=[dataset_name],
                    models=targets_models
                )

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
                    "fhe": {}
                }
            finally:
                # Restore original raw data
                try:
                    # Remove the bootstrap sample from raw data location
                    if raw_path.exists():
                        raw_path.unlink()

                    # Restore original raw data from backup
                    if backup_raw_path.exists():
                        shutil.move(str(backup_raw_path), str(raw_path))
                        logger.info(f"Restored original raw data for {dataset_name}")
                    else:
                        logger.error(f"Backup raw file not found for {dataset_name}: {backup_raw_path}")
                except Exception as e:
                    logger.error(f"Failed to restore original raw data for {dataset_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to setup bootstrap sample for {dataset_name}: {e}")
            results[dataset_name]["error"] = str(e)
            # Try to restore backup if it exists
            if backup_raw_path.exists():
                try:
                    if raw_path.exists():
                        raw_path.unlink()
                    shutil.move(str(backup_raw_path), str(raw_path))
                except Exception as restore_e:
                    logger.error(f"Failed to restore backup during error handling: {restore_e}")

    logger.info("Bootstrap pipeline complete.")
    return results