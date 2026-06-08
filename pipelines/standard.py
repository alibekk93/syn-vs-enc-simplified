"""Standard training pipeline — trains and evaluates models on processed data."""

import logging
import time
from src.utils import load_config
from src.models import Model
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG = "config/datasets.yaml"
MODELS_CFG   = "config/models.yaml"
RESOURCE_CFG = "config/resource_profiling.yaml"


def run(datasets: list[str] | None = None, models: list[str] | None = None, datasets_config: str = "config/datasets.yaml", resource_config: str = "config/resource_profiling.yaml", models_config: str = "config/models.yaml") -> dict:
    """
    Args:
        datasets: List of dataset names (default: all).
        models:   List of model names (default: all in config).
        datasets_config: Path to datasets configuration file.
        resource_config: Path to resource profiling configuration file.
        models_config: Path to models configuration file.

    Returns:
        Nested dict of {dataset: {model: metrics}}
    """
    targets_datasets = datasets or list(load_config(datasets_config).keys())
    targets_models   = models   or [m["name"] for m in load_config(models_config).get("models", [])]

    logger.info(f"Standard pipeline started — datasets: {targets_datasets}, models: {targets_models}")

    results = {}
    for dataset_name in targets_datasets:
        results[dataset_name] = {}
        for model_name in targets_models:
            logger.info(f"--- {model_name} on {dataset_name} ---")

            profiler = ResourceProfiler(load_config(resource_config))

            try:
                model = Model(model_name, cfg=models_config, mode="standard")

                # Explicit phase labels so training and inference
                # memory snapshots are stored separately.
                profiler.start_memory_sampling(phase="training")

                with profiler.time_block("data_loading"):
                    model.load_data(dataset_name, dataset_cfg=datasets_config)
                    model.split()

                with profiler.time_block("training"):
                    model.train()

                profiler.stop_memory_sampling()

                profiler.start_memory_sampling(phase="inference")

                start   = time.time()
                metrics = model.evaluate()
                end     = time.time()

                profiler.log_inference(end - start, len(model.X_test))
                profiler.stop_memory_sampling()

                model_path = f"models/standard__{model_name}__{dataset_name}.joblib"
                data_path  = f"data/processed/{dataset_name}.csv"
                profiler.log_storage(model_path=model_path, data_path=data_path)

                results[dataset_name][model_name] = {
                    "metrics":   metrics,
                    "profiling": profiler.export(),
                }

                # Persist profiling results to disk.
                profiler.save(f"standard__{model_name}__{dataset_name}")
                profiler.reset()

            except Exception as e:
                logger.error(f"Failed: {model_name} on {dataset_name}: {e}")
                results[dataset_name][model_name] = {
                    "error":     str(e),
                    "profiling": profiler.export(),
                }
                profiler.reset()

    logger.info("Standard pipeline complete.")
    return results