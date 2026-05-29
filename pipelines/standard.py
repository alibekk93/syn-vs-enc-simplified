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


def run(datasets: list[str] | None = None, models: list[str] | None = None) -> dict:
    """
    Args:
        datasets: List of dataset names (default: all).
        models:   List of model names (default: all in config).

    Returns:
        Nested dict of {dataset: {model: metrics}}
    """
    profiler = ResourceProfiler(load_config(RESOURCE_CFG))

    targets_datasets = datasets or list(load_config(DATASETS_CFG).keys())
    targets_models   = models   or [m["name"] for m in load_config(MODELS_CFG).get("models", [])]

    logger.info(f"Standard pipeline started — datasets: {targets_datasets}, models: {targets_models}")

    results = {}
    for dataset_name in targets_datasets:
        results[dataset_name] = {}
        for model_name in targets_models:
            logger.info(f"--- {model_name} on {dataset_name} ---")
            try:
                model = Model(model_name, cfg=MODELS_CFG, mode="standard")

                # Start memory sampling for training phase
                profiler.start_memory_sampling()

                # Time data loading
                with profiler.time_block("data_loading"):
                    model.load_data(dataset_name)
                    model.split()

                # Time training
                with profiler.time_block("training"):
                    model.train()

                # Stop memory sampling and get training memory stats
                profiler.stop_memory_sampling()

                # Start memory sampling for inference
                profiler.start_memory_sampling()

                # Time inference
                start = time.time()
                metrics = model.evaluate()
                end = time.time()

                # Log inference time
                profiler.log_inference(end - start, len(model.X_test))

                # Stop memory sampling and get inference memory stats
                profiler.stop_memory_sampling()

                # Log storage (model and data sizes)
                model_path = f"models/standard__{model_name}__{dataset_name}.joblib"
                data_path = f"data/processed/{dataset_name}.csv"
                profiler.log_storage(model_path=model_path, data_path=data_path)

                # Store results with profiling data
                results[dataset_name][model_name] = {
                    "metrics": metrics,
                    "profiling": profiler.export()
                }

                # Reset profiler for next iteration
                profiler.reset()

            except Exception as e:
                logger.error(f"Failed: {model_name} on {dataset_name}: {e}")
                results[dataset_name][model_name] = {
                    "error": str(e),
                    "profiling": profiler.export() if 'profiler' in locals() else {}
                }
                if 'profiler' in locals():
                    profiler.reset()

    logger.info("Standard pipeline complete.")
    return results