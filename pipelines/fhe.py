"""FHE training pipeline — trains and evaluates FHE models on processed data."""

import logging
import time
from src.utils import load_config
from src.fhe_models import FHEModel
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG = "config/datasets.yaml"
MODELS_CFG   = "config/models.yaml"
FHE_CFG      = "config/fhe.yaml"


def run(
    datasets:  list[str] | None = None,
    models:    list[str] | None = None,
    fhe_mode:  str = "simulate",
) -> dict:
    """
    Args:
        datasets:  List of dataset names (default: all).
        models:    List of model names (default: all in config).
        fhe_mode:  FHE inference mode — 'disable', 'simulate', or 'execute' (real FHE).

    Returns:
        Nested dict of {dataset: {model: metrics}}
    """
    targets_datasets = datasets or list(load_config(DATASETS_CFG).keys())
    targets_models   = models   or [m["name"] for m in load_config(MODELS_CFG).get("models", [])]
    fhe_config = load_config(FHE_CFG)

    logger.info(f"FHE pipeline started — datasets: {targets_datasets}, models: {targets_models}, fhe_mode: {fhe_mode}")

    results = {}
    for dataset_name in targets_datasets:
        results[dataset_name] = {}
        for model_name in targets_models:
            logger.info(f"--- FHE {model_name} on {dataset_name} ---")
            try:
                model = FHEModel(
                    model_name,
                    cfg=MODELS_CFG,
                    mode="fhe",
                    fhe_cfg=fhe_config
                )

                # Start memory profiling
                profiler = ResourceProfiler(load_config("config/resource_profiling.yaml"))
                profiler.start_memory_sampling()

                # Time data loading and splitting
                with profiler.time_block("data_loading"):
                    model.load_data(dataset_name)
                    model.split()

                # Time training (fitting)
                with profiler.time_block("training_fit"):
                    model.model.fit(model.X_train, model.y_train)

                # Time compilation
                compile_start = time.time()
                model.model.compile(model.X_train)
                compile_end = time.time()
                compile_time = compile_end - compile_start

                # Save model after compilation
                model._save_model()

                # Get training memory stats (for fit + compile)
                profiler.stop_memory_sampling()

                # Start memory profiling for inference
                profiler.start_memory_sampling()

                # Time inference
                start = time.time()
                metrics = model.evaluate(fhe=fhe_mode)
                end = time.time()

                # Log inference time
                profiler.log_inference(end - start, len(model.X_test))

                # Get inference memory stats
                profiler.stop_memory_sampling()

                # Log FHE info
                profiler.log_fhe(compile_time=compile_time, complexity=getattr(model.model, 'circuit_complexity', None))

                # Log storage
                model_path = f"models/fhe__{model_name}__{dataset_name}.json"
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
                logger.error(f"Failed: FHE {model_name} on {dataset_name}: {e}")
                results[dataset_name][model_name] = {
                    "error": str(e),
                    "profiling": profiler.export() if 'profiler' in locals() else {}
                }
                if 'profiler' in locals():
                    profiler.reset()

    logger.info("FHE pipeline complete.")
    return results