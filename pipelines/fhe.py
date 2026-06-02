"""FHE training pipeline — trains and evaluates FHE models on processed data."""

import logging
from src.utils import load_config
from src.fhe_models import FHEModel
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG  = "config/datasets.yaml"
MODELS_CFG    = "config/models.yaml"
FHE_CFG       = "config/fhe.yaml"
RESOURCE_CFG  = "config/resource_profiling.yaml"


def run(
    datasets: list[str] | None = None,
    models:   list[str] | None = None,
    fhe_mode: str = "simulate",
    fhe_config_override=None,
) -> dict:
    """
    Args:
        datasets:  List of dataset names (default: all).
        models:    List of model names (default: all in config).
        fhe_mode:  FHE inference mode — 'disable', 'simulate', or 'execute'.

    Returns:
        Nested dict of {dataset: {model: metrics}}
    """
    targets_datasets = datasets or list(load_config(DATASETS_CFG).keys())
    targets_models   = models   or [m["name"] for m in load_config(MODELS_CFG).get("models", [])]
    fhe_config       = fhe_config_override or load_config(FHE_CFG)

    logger.info(f"FHE pipeline started — datasets: {targets_datasets}, models: {targets_models}, fhe_mode: {fhe_mode}")

    results = {}
    for dataset_name in targets_datasets:
        results[dataset_name] = {}
        for model_name in targets_models:
            logger.info(f"--- FHE {model_name} on {dataset_name} ---")

            profiler = ResourceProfiler(load_config(RESOURCE_CFG))

            try:
                model = FHEModel(
                    model_name,
                    cfg=MODELS_CFG,
                    mode="fhe",
                    fhe_cfg=fhe_config,
                )

                # --- Training phase ---
                profiler.start_memory_sampling(phase="training")

                with profiler.time_block("data_loading"):
                    model.load_data(dataset_name)
                    model.split()

                with profiler.time_block("training_fit"):
                    model.model.fit(model.X_train, model.y_train)

                with profiler.time_block("training_compile"):
                    model.model.compile(model.X_train)

                model._save_model()
                profiler.stop_memory_sampling()

                # --- Inference phase ---
                profiler.start_memory_sampling(phase="inference")

                import time as _time  # local import to keep the removed top-level import tidy
                start   = _time.time()
                metrics = model.evaluate(fhe=fhe_mode)
                end     = _time.time()

                profiler.log_inference(end - start, len(model.X_test))
                profiler.stop_memory_sampling()

                profiler.log_fhe(
                    complexity=getattr(model.model, "circuit_complexity", None)
                )

                model_path = f"models/fhe__{model_name}__{dataset_name}.json"
                data_path  = f"data/processed/{dataset_name}.csv"
                profiler.log_storage(model_path=model_path, data_path=data_path)

                results[dataset_name][model_name] = {
                    "metrics":   metrics,
                    "profiling": profiler.export(),
                }

                # Persist profiling results to disk.
                n_bits = fhe_config.get("models", {}).get(model_name, {}).get("n_bits")
                profiler.save(f"fhe__{model_name}__{dataset_name}__n{n_bits}")
                profiler.reset()

            except Exception as e:
                logger.error(f"Failed: FHE {model_name} on {dataset_name}: {e}")
                results[dataset_name][model_name] = {
                    "error":     str(e),
                    "profiling": profiler.export(),
                }
                profiler.reset()

    logger.info("FHE pipeline complete.")
    return results