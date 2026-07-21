"""Standard training pipeline — trains and evaluates models on processed data."""

import logging
from src.utils import load_config, require_device
from src.models import Model, SUPPORTED_METRICS
from src.resource_profiling import ResourceProfiler
from src import bootstrap_utils

logger = logging.getLogger(__name__)

DATASETS_CFG = "config/datasets.yaml"


MODELS_CFG   = "config/models.yaml"
RESOURCE_CFG = "config/resource_profiling.yaml"


def _log_section(title: str) -> None:
    bar = "=" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def run(
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    device: str | None = None,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 42,
    datasets_config: str = "config/datasets.yaml",
    resource_config: str = "config/resource_profiling.yaml",
    models_config: str = "config/models.yaml",
) -> dict:
    """
    Args:
        datasets: List of dataset names (default: all).
        models:   List of model names (default: all in config).
        device: "cpu" or "cuda" (default: models.yaml's `device`). Only
            xgboost has a GPU path — other model types ignore it. Checked
            eagerly so a bad "cuda" request fails before any training happens.
        datasets_config: Path to datasets configuration file.
        resource_config: Path to resource profiling configuration file.
        models_config: Path to models configuration file.

    Returns:
        Nested dict of {dataset: {model: metrics}}
    """
    targets_datasets = datasets or list(load_config(datasets_config).keys())
    targets_models   = models   or [m["name"] for m in load_config(models_config).get("models", [])]

    active_device = device or load_config(models_config).get("device", "cpu")
    require_device(active_device)

    logger.debug(f"Standard pipeline started — datasets: {targets_datasets}, models: {targets_models}, device: {active_device}")

    _log_section(f"STANDARD  |  device: {active_device}")

    results = {}
    for dataset_name in targets_datasets:
        results[dataset_name] = {}
        for model_name in targets_models:
            logger.info(f"--- {model_name} on {dataset_name} ---")

            profiler = ResourceProfiler(load_config(resource_config))

            try:
                model = Model(model_name, cfg=models_config, mode="standard", device=active_device)

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

                with profiler.inference_block(len(model.X_test)):
                    y_pred, y_proba = bootstrap_utils.predict_once(model, model.X_test)

                profiler.stop_memory_sampling()

                metric_names = load_config(models_config).get("metrics") or list(SUPPORTED_METRICS)
                metrics = bootstrap_utils.compute_metrics(model.y_test, y_pred, y_proba, metric_names)

                if n_bootstrap > 0:
                    iter_results  = bootstrap_utils.run_bootstrap(
                        y_true=model.y_test, y_pred=y_pred, y_proba=y_proba,
                        n=n_bootstrap, seed=bootstrap_seed, metric_names=metric_names,
                    )
                    metrics_to_save = bootstrap_utils.to_metric_lists(iter_results)
                else:
                    metrics_to_save = metrics

                bootstrap_utils.save_metrics_json(
                    path=model.results_dir / f"standard__{model_name}__{dataset_name}__test__metrics.json",
                    mode="standard", model_name=model_name, dataset_name=dataset_name,
                    split="test", metrics=metrics_to_save, n_bootstrap=n_bootstrap,
                )

                bootstrap_utils.save_predictions_json(
                    path=model.predictions_dir / f"standard__{model_name}__{dataset_name}__test__predictions.json",
                    mode="standard", model_name=model_name, dataset_name=dataset_name,
                    split="test", y_true=model.y_test, y_proba=y_proba, y_pred=y_pred,
                )

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

    logger.debug("Standard pipeline complete.")
    return results