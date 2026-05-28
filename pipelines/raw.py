"""Raw training pipeline — trains and evaluates models on processed data."""

import logging
from src.utils import load_config
from src.models import Model

logger = logging.getLogger(__name__)

DATASETS_CFG = "config/datasets.yaml"
MODELS_CFG   = "config/models.yaml"


def run(datasets: list[str] | None = None, models: list[str] | None = None) -> dict:
    """
    Args:
        datasets: List of dataset names (default: all).
        models:   List of model names (default: all in config).

    Returns:
        Nested dict of {dataset: {model: metrics}}
    """
    targets_datasets = datasets or list(load_config(DATASETS_CFG).keys())
    targets_models   = models   or [m["name"] for m in load_config(MODELS_CFG).get("models", [])]

    logger.info(f"Raw pipeline started — datasets: {targets_datasets}, models: {targets_models}")

    results = {}
    for dataset_name in targets_datasets:
        results[dataset_name] = {}
        for model_name in targets_models:
            logger.info(f"--- {model_name} on {dataset_name} ---")
            try:
                metrics = Model(model_name, cfg=MODELS_CFG, mode="standard").run(dataset_name)
                results[dataset_name][model_name] = metrics
            except Exception as e:
                logger.error(f"Failed: {model_name} on {dataset_name}: {e}")
                results[dataset_name][model_name] = {"error": str(e)}

    logger.info("Raw pipeline complete.")
    return results
