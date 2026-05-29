"""FHE training pipeline — trains and evaluates FHE models on processed data."""

import logging
from src.utils import load_config
from src.fhe_models import FHEModel

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
                model.load_data(dataset_name)
                model.split()
                model.train()
                metrics = model.evaluate(fhe=fhe_mode)
                results[dataset_name][model_name] = metrics
            except Exception as e:
                logger.error(f"Failed: FHE {model_name} on {dataset_name}: {e}")
                results[dataset_name][model_name] = {"error": str(e)}

    logger.info("FHE pipeline complete.")
    return results
