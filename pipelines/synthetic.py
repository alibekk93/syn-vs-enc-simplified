"""Synthetic pipeline — synthesizes data and trains models on it."""

import logging
from src.utils import load_config
from src.synthesizers import Synthesizer
from src.models import Model

logger = logging.getLogger(__name__)

DATASETS_CFG     = "config/datasets.yaml"
MODELS_CFG       = "config/models.yaml"
SYNTHESIZERS_CFG = "config/synthesizers.yaml"


def run(
    datasets:      list[str] | None = None,
    synthesizers:  list[str] | None = None,
    models:        list[str] | None = None,
    skip_training: bool = False,
) -> dict:
    """
    Args:
        datasets:      List of dataset names (default: all).
        synthesizers:  List of synthesizer names (default: all in config).
        models:        List of model names (default: all in config).
        skip_training: If True, only synthesize data without training models.

    Returns:
        Nested dict of {dataset: {synthesizer: {model: metrics}}}
    """
    targets_datasets     = datasets     or list(load_config(DATASETS_CFG).keys())
    targets_synthesizers = synthesizers or [k for k in load_config(SYNTHESIZERS_CFG) if k != "output"]
    targets_models       = models       or [m["name"] for m in load_config(MODELS_CFG).get("models", [])]

    logger.info(f"Synthetic pipeline started — datasets: {targets_datasets}, synthesizers: {targets_synthesizers}, models: {targets_models}")

    results = {}
    for dataset_name in targets_datasets:
        results[dataset_name] = {}
        for synth_name in targets_synthesizers:
            results[dataset_name][synth_name] = {}

            logger.info(f"--- Synthesizing: {synth_name} on {dataset_name} ---")
            try:
                synth = Synthesizer(synth_name, cfg=SYNTHESIZERS_CFG)
                synth.fit(dataset_name)
                synth.sample()
                synth.save()
            except Exception as e:
                logger.error(f"Synthesis failed: {synth_name} on {dataset_name}: {e}")
                results[dataset_name][synth_name] = {"error": str(e)}
                continue

            if skip_training:
                continue

            synthetic_dataset = f"{synth_name}__{dataset_name}"
            for model_name in targets_models:
                logger.info(f"--- {model_name} on synthetic {synthetic_dataset} ---")
                try:
                    model = Model(model_name, cfg=MODELS_CFG)
                    model.load_data(synthetic_dataset, dataset_cfg=DATASETS_CFG)
                    model.split()
                    model.train()
                    metrics = model.evaluate()
                    results[dataset_name][synth_name][model_name] = metrics
                except Exception as e:
                    logger.error(f"Training failed: {model_name} on {synthetic_dataset}: {e}")
                    results[dataset_name][synth_name][model_name] = {"error": str(e)}

    logger.info("Synthetic pipeline complete.")
    return results
