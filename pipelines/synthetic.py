"""Synthetic pipeline — synthesizes data and trains models on it."""

import logging
import time
from src.utils import load_config, require_device
from src.synthesizers import Synthesizer
from src.models import Model
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG     = "config/datasets.yaml"


MODELS_CFG       = "config/models.yaml"
SYNTHESIZERS_CFG = "config/synthesizers.yaml"
RESOURCE_CFG     = "config/resource_profiling.yaml"


def _log_section(title: str) -> None:
    bar = "=" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def run(
    datasets:      list[str] | None = None,
    synthesizers:  list[str] | None = None,
    models:        list[str] | None = None,
    skip_training: bool = False,
    device: str | None = None,
    datasets_config: str = "config/datasets.yaml",
    resource_config: str = "config/resource_profiling.yaml",
    models_config: str = "config/models.yaml",
    synthesizers_config: str = "config/synthesizers.yaml",
) -> dict:
    """
    Args:
        datasets:      List of dataset names (default: all).
        synthesizers:  List of synthesizer names (default: all in config).
        models:        List of model names (default: all in config).
        skip_training: If True, only synthesize data without training models.
        device: "cpu" or "cuda" (default: synthesizers.yaml's `device`). Used
            for both synthesis (ctgan/nflow/arf only) and the downstream
            model training (xgboost only) — other synthesizers/models ignore
            it. Checked eagerly so a bad "cuda" request fails before any
            work happens.
        datasets_config: Path to datasets configuration file.
        resource_config: Path to resource profiling configuration file.
        models_config: Path to models configuration file.
        synthesizers_config: Path to synthesizers configuration file.

    Returns:
        Nested dict of {dataset: {synthesizer: {model: metrics}}}
    """
    targets_datasets     = datasets     or list(load_config(datasets_config).keys())
    targets_synthesizers = synthesizers or [k for k in load_config(synthesizers_config).get("methods", [])]
    targets_models       = models       or [m["name"] for m in load_config(models_config).get("models", [])]

    active_device = device or load_config(synthesizers_config).get("device", "cpu")
    require_device(active_device)

    logger.debug(
        f"Synthetic pipeline started — datasets: {targets_datasets}, "
        f"synthesizers: {targets_synthesizers}, models: {targets_models}, device: {active_device}"
    )

    results = {}
    for synth_name in targets_synthesizers:
        _log_section(f"SYNTHETIC: {synth_name}  |  device: {active_device}")

        for dataset_name in targets_datasets:
            results.setdefault(dataset_name, {})
            results[dataset_name][synth_name] = {}

            logger.info(f"--- Synthesizing: {synth_name} on {dataset_name} ---")

            # Create profiler before the try block.
            synth_profiler = ResourceProfiler(load_config(resource_config))

            try:
                synth = Synthesizer(synth_name, cfg=synthesizers_config, device=active_device)

                # Explicit phase label.
                synth_profiler.start_memory_sampling(phase="synthesis")

                with synth_profiler.time_block("synthesis_load"):
                    synth.load_data(dataset_name, dataset_cfg=datasets_config)
                with synth_profiler.time_block("synthesis_fit"):
                    synth.fit()
                with synth_profiler.time_block("synthesis_sample"):
                    synth.sample()
                    synth.save()

                synth_profiler.stop_memory_sampling()
                synth_profiler.save(f"{synth_name}__synthesis__{dataset_name}")

                if skip_training:
                    results[dataset_name][synth_name] = {
                        "synthesis": synth_profiler.export()
                    }
                    continue

                synthetic_dataset = f"{synth_name}__{dataset_name}"
                for model_name in targets_models:
                    logger.info(f"--- {model_name} on synthetic {synthetic_dataset} ---")

                    # Create per-model profiler before the inner try block.
                    train_profiler = ResourceProfiler(load_config(resource_config))

                    try:
                        model = Model(model_name, cfg=models_config, mode=synth_name, device=active_device)

                        # Explicit phase labels.
                        train_profiler.start_memory_sampling(phase="training")

                        with train_profiler.time_block("data_loading"):
                            model.load_data(synthetic_dataset, dataset_cfg=datasets_config)
                            model.save_dataset_name = dataset_name
                            model.use_all_as_train()
                            model.load_test_data(dataset_name, dataset_cfg=datasets_config)

                        with train_profiler.time_block("training"):
                            model.train()

                        train_profiler.stop_memory_sampling()

                        train_profiler.start_memory_sampling(phase="inference")

                        start   = time.time()
                        metrics = model.evaluate()
                        end     = time.time()

                        train_profiler.log_inference(end - start, len(model.X_test))
                        train_profiler.stop_memory_sampling()

                        model_path = f"models/{synth_name}__{model_name}__{dataset_name}.joblib"
                        data_path  = f"data/processed/{synthetic_dataset}.csv"
                        train_profiler.log_storage(model_path=model_path, data_path=data_path)

                        results[dataset_name][synth_name][model_name] = {
                            "metrics":   metrics,
                            "profiling": {
                                "synthesis": synth_profiler.export(),
                                "training":  train_profiler.export(),
                            },
                        }

                        # Persist profiling results to disk.
                        train_profiler.save(
                            f"{synth_name}__{model_name}__{dataset_name}"
                        )

                    except Exception as e:
                        logger.error(f"Training failed: {model_name} on {synthetic_dataset}: {e}")
                        results[dataset_name][synth_name][model_name] = {
                            "error":     str(e),
                            "profiling": train_profiler.export(),
                        }
                    finally:
                        train_profiler.reset()

            except Exception as e:
                logger.error(f"Synthesis failed: {synth_name} on {dataset_name}: {e}")
                results[dataset_name][synth_name] = {
                    "error":     str(e),
                    "profiling": synth_profiler.export(),
                }
            finally:
                synth_profiler.reset()

    logger.debug("Synthetic pipeline complete.")
    return results