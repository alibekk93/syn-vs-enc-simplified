"""Synthetic pipeline — synthesizes data and trains models on it."""

import logging
import time
from src.utils import load_config
from src.synthesizers import Synthesizer
from src.models import Model
from src.resource_profiling import ResourceProfiler

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
                # Create synthesizer
                synth = Synthesizer(synth_name, cfg=SYNTHESIZERS_CFG)

                # Create profiler for synthesis phase
                synth_profiler = ResourceProfiler(load_config("config/resource_profiling.yaml"))
                synth_profiler.start_memory_sampling()

                # Time synthesis process
                with synth_profiler.time_block("synthesis"):
                    synth.load_data(dataset_name, dataset_cfg=DATASETS_CFG)
                    synth.fit()
                    synth.sample()
                    synth.save()

                # Get synthesis memory stats
                synth_profiler.stop_memory_sampling()

                if skip_training:
                    results[dataset_name][synth_name] = {
                        "synthesis": synth_profiler.export()
                    }
                    continue

                synthetic_dataset = f"{synth_name}__{dataset_name}"
                for model_name in targets_models:
                    logger.info(f"--- {model_name} on synthetic {synthetic_dataset} ---")
                    try:
                        # Create model
                        model = Model(model_name, cfg=MODELS_CFG, mode=synth_name)

                        # Create profiler for training phase
                        train_profiler = ResourceProfiler(load_config("config/resource_profiling.yaml"))
                        train_profiler.start_memory_sampling()

                        # Time data loading (synthetic data)
                        with train_profiler.time_block("data_loading"):
                            model.load_data(synthetic_dataset, dataset_cfg=DATASETS_CFG)
                            # Save using the original dataset name, not the synthetic dataset name
                            model.save_dataset_name = dataset_name
                            model.split()

                        # Time training
                        with train_profiler.time_block("training"):
                            model.train()

                        # Get training memory stats
                        train_profiler.stop_memory_sampling()

                        # Start memory profiling for inference
                        train_profiler.start_memory_sampling()

                        # Time inference
                        start = time.time()
                        metrics = model.evaluate()
                        end = time.time()

                        # Log inference time
                        train_profiler.log_inference(end - start, len(model.X_test))

                        # Get inference memory stats
                        train_profiler.stop_memory_sampling()

                        # Log storage
                        model_path = f"models/{synth_name}__{model_name}__{dataset_name}.joblib"
                        data_path = f"data/processed/{synthetic_dataset}.csv"
                        train_profiler.log_storage(model_path=model_path, data_path=data_path)

                        # Store results with profiling data
                        results[dataset_name][synth_name][model_name] = {
                            "metrics": metrics,
                            "profiling": {
                                "synthesis": synth_profiler.export(),
                                "training": train_profiler.export()
                            }
                        }

                    except Exception as e:
                        logger.error(f"Training failed: {model_name} on {synthetic_dataset}: {e}")
                        results[dataset_name][synth_name][model_name] = {
                            "error": str(e),
                            "profiling": train_profiler.export() if 'train_profiler' in locals() else {}
                        }
                    finally:
                        if 'train_profiler' in locals():
                            train_profiler.reset()

            except Exception as e:
                logger.error(f"Synthesis failed: {synth_name} on {dataset_name}: {e}")
                results[dataset_name][synth_name] = {
                    "error": str(e),
                    "profiling": synth_profiler.export() if 'synth_profiler' in locals() else {}
                }
            finally:
                if 'synth_profiler' in locals():
                    synth_profiler.reset()

    logger.info("Synthetic pipeline complete.")
    return results