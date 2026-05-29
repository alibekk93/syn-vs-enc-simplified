"""Preprocessing pipeline — loads raw data, preprocesses, saves to data/processed/."""

import logging
from src.utils import load_config
from src.dataset import Dataset
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG = "config/datasets.yaml"
RESOURCE_CFG = "config/resource_profiling.yaml"


def run(datasets: list[str] | None = None) -> dict:
    """
    Args:
        datasets: List of dataset names to process. If None, all datasets are processed.

    Returns:
        Dict of {dataset: processing_results}
    """
    targets = datasets or list(load_config(DATASETS_CFG).keys())
    logger.info(f"Preprocessing pipeline started — datasets: {targets}")

    results = {}
    for name in targets:
        logger.info(f"--- Processing: {name} ---")

        profiler = ResourceProfiler(load_config(RESOURCE_CFG))

        try:
            dataset = Dataset(name, cfg=DATASETS_CFG)

            # Only one memory phase in preprocessing (no separate inference).
            profiler.start_memory_sampling(phase="processing")

            with profiler.time_block("processing"):
                dataset.run()

            profiler.stop_memory_sampling()

            raw_path       = f"data/raw/{name}.csv"
            processed_path = f"data/processed/{name}.csv"
            profiler.log_storage(model_path=None, data_path=processed_path)

            raw_size = profiler.file_size_mb(raw_path)
            if raw_size > 0:
                processed_size = profiler.file_size_mb(processed_path)
                if processed_size > 0:
                    profiler.log_storage_extra(
                        "compression_ratio",
                        round(processed_size / raw_size, 4),
                    )

            results[name] = {
                "status":     "success",
                "processing": profiler.export(),
            }

            # Persist profiling results to disk.
            profiler.save(f"preprocessing__{name}")

        except Exception as e:
            logger.error(f"Failed processing {name}: {e}")
            results[name] = {
                "status":     "error",
                "error":      str(e),
                "profiling":  profiler.export(),
            }
        finally:
            profiler.reset()

    logger.info("Preprocessing pipeline complete.")
    return results