"""Preprocessing pipeline — loads raw data, preprocesses, saves to data/processed/."""

import logging
import time
from src.utils import load_config
from src.dataset import Dataset
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG = "config/datasets.yaml"


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
        try:
            # Create processor
            dataset = Dataset(name, cfg=DATASETS_CFG)

            # Create profiler
            profiler = ResourceProfiler(load_config("config/resource_profiling.yaml"))
            profiler.start_memory_sampling()

            # Time processing
            with profiler.time_block("processing"):
                dataset.run()

            # Get memory stats
            profiler.stop_memory_sampling()

            # Calculate storage info
            raw_path = f"data/raw/{name}.csv"
            processed_path = f"data/processed/{name}.csv"
            profiler.log_storage(
                model_path=None,  # No model in preprocessing
                data_path=processed_path
            )
            # Also store raw data size for comparison
            raw_size = profiler.file_size_mb(raw_path)
            if raw_size > 0 and "data_size_mb" in profiler.results["storage"]:
                processed_size = profiler.results["storage"]["data_size_mb"]
                profiler.results["storage"]["compression_ratio"] = round(processed_size / raw_size, 4) if raw_size > 0 else None

            results[name] = {
                "status": "success",
                "processing": profiler.export()
            }

        except Exception as e:
            logger.error(f"Failed processing {name}: {e}")
            results[name] = {
                "status": "error",
                "error": str(e),
                "processing": profiler.export() if 'profiler' in locals() else {}
            }
        finally:
            if 'profiler' in locals():
                profiler.reset()

    logger.info("Preprocessing pipeline complete.")
    return results