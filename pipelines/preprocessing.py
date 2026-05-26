"""Preprocessing pipeline — loads raw data, preprocesses, saves to data/processed/."""

import logging
from src.utils import load_config
from src.dataset import Dataset

logger = logging.getLogger(__name__)

DATASETS_CFG = "config/datasets.yaml"


def run(datasets: list[str] | None = None) -> None:
    """
    Args:
        datasets: List of dataset names to process. If None, all datasets are processed.
    """
    targets = datasets or list(load_config(DATASETS_CFG).keys())
    logger.info(f"Preprocessing pipeline started — datasets: {targets}")

    for name in targets:
        logger.info(f"--- Processing: {name} ---")
        Dataset(name, cfg=DATASETS_CFG).run()

    logger.info("Preprocessing pipeline complete.")
