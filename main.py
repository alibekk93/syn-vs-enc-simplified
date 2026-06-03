"""
Main entry point — runs pipelines as configured in config/main.yaml.

Usage:
    python main.py
    python main.py --config config/main.yaml
"""

import argparse
import logging

from src.utils import load_config
from pipelines import preprocessing, standard, synthetic, fhe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import torch
    cuda_available = torch.cuda.is_available()
    logging.info(f"[Torch] CUDA available: {cuda_available}")
    if cuda_available:
        logging.info(f"[Torch] Using GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    logging.info("[Torch] PyTorch not installed — skipping CUDA check")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/main.yaml", help="Path to master config")
    args = parser.parse_args()

    cfg          = load_config(args.config)
    datasets     = cfg.get("datasets")
    models       = cfg.get("models")
    synthesizers = cfg.get("synthesizers")
    fhe_mode     = cfg.get("fhe_mode", "simulate")
    pipelines    = cfg.get("pipelines", {})

    logger.info(f"=== Starting full pipeline (config: {args.config}) ===")

    if pipelines.get("preprocessing"):
        logger.info("=== Preprocessing ===")
        preprocessing.run(datasets=datasets)

    if pipelines.get("raw"):
        logger.info("=== Raw ===")
        standard.run(datasets=datasets, models=models)

    if pipelines.get("synthetic"):
        logger.info("=== Synthetic ===")
        synthetic.run(datasets=datasets, synthesizers=synthesizers, models=models)

    if pipelines.get("ablation"):
        logger.info("=== FHE (Ablation) ===")
        ablation.run(
            datasets=datasets,
            models=models,
            fhe_mode=fhe_mode
        )

    if pipelines.get("fhe"):
        fhe.run(
            datasets=datasets,
            models=models,
            fhe_mode=fhe_mode
        )

    logger.info("=== Full pipeline complete ===")
