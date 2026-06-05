"""
Main entry point with subcommands.

Usage:
    python main.py run-experiment --config config/main.yaml
    python main.py run-single-bootstrap --config config/main.yaml --seed 42
    python main.py create-visuals
"""

import argparse
import logging
import random
import numpy as np

from src.utils import load_config
from pipelines import preprocessing, standard, synthetic, fhe, bootstrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def check_torch():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        logging.info(f"[Torch] CUDA available: {cuda_available}")
        if cuda_available:
            logging.info(f"[Torch] Using GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logging.info("[Torch] PyTorch not installed — skipping CUDA check")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.info(f"[Seed] Using seed: {seed}")


# --------------------------------------------------
# Commands
# --------------------------------------------------

def run_experiment(config_path: str):
    check_torch()

    cfg          = load_config(config_path)
    datasets     = cfg.get("datasets")
    models       = cfg.get("models")
    synthesizers = cfg.get("synthesizers")
    fhe_mode     = cfg.get("fhe_mode", "simulate")
    pipelines_cfg = cfg.get("pipelines", {})

    logger.info(f"=== Starting full pipeline (config: {config_path}) ===")

    if pipelines_cfg.get("preprocessing"):
        logger.info("=== Preprocessing ===")
        preprocessing.run(datasets=datasets)

    if pipelines_cfg.get("raw"):
        logger.info("=== Raw ===")
        standard.run(datasets=datasets, models=models)

    if pipelines_cfg.get("synthetic"):
        logger.info("=== Synthetic ===")
        synthetic.run(datasets=datasets, synthesizers=synthesizers, models=models)

    if pipelines_cfg.get("fhe"):
        logger.info("=== FHE ===")
        fhe.run(
            datasets=datasets,
            models=models,
            fhe_mode=fhe_mode
        )

    logger.info("=== Experiment complete ===")


def run_single_bootstrap(config_path: str, seed: int):
    check_torch()
    set_seed(seed)

    cfg          = load_config(config_path)
    datasets     = cfg.get("datasets")
    models       = cfg.get("models")
    evaluation   = cfg.get("evaluation", {})  # optional if you store eval config

    logger.info(
        f"=== Running single bootstrap "
        f"(config: {config_path}, seed: {seed}) ==="
    )

    bootstrap.run(
        datasets=datasets,
        models=models,
        seed=seed,
        evaluation=evaluation,
    )

    logger.info("=== Single bootstrap run complete ===")


def create_visuals():
    logger.info("=== Creating visualizations ===")
    from src.visualization import generate_all_figures
    generate_all_figures()
    logger.info("=== Visualization complete ===")


def generate_seeds(seed: int, length: int):
    """Generate a list of random seeds and save to file."""
    random.seed(seed)
    # Generate list of integers in a large range, e.g., 0 to 2**32 - 1
    seeds = [random.randint(0, 2**32 - 1) for _ in range(length)]
    # Write to file, one per line
    with open("bootstrap_seeds.txt", "w") as f:
        for s in seeds:
            f.write(f"{s}\n")
        f.write(f"{s}\n")
    logger.info(f"Generated {length} seeds and saved to bootstrap_seeds.txt")

# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline runner with commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- run-experiment ----
    run_parser = subparsers.add_parser(
        "run-experiment",
        help="Run full pipeline experiment"
    )
    run_parser.add_argument(
        "--config",
        default="config/main.yaml",
        help="Path to master config"
    )

    # ---- run-single-bootstrap ----
    bootstrap_parser = subparsers.add_parser(
        "run-single-bootstrap",
        help="Run a single bootstrap evaluation on existing models"
    )
    bootstrap_parser.add_argument(
        "--config",
        default="config/main.yaml",
        help="Path to master config"
    )
    bootstrap_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # ---- create-visuals ----
    subparsers.add_parser(
        "create-visuals",
        help="Generate visualizations only"
    )

    # ---- generate-seeds ----
    seeds_parser = subparsers.add_parser(
        "generate-seeds",
        help="Generate a list of random seeds for bootstrap sampling"
    )
    seeds_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to initialize the generator (default: 42)"
    )
    seeds_parser.add_argument(
        "--length",
        type=int,
        default=1000,
        help="Number of seeds to generate (default: 1000)"
    )

    args = parser.parse_args()

    if args.command == "run-experiment":
        run_experiment(args.config)

    elif args.command == "run-single-bootstrap":
        run_single_bootstrap(args.config, args.seed)

    elif args.command == "create-visuals":
        create_visuals()

    elif args.command == "generate-seeds":
        generate_seeds(args.seed, args.length)