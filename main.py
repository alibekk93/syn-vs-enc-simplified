"""
Main entry point with subcommands.

Usage:
    python main.py run-experiment --config config/main.yaml
    python main.py run-experiment --config config/main.yaml --n-bits 4
    python main.py run-single-bootstrap --config config/main.yaml --seed 42
    python main.py list-n-bits
    python main.py create-visuals
"""

import argparse
import logging
import random
import numpy as np

# Configure logging before importing pipelines — synthcity's dependency tree
# (transformers, optuna, pgmpy, ...) attaches its own root logging handlers
# on import, which makes a later basicConfig() call a silent no-op.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

from src.utils import load_config, aggregate_bootstrap


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

def run_experiment(config_path: str, n_bits: int | None = None, device: str | None = None):
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
        from pipelines import preprocessing
        preprocessing.run(datasets=datasets)

    if pipelines_cfg.get("raw"):
        logger.info("=== Raw ===")
        from pipelines import standard
        standard.run(datasets=datasets, models=models, device=device)

    if pipelines_cfg.get("synthetic"):
        logger.info("=== Synthetic ===")
        from pipelines import synthetic
        synthetic.run(datasets=datasets, synthesizers=synthesizers, models=models, device=device)

    if pipelines_cfg.get("fhe"):
        logger.info("=== FHE ===")
        from pipelines import fhe
        fhe.run(
            datasets=datasets,
            models=models,
            fhe_mode=fhe_mode,
            n_bits=n_bits,
            device=device,
        )

    logger.info("=== Experiment complete ===")


def run_single_bootstrap(config_path: str, seed: int, n_bits: int | None = None, device: str | None = None):
    check_torch()
    set_seed(seed)

    cfg           = load_config(config_path)
    datasets      = cfg.get("datasets")
    models        = cfg.get("models")
    fhe_mode      = cfg.get("fhe_mode", "simulate")
    pipelines_cfg = cfg.get("pipelines", {})

    logger.info(
        f"=== Running single bootstrap "
        f"(config: {config_path}, seed: {seed}) ==="
    )

    from pipelines import bootstrap
    bootstrap.run(
        datasets=datasets,
        models=models,
        seed=seed,
        fhe_mode=fhe_mode,
        n_bits=n_bits,
        device=device,
        pipelines_cfg=pipelines_cfg,
    )

    logger.info("=== Single bootstrap run complete ===")


def create_visuals():
    logger.info("=== Creating visualizations ===")
    from src.visualization import generate_all_figures
    generate_all_figures()
    logger.info("=== Visualization complete ===")


def aggregate_bootstrap_results(results_dir: str, output_path: str):
    logger.info("=== Aggregating bootstrap results ===")
    aggregate_bootstrap(results_dir=results_dir, output_path=output_path)
    logger.info("=== Aggregation complete ===")


def generate_seeds(seed: int, length: int):
    """Generate a list of random seeds and save to file."""
    random.seed(seed)
    # Generate list of integers in a large range, e.g., 0 to 2**32 - 1
    seeds = [random.randint(0, 2**32 - 1) for _ in range(length)]
    # Write to file, one per line
    with open("bootstrap_seeds.txt", "w") as f:
        for s in seeds:
            f.write(f"{s}\n")
    logger.info(f"Generated {length} seeds and saved to bootstrap_seeds.txt")


def list_n_bits(config_path: str, out_path: str):
    """Expand the configured n_bits sweep and save to a flat file (one value per line)."""
    from src.utils import expand_n_bits

    cfg = load_config(config_path)
    values = [v for v in expand_n_bits(cfg) if v is not None]

    with open(out_path, "w") as f:
        for v in values:
            f.write(f"{v}\n")

    logger.info(f"Wrote {len(values)} n_bits values to {out_path}")

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
    run_parser.add_argument(
        "--n-bits",
        type=int,
        default=None,
        help="Override n_bits for every FHE model (default: per-model values in config/fhe.yaml). "
             "Run once per value, in parallel, to sweep n_bits — see list-n-bits."
    )
    run_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Override the compute device for FHE/standard/synthetic stages (default: each "
             "stage's own config `device`, normally cpu). Only GPU-capable models/synthesizers "
             "use it (xgboost; ctgan/nflow/arf; all FHE models) — others ignore it. FHE's cuda "
             "needs the GPU build of concrete-python; standard/synthetic's cuda needs a "
             "CUDA-enabled PyTorch install."
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
    bootstrap_parser.add_argument(
        "--n-bits",
        type=int,
        default=None,
        help="Override n_bits for every FHE model (default: per-model values in config/fhe.yaml). "
             "Run once per (seed, n_bits) pair, in parallel, to sweep n_bits within bootstrap."
    )
    bootstrap_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Override the compute device for FHE/standard/synthetic stages (default: each "
             "stage's own config `device`, normally cpu). Only GPU-capable models/synthesizers "
             "use it (xgboost; ctgan/nflow/arf; all FHE models) — others ignore it. FHE's cuda "
             "needs the GPU build of concrete-python; standard/synthetic's cuda needs a "
             "CUDA-enabled PyTorch install."
    )

    # ---- create-visuals ----
    subparsers.add_parser(
        "create-visuals",
        help="Generate visualizations only"
    )

    # ---- aggregate-bootstrap ----
    aggregate_parser = subparsers.add_parser(
        "aggregate-bootstrap",
        help="Aggregate per-seed bootstrap results into a single JSON file"
    )
    aggregate_parser.add_argument(
        "--results-dir",
        default="results/bootstrap",
        help="Directory containing per-seed bootstrap result subdirectories (default: results/bootstrap)"
    )
    aggregate_parser.add_argument(
        "--output",
        default="results/bootstrap/aggregated.json",
        help="Path to write the aggregated JSON file (default: results/bootstrap/aggregated.json)"
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

    # ---- list-n-bits ----
    n_bits_parser = subparsers.add_parser(
        "list-n-bits",
        help="Expand the configured n_bits sweep to a flat file (one value per line)"
    )
    n_bits_parser.add_argument(
        "--config",
        default="config/fhe.yaml",
        help="Path to FHE config (default: config/fhe.yaml)"
    )
    n_bits_parser.add_argument(
        "--out",
        default="fhe_n_bits.txt",
        help="Path to write the n_bits list (default: fhe_n_bits.txt)"
    )

    args = parser.parse_args()

    if args.command == "run-experiment":
        run_experiment(args.config, args.n_bits, args.device)

    elif args.command == "run-single-bootstrap":
        run_single_bootstrap(args.config, args.seed, args.n_bits, args.device)

    elif args.command == "create-visuals":
        create_visuals()

    elif args.command == "aggregate-bootstrap":
        aggregate_bootstrap_results(args.results_dir, args.output)

    elif args.command == "generate-seeds":
        generate_seeds(args.seed, args.length)

    elif args.command == "list-n-bits":
        list_n_bits(args.config, args.out)