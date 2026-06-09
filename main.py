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
    fhe_mode     = cfg.get("fhe_mode", "simulate")

    logger.info(
        f"=== Running single bootstrap "
        f"(config: {config_path}, seed: {seed}) ==="
    )

    bootstrap.run(
        datasets=datasets,
        models=models,
        seed=seed,
        fhe_mode=fhe_mode,
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
    logger.info(f"Generated {length} seeds and saved to bootstrap_seeds.txt")


def aggregate_bootstrap(results_dir: str = "results/bootstrap", output_path: str = "results/bootstrap_aggregated.json"):
    """Concatenate all bootstrap results into a single hierarchical JSON file.

    Output structure:
        {metrics|resource_profiles} -> mode -> model -> dataset -> [per-seed records]

    Resource profile filename conventions handled:
        preprocessing__{dataset}                           -> mode=preprocessing, model=_
        {synthesizer}__{dataset}__synthesis                -> mode=<synthesizer>, model=_synthesis
        {mode}__{model}__{dataset}                         -> standard / fhe_N model files
        synthetic__{synthesizer}__{model}__{dataset}       -> mode=<synthesizer> (aligns with metrics)
    """
    import json
    from pathlib import Path

    results_path = Path(results_dir)
    output: dict = {"metrics": {}, "resource_profiles": {}}

    def get_leaf(root: dict, mode: str, model: str, dataset: str) -> list:
        return (
            root
            .setdefault(mode, {})
            .setdefault(model, {})
            .setdefault(dataset, [])
        )

    seed_dirs = sorted(
        (d for d in results_path.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not seed_dirs:
        logger.warning(f"No seed directories found in {results_dir}")
        return

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name)

        metrics_dir = seed_dir / "metrics"
        if metrics_dir.exists():
            for f in sorted(metrics_dir.glob("*.json")):
                parts = f.stem.split("__")
                if len(parts) == 5 and parts[3] == "test" and parts[4] == "metrics":
                    mode, model, dataset = parts[0], parts[1], parts[2]
                    data = json.loads(f.read_text(encoding="utf-8"))
                    get_leaf(output["metrics"], mode, model, dataset).append(
                        {"seed": seed, **data}
                    )
                else:
                    logger.warning(f"Unrecognized metrics filename: {f.name}")

        resource_dir = seed_dir / "resource_profiles"
        if resource_dir.exists():
            for f in sorted(resource_dir.glob("*.json")):
                parts = f.stem.split("__")
                data = json.loads(f.read_text(encoding="utf-8"))

                if len(parts) == 2:
                    # preprocessing__{dataset}
                    mode, model, dataset = parts[0], "_", parts[1]
                elif len(parts) == 3 and parts[2] == "synthesis":
                    # {synthesizer}__{dataset}__synthesis
                    mode, model, dataset = parts[0], "_synthesis", parts[1]
                elif len(parts) == 3:
                    # {mode}__{model}__{dataset}  (standard / fhe_N)
                    mode, model, dataset = parts[0], parts[1], parts[2]
                elif len(parts) == 4 and parts[0] == "synthetic":
                    # synthetic__{synthesizer}__{model}__{dataset}
                    mode, model, dataset = parts[1], parts[2], parts[3]
                else:
                    logger.warning(f"Unrecognized resource profile filename: {f.name}")
                    continue

                get_leaf(output["resource_profiles"], mode, model, dataset).append(
                    {"seed": seed, **data}
                )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Aggregated bootstrap results saved to {output_path}")

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

    # ---- aggregate-bootstrap ----
    agg_parser = subparsers.add_parser(
        "aggregate-bootstrap",
        help="Concatenate all bootstrap results into a single hierarchical JSON file"
    )
    agg_parser.add_argument(
        "--results-dir",
        default="results/bootstrap",
        help="Directory containing per-seed bootstrap results (default: results/bootstrap)"
    )
    agg_parser.add_argument(
        "--output",
        default="results/bootstrap/aggregated.json",
        help="Output file path (default: results/bootstrap/aggregated.json)"
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

    elif args.command == "aggregate-bootstrap":
        aggregate_bootstrap(args.results_dir, args.output)