"""
Main entry point with subcommands.

Usage:
    python main.py run-experiment --config config/main.yaml
    python main.py run-experiment --config config/main.yaml --n-bits 4
    python main.py run-single-internal-validation-bootstrap --config config/main.yaml --seed 42
    python main.py list-n-bits
    python main.py create-visuals
    python main.py aggregate-metrics-csv
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

from src.utils import load_config, aggregate_internal_validation_bootstrap, aggregate_metrics_csv

BOOTSTRAP_CFG = "config/bootstrap.yaml"


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

    bootstrap_enabled = cfg.get("bootstrap", False)
    bootstrap_cfg     = load_config(BOOTSTRAP_CFG) if bootstrap_enabled else {}
    n_bootstrap       = bootstrap_cfg.get("n", 0)
    bootstrap_seed    = bootstrap_cfg.get("seed", 42)

    logger.info(f"=== Starting full pipeline (config: {config_path}) ===")

    if pipelines_cfg.get("preprocessing"):
        logger.info("=== Preprocessing ===")
        from pipelines import preprocessing
        preprocessing.run(datasets=datasets)

    if pipelines_cfg.get("raw"):
        logger.info("=== Raw ===")
        from pipelines import standard
        standard.run(datasets=datasets, models=models, device=device,
                     n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)

    if pipelines_cfg.get("synthetic"):
        logger.info("=== Synthetic ===")
        from pipelines import synthetic
        synthetic.run(datasets=datasets, synthesizers=synthesizers, models=models, device=device,
                      n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)

    if pipelines_cfg.get("fhe"):
        logger.info("=== FHE ===")
        from pipelines import fhe
        fhe.run(
            datasets=datasets,
            models=models,
            fhe_mode=fhe_mode,
            n_bits=n_bits,
            device=device,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )

    logger.info("=== Experiment complete ===")


def run_single_internal_validation_bootstrap(config_path: str, seed: int, n_bits: int | None = None, device: str | None = None):
    check_torch()
    set_seed(seed)

    cfg           = load_config(config_path)
    datasets      = cfg.get("datasets")
    models        = cfg.get("models")
    synthesizers  = cfg.get("synthesizers")
    fhe_mode      = cfg.get("fhe_mode", "simulate")
    pipelines_cfg = cfg.get("pipelines", {})

    logger.info(
        f"=== Running single internal validation bootstrap "
        f"(config: {config_path}, seed: {seed}) ==="
    )

    from pipelines import internal_validation_bootstrap
    internal_validation_bootstrap.run(
        datasets=datasets,
        models=models,
        synthesizers=synthesizers,
        seed=seed,
        fhe_mode=fhe_mode,
        n_bits=n_bits,
        device=device,
        pipelines_cfg=pipelines_cfg,
    )

    logger.info("=== Single internal validation bootstrap run complete ===")


def verify_gpu(venv: str, device: str = "cuda"):
    """Verifies that every GPU-capable synthesizer/model config/{synthesizers,
    models,fhe}.yaml configures for this venv actually computes on the GPU,
    not just that device='cuda' was accepted — see src/gpu_verification.py
    for why a config-only check isn't enough."""
    from src.gpu_verification import run as run_gpu_verification

    passed = run_gpu_verification(venv, device=device)
    if passed:
        logger.info("=== GPU verification PASSED ===")
    else:
        logger.error("=== GPU verification FAILED ===")
        raise SystemExit(1)


def create_visuals():
    logger.info("=== Creating visualizations ===")
    from src.visualization import generate_all_figures
    generate_all_figures()
    logger.info("=== Visualization complete ===")


def aggregate_internal_validation_bootstrap_results(results_dir: str, output_path: str):
    logger.info("=== Aggregating internal validation bootstrap results ===")
    aggregate_internal_validation_bootstrap(results_dir=results_dir, output_path=output_path)
    logger.info("=== Aggregation complete ===")


def aggregate_metrics_to_csv(metrics_dir: str, output_path: str):
    logger.info("=== Aggregating metrics to CSV ===")
    aggregate_metrics_csv(metrics_dir=metrics_dir, output_path=output_path)
    logger.info("=== Aggregation complete ===")


def generate_seeds(seed: int, length: int):
    """Generate a list of random seeds and save to file."""
    random.seed(seed)
    # Generate list of integers in a large range, e.g., 0 to 2**32 - 1
    seeds = [random.randint(0, 2**32 - 1) for _ in range(length)]
    # Write to file, one per line
    with open("internal_validation_bootstrap_seeds.txt", "w") as f:
        for s in seeds:
            f.write(f"{s}\n")
    logger.info(f"Generated {length} seeds and saved to internal_validation_bootstrap_seeds.txt")


def list_n_bits(config_path: str, out_path: str):
    """Expand the configured n_bits sweep and save to a flat file (one value per line)."""
    from src.utils import expand_n_bits

    cfg = load_config(config_path)
    values = [v for v in expand_n_bits(cfg) if v is not None]

    with open(out_path, "w") as f:
        for v in values:
            f.write(f"{v}\n")

    logger.info(f"Wrote {len(values)} n_bits values to {out_path}")


def list_synth_scales(config_path: str, out_path: str):
    """Expand the configured synth_scale values and save to a flat file (one value per line)."""
    from src.utils import expand_synth_scales

    cfg = load_config(config_path)
    values = expand_synth_scales(cfg)

    with open(out_path, "w") as f:
        for v in values:
            f.write(f"{v}\n")

    logger.info(f"Wrote {len(values)} synth_scale values to {out_path}")

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

    # ---- run-single-internal-validation-bootstrap ----
    bootstrap_parser = subparsers.add_parser(
        "run-single-internal-validation-bootstrap",
        help="Run a single internal validation bootstrap evaluation on existing models"
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
             "Run once per (seed, n_bits) pair, in parallel, to sweep n_bits within internal validation bootstrap."
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

    # ---- verify-gpu ----
    verify_gpu_parser = subparsers.add_parser(
        "verify-gpu",
        help="Verify every GPU-capable synthesizer/model configured for a venv actually computes on the GPU "
             "(not just that device='cuda' was accepted)"
    )
    verify_gpu_parser.add_argument(
        "--venv",
        choices=["sdv", "synthcity", "fhe"],
        required=True,
        help="Which venv this is running in (.venv-sdv / .venv-synthcity / .venv-fhe) — determines which "
             "synthesizers (from config/synthesizers.yaml) and models (config/models.yaml / config/fhe.yaml) "
             "are checked. All configured GPU-capable methods/models for that venv are checked automatically."
    )
    verify_gpu_parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to verify (default: cuda). Use --device cpu as a negative control to confirm "
             "the check correctly reports GPU NOT used."
    )

    # ---- aggregate-internal-validation-bootstrap ----
    aggregate_parser = subparsers.add_parser(
        "aggregate-internal-validation-bootstrap",
        help="Aggregate per-seed internal validation bootstrap results into a single JSON file"
    )
    aggregate_parser.add_argument(
        "--results-dir",
        default="results/internal_validation_bootstrap",
        help="Directory containing per-seed internal validation bootstrap result subdirectories (default: results/internal_validation_bootstrap)"
    )
    aggregate_parser.add_argument(
        "--output",
        default="results/internal_validation_bootstrap/aggregated.json",
        help="Path to write the aggregated JSON file (default: results/internal_validation_bootstrap/aggregated.json)"
    )

    # ---- aggregate-metrics-csv ----
    metrics_csv_parser = subparsers.add_parser(
        "aggregate-metrics-csv",
        help="Aggregate results/metrics/*.json into one CSV (mean + 95% CI per metric, "
             "one row per mode/dataset/model)"
    )
    metrics_csv_parser.add_argument(
        "--metrics-dir",
        default="results/metrics",
        help="Directory containing per-run metrics JSON files (default: results/metrics)"
    )
    metrics_csv_parser.add_argument(
        "--output",
        default="results/metrics_aggregated.csv",
        help="Path to write the aggregated CSV file (default: results/metrics_aggregated.csv)"
    )

    # ---- generate-seeds ----
    seeds_parser = subparsers.add_parser(
        "generate-seeds",
        help="Generate a list of random seeds for internal validation bootstrap sampling"
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

    # ---- list-synth-scales ----
    scales_parser = subparsers.add_parser(
        "list-synth-scales",
        help="Expand the configured synth_scale values to a flat file (one value per line)"
    )
    scales_parser.add_argument(
        "--config",
        default="config/synthesizers.yaml",
        help="Path to synthesizers config (default: config/synthesizers.yaml)"
    )
    scales_parser.add_argument(
        "--out",
        default="synth_scales.txt",
        help="Path to write the synth_scale values list (default: synth_scales.txt)"
    )

    args = parser.parse_args()

    if args.command == "run-experiment":
        run_experiment(args.config, args.n_bits, args.device)

    elif args.command == "run-single-internal-validation-bootstrap":
        run_single_internal_validation_bootstrap(args.config, args.seed, args.n_bits, args.device)

    elif args.command == "create-visuals":
        create_visuals()

    elif args.command == "verify-gpu":
        verify_gpu(args.venv, args.device)

    elif args.command == "aggregate-internal-validation-bootstrap":
        aggregate_internal_validation_bootstrap_results(args.results_dir, args.output)

    elif args.command == "aggregate-metrics-csv":
        aggregate_metrics_to_csv(args.metrics_dir, args.output)

    elif args.command == "generate-seeds":
        generate_seeds(args.seed, args.length)

    elif args.command == "list-n-bits":
        list_n_bits(args.config, args.out)

    elif args.command == "list-synth-scales":
        list_synth_scales(args.config, args.out)