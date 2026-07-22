"""
Main entry point with subcommands.

Usage:
    python main.py run-experiment --config config/main.yaml
    python main.py run-experiment --config config/main.yaml --n-bits 4
    python main.py run-single-internal-validation-bootstrap --config config/main.yaml --seed 42
    python main.py list-n-bits
    python main.py create-all-visuals
    python main.py create-multipanel-visuals
    python main.py aggregate-metrics-csv
    python main.py paired-bootstrap-tests --metric roc_auc --modes standard 'fhe_*'
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

from src.utils import (
    load_config,
    aggregate_internal_validation_bootstrap,
    aggregate_metrics_csv,
    install_log_filters,
)

# Collapse pgmpy's float-epsilon "probabilities don't sum to 1" warnings, which
# otherwise repeat hundreds of times per run and bury real errors. Installed
# after basicConfig so the filter lands on the handler it just created.
install_log_filters()

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


def create_all_visuals():
    logger.info("=== Creating visualizations ===")
    from src.visualization import generate_all_figures
    generate_all_figures()
    logger.info("=== Visualization complete ===")


def create_multipanel_visuals():
    logger.info("=== Creating multipanel visualizations ===")
    from src.visualization import generate_multipanel_figures
    generate_multipanel_figures()
    logger.info("=== Multipanel visualization complete ===")


def aggregate_internal_validation_bootstrap_results(results_dir: str, output_path: str):
    logger.info("=== Aggregating internal validation bootstrap results ===")
    aggregate_internal_validation_bootstrap(results_dir=results_dir, output_path=output_path)
    logger.info("=== Aggregation complete ===")


def aggregate_metrics_to_csv(metrics_dir: str, output_path: str, profiles_dir: str):
    logger.info("=== Aggregating metrics to CSV ===")
    aggregate_metrics_csv(metrics_dir=metrics_dir, output_path=output_path, profiles_dir=profiles_dir)
    logger.info("=== Aggregation complete ===")


def run_paired_bootstrap_tests(metrics_dir: str, output, metric: str, modes,
                               datasets, models, alpha: float, fmt: str,
                               comparison: str, reference, tie_rule: str,
                               margin: float, test_type: str):
    logger.info(f"=== Paired bootstrap significance tests ({metric}) ===")
    # --reference only makes sense in the reference design, so supplying it is
    # taken as the intent, saving the user from passing both flags.
    if reference and comparison == "pairwise":
        comparison = "reference"
    from src.stats_tests import run as run_stats_tests
    run_stats_tests(
        metrics_dir=metrics_dir, output=output, metric=metric,
        modes=modes, datasets=datasets, models=models, alpha=alpha, fmt=fmt,
        comparison=comparison, reference=reference, tie_rule=tie_rule,
        margin=margin, test_type=test_type,
    )
    logger.info("=== Significance testing complete ===")


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

    # ---- create-all-visuals ----
    subparsers.add_parser(
        "create-all-visuals",
        help="Generate visualizations only"
    )

    # ---- create-multipanel-visuals ----
    subparsers.add_parser(
        "create-multipanel-visuals",
        help="Generate only the IEEE multipanel figures (skips per-combination single-panel plots)"
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
        help="Aggregate results/metrics/*.json into one CSV (mean + 95%% CI per metric "
             "plus resource-profiling columns, one row per mode/dataset/model)"
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
    metrics_csv_parser.add_argument(
        "--profiles-dir",
        default="results/resource_profiles",
        help="Directory containing per-run resource-profile JSON files, joined per "
             "mode/model/dataset (default: results/resource_profiles)"
    )

    # ---- paired-bootstrap-tests ----
    stats_parser = subparsers.add_parser(
        "paired-bootstrap-tests",
        help="Paired bootstrap significance tests on the stored replicate metrics: all "
             "pairwise mode comparisons within each (dataset, classifier) cell, "
             "Holm-Bonferroni corrected per cell"
    )
    stats_parser.add_argument(
        "--metrics-dir",
        default="results/metrics",
        help="Directory containing per-run metrics JSON files (default: results/metrics)"
    )
    stats_parser.add_argument(
        "--output",
        default=None,
        help="Path to write the results table "
             "(default: results/stats/paired_bootstrap__{metric}.csv)"
    )
    stats_parser.add_argument(
        "--metric",
        default="roc_auc",
        choices=["accuracy", "precision", "recall", "f1", "roc_auc"],
        help="Metric to test (default: roc_auc)"
    )
    stats_parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Modes to compare: exact names and/or glob patterns, e.g. "
             "--modes standard 'fhe_*' arf_300 (default: every mode found on disk). "
             "Quote patterns so the shell does not expand them. Filtering matters: the "
             "Holm family is every pairwise test in a (dataset, classifier) cell, so "
             "selecting all 32 modes means 496 tests per cell, and against the "
             "~0.002 bootstrap p-value floor that leaves essentially no power."
    )
    stats_parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to include; exact names and/or globs (default: all)"
    )
    stats_parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Classifiers to include; exact names and/or globs (default: all)"
    )
    stats_parser.add_argument(
        "--comparison",
        default="pairwise",
        choices=["pairwise", "reference", "adjacent"],
        help="Contrast design within each cell, which sets the Holm family size: "
             "'pairwise' = every pair, C(k,2) tests; 'reference' = --reference vs "
             "each other mode, k-1 tests (use for prespecified questions such as "
             "each synthesis scale vs 100%%, or each bit width vs Real); "
             "'adjacent' = consecutive modes in canonical order, k-1 tests "
             "(dose-response secondary analyses). Default: pairwise"
    )
    stats_parser.add_argument(
        "--reference",
        default=None,
        help="Reference mode for --comparison reference, e.g. --reference standard "
             "or --reference arf_100. Implies --comparison reference."
    )
    stats_parser.add_argument(
        "--tie-rule",
        default="split",
        choices=["split", "conservative", "exclude"],
        help="How replicates with an exact zero difference are counted. 'split' "
             "(default) halves them, the standard sign-test treatment. "
             "'conservative' charges them all against significance, which is right "
             "when ties mean identical predictions but makes any comparison with "
             ">24 ties impossible to call significant. 'exclude' drops them and "
             "reduces the effective B."
    )
    stats_parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Non-inferiority / equivalence margin on the metric scale "
             "(default: 0.05 ROC-AUC). Used by --test-type noninferiority and "
             "equivalence; p_noninf and p_equiv are reported for every run."
    )
    stats_parser.add_argument(
        "--test-type",
        default="difference",
        choices=["difference", "noninferiority", "equivalence"],
        help="Which p-value is the primary, Holm-corrected one. 'difference' "
             "(default) tests against no difference; 'noninferiority' tests that "
             "mode_a is not worse than mode_b by more than --margin; "
             "'equivalence' is TOST against +/- --margin. All three are always "
             "reported raw in p_diff / p_noninf / p_equiv."
    )
    stats_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the significant_holm column (default: 0.05)"
    )
    stats_parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "markdown", "both"],
        help="Output format (default: csv). 'markdown'/'both' also writes a .md table."
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

    elif args.command == "create-all-visuals":
        create_all_visuals()

    elif args.command == "create-multipanel-visuals":
        create_multipanel_visuals()

    elif args.command == "verify-gpu":
        verify_gpu(args.venv, args.device)

    elif args.command == "aggregate-internal-validation-bootstrap":
        aggregate_internal_validation_bootstrap_results(args.results_dir, args.output)

    elif args.command == "aggregate-metrics-csv":
        aggregate_metrics_to_csv(args.metrics_dir, args.output, args.profiles_dir)

    elif args.command == "paired-bootstrap-tests":
        run_paired_bootstrap_tests(args.metrics_dir, args.output, args.metric,
                                   args.modes, args.datasets, args.models,
                                   args.alpha, args.format, args.comparison,
                                   args.reference, args.tie_rule, args.margin,
                                   args.test_type)

    elif args.command == "generate-seeds":
        generate_seeds(args.seed, args.length)

    elif args.command == "list-n-bits":
        list_n_bits(args.config, args.out)

    elif args.command == "list-synth-scales":
        list_synth_scales(args.config, args.out)