# src/utils.py
"""Utility functions."""

import json
import logging
import random
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def generate_seeds(seed: int, length: int):
    """Generate a list of random seeds and save to file."""
    random.seed(seed)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(length)]
    with open("bootstrap_seeds.txt", "w") as f:
        for s in seeds:
            f.write(f"{s}\n")
    logger.info(f"Generated {length} seeds and saved to bootstrap_seeds.txt")


def aggregate_bootstrap(results_dir: str = "results/bootstrap", output_path: str = "results/bootstrap/aggregated.json"):
    """Concatenate all bootstrap results into a single hierarchical JSON file.

    Output structure:
        {metrics|resource_profiles} -> mode -> model -> dataset -> [per-seed records]

    Resource profile filename conventions handled:
        preprocessing__{dataset}                           -> mode=preprocessing, model=_
        {synthesizer}__{dataset}__synthesis                -> mode=<synthesizer>, model=_synthesis
        {mode}__{model}__{dataset}                         -> standard / fhe_N model files
        synthetic__{synthesizer}__{model}__{dataset}       -> mode=<synthesizer> (aligns with metrics)
    """
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
                    mode, model, dataset = parts[0], "_", parts[1]
                elif len(parts) == 3 and parts[2] == "synthesis":
                    mode, model, dataset = parts[0], "_synthesis", parts[1]
                elif len(parts) == 3:
                    mode, model, dataset = parts[0], parts[1], parts[2]
                elif len(parts) == 4 and parts[0] == "synthetic":
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