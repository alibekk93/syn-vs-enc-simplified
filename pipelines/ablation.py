# pipelines/ablation.py

import logging
import copy
import json
from pathlib import Path

from src.utils import load_config
from pipelines import fhe

logger = logging.getLogger(__name__)

FHE_CFG = "config/fhe.yaml"
ABL_CFG = "config/ablation.yaml"


def _expand_n_bits(cfg):
    nb = cfg["fhe"]["n_bits"]

    if isinstance(nb, list):
        return nb

    start = nb.get("start")
    end   = nb.get("end")
    step  = nb.get("step", 1)

    return list(range(start, end + 1, step))


def _inject_n_bits(fhe_cfg, n_bits):
    new_cfg = copy.deepcopy(fhe_cfg)

    for model_name in new_cfg.get("models", {}):
        new_cfg["models"][model_name]["n_bits"] = n_bits

    return new_cfg


def run(datasets=None, models=None, fhe_mode="simulate"):
    logger.info("=== Ablation study (n_bits) ===")

    base_fhe_cfg = load_config(FHE_CFG)
    abl_cfg      = load_config(ABL_CFG)

    n_bits_values = _expand_n_bits(abl_cfg)

    output_dir = Path(abl_cfg.get("output", {}).get("dir", "results/ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for n_bits in n_bits_values:
        logger.info(f"[Ablation] Running n_bits={n_bits}")

        # override config
        modified_cfg = _inject_n_bits(base_fhe_cfg, n_bits)

        # monkey-patch through argument (small extension needed in fhe.run)
        results = fhe.run(
            datasets=datasets,
            models=models,
            fhe_mode=fhe_mode,
            fhe_config_override=modified_cfg,
        )

        all_results[n_bits] = results

        # save per-run
        out_file = output_dir / f"n_bits_{n_bits}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    # save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("=== Ablation complete ===")
    return all_results