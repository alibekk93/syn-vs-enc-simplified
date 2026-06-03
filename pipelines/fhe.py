"""FHE training pipeline — trains and evaluates FHE models, with optional n_bits sweep."""

import logging
import copy
from src.utils import load_config
from src.fhe_models import FHEModel
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG  = "config/datasets.yaml"
MODELS_CFG    = "config/models.yaml"
FHE_CFG       = "config/fhe.yaml"
RESOURCE_CFG  = "config/resource_profiling.yaml"


# ------------------------------------------------------------------
# Helpers for n_bits sweep (migrated from ablation.py)
# ------------------------------------------------------------------

def _expand_n_bits(cfg):
    """
    Returns a list of n_bits values to iterate over.

    Behavior:
        - If sweep not defined -> single run (None)
        - If list provided     -> return list
        - If start/end/step    -> expand range
    """
    sweep_cfg = cfg.get("sweep", {})

    if not sweep_cfg or not sweep_cfg.get("enabled", False):
        return [None]

    nb = sweep_cfg.get("n_bits")

    if nb is None:
        return [None]

    if isinstance(nb, list):
        return nb

    start = nb.get("start")
    end   = nb.get("end")
    step  = nb.get("step", 1)

    if start is None or end is None:
        return [None]

    return list(range(start, end + 1, step))


def _inject_n_bits(fhe_cfg, n_bits):
    """
    Inject n_bits into all model configs.
    """
    if n_bits is None:
        return fhe_cfg

    new_cfg = copy.deepcopy(fhe_cfg)

    for model_name in new_cfg.get("models", {}):
        new_cfg["models"][model_name]["n_bits"] = n_bits

    return new_cfg


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run(
    datasets: list[str] | None = None,
    models:   list[str] | None = None,
    fhe_mode: str = "simulate",
    fhe_config_override=None,
) -> dict:
    """
    Runs FHE pipeline with optional n_bits sweep.

    Returns:
        dict:
            {
              n_bits_value: {
                dataset: {
                  model: {
                    metrics,
                    profiling,
                    n_bits
                  }
                }
              }
            }
    """

    targets_datasets = datasets or list(load_config(DATASETS_CFG).keys())
    targets_models   = models   or [m["name"] for m in load_config(MODELS_CFG).get("models", [])]

    base_fhe_cfg = fhe_config_override or load_config(FHE_CFG)
    n_bits_values = _expand_n_bits(base_fhe_cfg)

    logger.info(
        f"FHE pipeline started — datasets: {targets_datasets}, "
        f"models: {targets_models}, fhe_mode: {fhe_mode}, "
        f"n_bits sweep: {n_bits_values}"
    )

    all_results = {}

    # --------------------------------------------------------------
    # Sweep loop (replaces ablation pipeline)
    # --------------------------------------------------------------
    for n_bits in n_bits_values:

        logger.info(f"[FHE] Running n_bits={n_bits}")

        fhe_config = _inject_n_bits(base_fhe_cfg, n_bits)
        run_results = {}

        for dataset_name in targets_datasets:
            run_results[dataset_name] = {}

            for model_name in targets_models:

                logger.info(f"--- FHE {model_name} on {dataset_name} (n_bits={n_bits}) ---")

                profiler = ResourceProfiler(load_config(RESOURCE_CFG))

                try:
                    model = FHEModel(
                        model_name,
                        cfg=MODELS_CFG,
                        mode="fhe",
                        fhe_cfg=fhe_config,
                    )

                    # ------------------------
                    # Training phase
                    # ------------------------
                    profiler.start_memory_sampling(phase="training")

                    with profiler.time_block("data_loading"):
                        model.load_data(dataset_name)
                        model.split()

                    with profiler.time_block("training_fit"):
                        model.model.fit(model.X_train, model.y_train)

                    with profiler.time_block("training_compile"):
                        model.model.compile(model.X_train)

                    model._save_model()

                    profiler.stop_memory_sampling()

                    # ------------------------
                    # Inference phase
                    # ------------------------
                    profiler.start_memory_sampling(phase="inference")

                    import time as _time
                    start = _time.time()
                    metrics = model.evaluate(fhe=fhe_mode)
                    end = _time.time()

                    profiler.log_inference(end - start, len(model.X_test))
                    profiler.stop_memory_sampling()

                    profiler.log_fhe(
                        complexity=getattr(model.model, "circuit_complexity", None)
                    )

                    model_path = f"models/fhe__{model_name}__{dataset_name}.json"
                    data_path  = f"data/processed/{dataset_name}.csv"

                    profiler.log_storage(model_path=model_path, data_path=data_path)

                    result_obj = {
                        "metrics": metrics,
                        "profiling": profiler.export(),
                        "n_bits": n_bits,
                    }

                    run_results[dataset_name][model_name] = result_obj

                    # Save profiling with n_bits included
                    profiler.save(
                        f"fhe_{n_bits}__{model_name}__{dataset_name}"
                    )

                    profiler.reset()

                except Exception as e:
                    logger.error(f"Failed: FHE {model_name} on {dataset_name}: {e}")

                    run_results[dataset_name][model_name] = {
                        "error": str(e),
                        "profiling": profiler.export(),
                        "n_bits": n_bits,
                    }

                    profiler.reset()

        # store results per n_bits
        all_results[n_bits] = run_results

    logger.info("FHE pipeline complete.")
    return all_results