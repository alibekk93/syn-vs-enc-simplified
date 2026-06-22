"""FHE training pipeline — trains and evaluates FHE models."""

import logging
from src.utils import load_config, inject_n_bits, model_n_bits
from src.fhe_models import FHEModel
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG  = "config/datasets.yaml"


MODELS_CFG    = "config/models.yaml"
FHE_CFG       = "config/fhe.yaml"
RESOURCE_CFG  = "config/resource_profiling.yaml"


def _log_section(title: str) -> None:
    bar = "=" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def _check_device(device: str) -> None:
    """Fails fast if `device='cuda'` is requested but unavailable, instead of
    failing deep inside `.compile()` (wasting a GPU job allocation)."""
    if device != "cuda":
        return

    import concrete.compiler

    if not concrete.compiler.check_gpu_enabled():
        raise RuntimeError(
            "device='cuda' requested, but this concrete-python build has no GPU support. "
            "Install the GPU build with: "
            "pip install --extra-index-url https://pypi.zama.ai/gpu concrete-python"
        )
    if not concrete.compiler.check_gpu_available():
        raise RuntimeError(
            "device='cuda' requested, but no GPU is available to this process."
        )


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run(
    datasets: list[str] | None = None,
    models:   list[str] | None = None,
    fhe_mode: str = "simulate",
    n_bits: int | None = None,
    device: str | None = None,
    fhe_config_override=None,
    datasets_config: str = "config/datasets.yaml",
    resource_config: str = "config/resource_profiling.yaml",
    models_config: str = "config/models.yaml",
) -> dict:
    """
    Runs the FHE pipeline once, training each model at its configured n_bits.

    To run a single explicit n_bits value across all models, pass `n_bits`.
    To sweep multiple n_bits values, call this once per value (see
    `main.py`'s `run-experiment --n-bits N` and `list-n-bits`) — each call
    is independent and writes uniquely-named output files, so calls can run
    in parallel as separate processes.

    `device` overrides config/fhe.yaml's `device` setting ("cpu" or "cuda")
    for `.compile()`. Checked eagerly so a bad "cuda" request fails before
    any training happens.

    Returns:
        dict: {dataset: {model: {metrics, profiling, n_bits}}}
    """

    targets_datasets = datasets or list(load_config(datasets_config).keys())
    targets_models   = models   or [m["name"] for m in load_config(models_config).get("models", [])]

    fhe_config = fhe_config_override or load_config(FHE_CFG)
    if n_bits is not None:
        fhe_config = inject_n_bits(fhe_config, n_bits)

    active_device = device or fhe_config.get("device", "cpu")
    _check_device(active_device)

    logger.debug(
        f"FHE pipeline started — datasets: {targets_datasets}, "
        f"models: {targets_models}, fhe_mode: {fhe_mode}, device: {active_device}"
    )

    _log_section(f"FHE  |  mode: {fhe_mode}  |  device: {active_device}")

    results = {}

    for dataset_name in targets_datasets:
        results[dataset_name] = {}

        for model_name in targets_models:

            n_bits_for_model = model_n_bits(fhe_config, model_name)

            logger.info(f"--- FHE {model_name} on {dataset_name} (n_bits={n_bits_for_model}) ---")

            profiler = ResourceProfiler(load_config(resource_config))

            try:
                model = FHEModel(
                    model_name,
                    cfg=models_config,
                    mode="fhe",
                    fhe_cfg=fhe_config,
                )

                # ------------------------
                # Training phase
                # ------------------------
                profiler.start_memory_sampling(phase="training")

                with profiler.time_block("data_loading"):
                    model.load_data(dataset_name, dataset_cfg=datasets_config)
                    model.split()

                with profiler.time_block("training_fit"):
                    model.model.fit(model.X_train, model.y_train)

                with profiler.time_block("training_compile"):
                    model.model.compile(model.X_train, device=active_device)

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

                fhe_circuit = getattr(model.model, "fhe_circuit", None)
                complexity  = getattr(fhe_circuit, "complexity", None)
                profiler.log_fhe(complexity=complexity)

                model_path = f"models/fhe_{n_bits_for_model}__{model_name}__{dataset_name}.json"
                data_path  = f"data/processed/{dataset_name}.csv"

                profiler.log_storage(model_path=model_path, data_path=data_path)

                result_obj = {
                    "metrics": metrics,
                    "profiling": profiler.export(),
                    "n_bits": n_bits_for_model,
                }

                results[dataset_name][model_name] = result_obj

                logger.info(
                    f"[FHE profiling] n_bits={n_bits_for_model} | {dataset_name} | {model_name} "
                    f"| inference={round(end - start, 4)}s | circuit_complexity={complexity}"
                )

                # Save profiling with n_bits included
                profiler.save(
                    f"fhe_{n_bits_for_model}__{model_name}__{dataset_name}"
                )

                profiler.reset()

            except Exception as e:
                logger.error(f"Failed: FHE {model_name} on {dataset_name}: {e}")

                results[dataset_name][model_name] = {
                    "error": str(e),
                    "profiling": profiler.export(),
                    "n_bits": n_bits_for_model,
                }

                profiler.reset()

    logger.debug("FHE pipeline complete.")
    return results