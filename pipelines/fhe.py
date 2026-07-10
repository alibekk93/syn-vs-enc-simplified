"""FHE training pipeline — trains and evaluates FHE models."""

import gc
import ctypes
import logging
from src.utils import load_config, inject_n_bits, model_n_bits
from src.fhe_models import FHEModel, SUPPORTED_METRICS
from src.resource_profiling import ResourceProfiler
from src import bootstrap_utils

logger = logging.getLogger(__name__)

DATASETS_CFG  = "config/datasets.yaml"


MODELS_CFG    = "config/models.yaml"
FHE_CFG       = "config/fhe.yaml"
RESOURCE_CFG  = "config/resource_profiling.yaml"


def _malloc_trim() -> None:
    """Ask glibc to return freed arenas to the OS. No-op on non-glibc platforms."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


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
    n_bootstrap: int = 0,
    bootstrap_seed: int = 42,
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
                start   = _time.time()
                y_pred  = model.predict(model.X_test, fhe=fhe_mode)
                y_proba = model.predict_proba(model.X_test, fhe=fhe_mode)
                end     = _time.time()

                profiler.log_inference(end - start, len(model.X_test))
                profiler.stop_memory_sampling()

                fhe_circuit = getattr(model.model, "fhe_circuit", None)
                complexity  = getattr(fhe_circuit, "complexity", None)
                profiler.log_fhe(complexity=complexity)

                metric_names = load_config(models_config).get("metrics") or list(SUPPORTED_METRICS)
                metrics = bootstrap_utils.compute_metrics(model.y_test, y_pred, y_proba, metric_names)

                if n_bootstrap > 0:
                    iter_results  = bootstrap_utils.run_bootstrap(
                        y_true=model.y_test, y_pred=y_pred, y_proba=y_proba,
                        n=n_bootstrap, seed=bootstrap_seed, metric_names=metric_names,
                    )
                    metrics_to_save = bootstrap_utils.to_metric_lists(iter_results)
                else:
                    metrics_to_save = metrics

                bootstrap_utils.save_metrics_json(
                    path=model.results_dir / f"{model.mode}__{model_name}__{dataset_name}__test__metrics.json",
                    mode=model.mode, model_name=model_name, dataset_name=dataset_name,
                    split="test", metrics=metrics_to_save, n_bootstrap=n_bootstrap,
                    extra_fields={"fhe": fhe_mode, "n_bits": n_bits_for_model},
                )

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

            finally:
                # Release the compiled circuit and fitted model so their native memory
                # (Concrete ML's Rust/C++ allocations) is freed before the next model
                # starts.  Both profiler.save() and profiler.reset() have already run
                # in either the try or except branch above, so all measurements are
                # captured and written before we delete anything here.
                try:
                    del fhe_circuit
                except NameError:
                    pass
                try:
                    del model
                except NameError:
                    pass
                gc.collect()
                _malloc_trim()

        # Dataset boundary: return any remaining glibc arena pages to the OS so
        # accumulated native memory from this dataset doesn't carry into the next.
        gc.collect()
        _malloc_trim()

    logger.debug("FHE pipeline complete.")
    return results