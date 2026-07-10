"""Synthetic pipeline — synthesizes data and trains models on it."""

import logging
import time
from pathlib import Path
from src.utils import load_config, require_device
from src.synthesizers import Synthesizer
from src.models import Model, SUPPORTED_METRICS
from src.resource_profiling import ResourceProfiler

logger = logging.getLogger(__name__)

DATASETS_CFG     = "config/datasets.yaml"


MODELS_CFG       = "config/models.yaml"
SYNTHESIZERS_CFG = "config/synthesizers.yaml"
RESOURCE_CFG     = "config/resource_profiling.yaml"


def _log_section(title: str) -> None:
    bar = "=" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def run(
    datasets:             list[str] | None = None,
    synthesizers:         list[str] | None = None,
    models:               list[str] | None = None,
    oversampling_factors: list[int] | None = None,
    skip_training:        bool = False,
    device: str | None = None,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 42,
    datasets_config: str = "config/datasets.yaml",
    resource_config: str = "config/resource_profiling.yaml",
    models_config: str = "config/models.yaml",
    synthesizers_config: str = "config/synthesizers.yaml",
) -> dict:
    """
    Args:
        datasets:             List of dataset names (default: all).
        synthesizers:         List of synthesizer names (default: all in config).
        models:               List of model names (default: all in config).
        oversampling_factors: Percentages of original dataset size to sample
            (default: synthesizers.yaml `oversampling.factors`, e.g. [100, 150, 300]).
            Each synthesizer is fit once per dataset then sampled at every factor.
            Results and output files include the factor in the synthesizer name,
            e.g. "arf_100", "arf_150", "arf_300".
        skip_training: If True, only synthesize data without training models.
        device: "cpu" or "cuda" (default: synthesizers.yaml's `device`). Used
            for both synthesis (ctgan/nflow/arf only) and the downstream
            model training (xgboost only) — other synthesizers/models ignore
            it. Checked eagerly so a bad "cuda" request fails before any
            work happens.
        datasets_config: Path to datasets configuration file.
        resource_config: Path to resource profiling configuration file.
        models_config: Path to models configuration file.
        synthesizers_config: Path to synthesizers configuration file.

    Returns:
        Nested dict of {dataset: {synthesizer_factor: {model: metrics}}}
        where synthesizer_factor is e.g. "arf_100", "arf_150".
    """
    synth_cfg_all = load_config(synthesizers_config)

    targets_datasets     = datasets             or list(load_config(datasets_config).keys())
    targets_synthesizers = synthesizers         or [k for k in synth_cfg_all.get("methods", [])]
    targets_models       = models               or [m["name"] for m in load_config(models_config).get("models", [])]
    targets_factors      = oversampling_factors or list(synth_cfg_all.get("oversampling", {}).get("factors", [100]))

    active_device = device or synth_cfg_all.get("device", "cpu")
    require_device(active_device)

    logger.debug(
        f"Synthetic pipeline started — datasets: {targets_datasets}, "
        f"synthesizers: {targets_synthesizers}, models: {targets_models}, "
        f"oversampling_factors: {targets_factors}, device: {active_device}"
    )

    results = {}
    for synth_name in targets_synthesizers:
        _log_section(f"SYNTHETIC: {synth_name}  |  device: {active_device}")

        for dataset_name in targets_datasets:
            results.setdefault(dataset_name, {})

            logger.info(f"--- Fitting: {synth_name} on {dataset_name} ---")

            # Profiler for the fit phase (shared across all factors).
            synth_profiler = ResourceProfiler(load_config(resource_config))

            try:
                synth = Synthesizer(synth_name, cfg=synthesizers_config, device=active_device)

                synth_profiler.start_memory_sampling(phase="synthesis")

                with synth_profiler.time_block("synthesis_load"):
                    synth.load_data(dataset_name, dataset_cfg=datasets_config)
                with synth_profiler.time_block("synthesis_fit"):
                    synth.fit()
                with synth_profiler.time_block("synthesis_save"):
                    synth.save()

                synth_profiler.stop_memory_sampling()
                synth_profiler.save(f"{synth_name}__synthesis__{dataset_name}")

                for factor in targets_factors:
                    effective_name = f"{synth_name}_{factor}"
                    results[dataset_name][effective_name] = {}

                    logger.info(f"--- Sampling: {effective_name} on {dataset_name} ---")

                    # Per-factor profiler covers only the sampling step.
                    factor_profiler = ResourceProfiler(load_config(resource_config))

                    try:
                        factor_profiler.start_memory_sampling(phase="sampling")

                        n_rows = int(synth.n_rows_original * factor / 100)
                        with factor_profiler.time_block("synthesis_sample"):
                            synth.sample(num_rows=n_rows, oversampling_factor=factor)

                        factor_profiler.stop_memory_sampling()
                        factor_profiler.save(f"{effective_name}__sampling__{dataset_name}")

                        if skip_training:
                            results[dataset_name][effective_name] = {
                                "synthesis": synth_profiler.export(),
                                "sampling":  factor_profiler.export(),
                            }
                            continue

                        synthetic_dataset = f"{effective_name}__{dataset_name}"
                        for model_name in targets_models:
                            logger.info(f"--- {model_name} on synthetic {synthetic_dataset} ---")

                            # Create per-model profiler before the inner try block.
                            train_profiler = ResourceProfiler(load_config(resource_config))

                            try:
                                model = Model(model_name, cfg=models_config, mode=effective_name, device=active_device)

                                train_profiler.start_memory_sampling(phase="training")

                                with train_profiler.time_block("data_loading"):
                                    model.load_data(synthetic_dataset, dataset_cfg=datasets_config)
                                    model.save_dataset_name = dataset_name
                                    model.use_all_as_train()
                                    model.load_test_data(dataset_name, dataset_cfg=datasets_config)

                                with train_profiler.time_block("training"):
                                    model.train()

                                train_profiler.stop_memory_sampling()

                                train_profiler.start_memory_sampling(phase="inference")

                                start   = time.time()
                                metrics = model.evaluate()
                                end     = time.time()

                                train_profiler.log_inference(end - start, len(model.X_test))
                                train_profiler.stop_memory_sampling()

                                if n_bootstrap > 0:
                                    from src import bootstrap_utils
                                    metric_names = load_config(models_config).get("metrics") or list(SUPPORTED_METRICS)

                                    train_profiler.start_memory_sampling(phase="bootstrap_inference")
                                    iter_metrics, iter_times = bootstrap_utils.run_bootstrap(
                                        predict_fn=model.predict,
                                        predict_proba_fn=model.predict_proba,
                                        X_test=model.X_test, y_test=model.y_test,
                                        n=n_bootstrap, seed=bootstrap_seed, metric_names=metric_names,
                                    )
                                    train_profiler.stop_memory_sampling()

                                    train_profiler.results["inference_time"] = {
                                        "total":      [t["total"]      for t in iter_times],
                                        "per_sample": [t["per_sample"] for t in iter_times],
                                    }

                                    metrics_path = (
                                        model.results_dir
                                        / f"{effective_name}__{model_name}__{dataset_name}__test__metrics.json"
                                    )
                                    bootstrap_utils.overwrite_metrics_with_bootstrap(
                                        path=metrics_path,
                                        metric_lists=bootstrap_utils.to_metric_lists(iter_metrics),
                                        n_bootstrap=n_bootstrap,
                                    )

                                model_path = f"models/{effective_name}__{model_name}__{dataset_name}.joblib"
                                data_path  = f"data/processed/{synthetic_dataset}.csv"
                                train_profiler.log_storage(model_path=model_path, data_path=data_path)

                                results[dataset_name][effective_name][model_name] = {
                                    "metrics":   metrics,
                                    "profiling": {
                                        "synthesis": synth_profiler.export(),
                                        "sampling":  factor_profiler.export(),
                                        "training":  train_profiler.export(),
                                    },
                                }

                                train_profiler.save(
                                    f"{effective_name}__{model_name}__{dataset_name}"
                                )

                            except Exception as e:
                                logger.error(f"Training failed: {model_name} on {synthetic_dataset}: {e}")
                                results[dataset_name][effective_name][model_name] = {
                                    "error":     str(e),
                                    "profiling": train_profiler.export(),
                                }
                            finally:
                                train_profiler.reset()

                    except Exception as e:
                        logger.error(f"Sampling failed: {effective_name} on {dataset_name}: {e}")
                        results[dataset_name][effective_name] = {
                            "error":     str(e),
                            "profiling": factor_profiler.export(),
                        }
                    finally:
                        factor_profiler.reset()

            except Exception as e:
                logger.error(f"Synthesis failed: {synth_name} on {dataset_name}: {e}")
                results[dataset_name][synth_name] = {
                    "error":     str(e),
                    "profiling": synth_profiler.export(),
                }
            finally:
                synth_profiler.reset()

    logger.debug("Synthetic pipeline complete.")
    return results