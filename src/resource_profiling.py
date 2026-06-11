import copy
import json
import time
import tracemalloc
import psutil
import os
import threading
from pathlib import Path
from contextlib import contextmanager


class ResourceProfiler:
    def __init__(self, config: dict):
        self.cfg = config or {}

        self.enabled       = self.cfg.get("enabled", True)
        self.track_time    = self.enabled and self.cfg.get("time",    {}).get("enabled", True)
        self.track_memory  = self.enabled and self.cfg.get("memory",  {}).get("enabled", True)
        self.track_storage = self.enabled and self.cfg.get("storage", {}).get("enabled", True)
        self.track_fhe     = self.enabled and self.cfg.get("fhe",     {}).get("enabled", True)

        mem_cfg = self.cfg.get("memory", {})
        self.memory_interval = mem_cfg.get("interval", 0.1)

        log_cfg = self.cfg.get("logging", {})
        self.save_to_disk = log_cfg.get("save", False)
        self.output_dir   = Path(log_cfg.get("output_dir", "results/resource_profiles"))

        self.reset()

    def reset(self):
        thread = getattr(self, "_memory_sample_thread", None)
        if thread and thread.is_alive():
            self._memory_sampling = False
            thread.join(timeout=1)

        self.results = {
            "training_time":  {},
            "inference_time": {},
            "memory":  {},
            "storage": {},
            "fhe":     {},
        }
        self._memory_samples       = []
        self._memory_sampling      = False
        self._memory_sample_thread = None
        self._current_memory_phase = None

    # ------------------------------------------------------------------
    # TIME PROFILING
    # ------------------------------------------------------------------

    @contextmanager
    def time_block(self, name):
        """Context manager that records wall-clock time for a named block."""
        if not self.track_time:
            yield
            return

        start = time.time()
        yield
        self.results["training_time"][name] = round(time.time() - start, 4)

    def log_inference(self, total_time, n_samples):
        if not self.track_time:
            return

        self.results["inference_time"] = {
            "total":      round(total_time, 4),
            "per_sample": round(total_time / max(n_samples, 1), 6),
        }

    # ------------------------------------------------------------------
    # MEMORY PROFILING
    # ------------------------------------------------------------------

    def start_memory_sampling(self, interval=None, phase="training"):
        """
        Start a background thread that samples RSS memory at *interval* seconds.

        Args:
            interval: Sampling interval in seconds (default 0.1).
            phase:    Label written into results["memory"] (default "training").
                      Pass "inference" for the inference phase so results from
                      both phases are preserved.
        """
        if not self.track_memory:
            return

        if self._memory_sample_thread and self._memory_sample_thread.is_alive():
            self._memory_sampling = False
            self._memory_sample_thread.join(timeout=1)

        effective_interval = interval if interval is not None else self.memory_interval
        self._current_memory_phase = phase
        self._memory_samples       = []
        self._memory_sampling      = True
        self._memory_sample_thread = threading.Thread(
            target=self._memory_sampler,
            args=(effective_interval,),
            daemon=True,
        )
        self._memory_sample_thread.start()

    def stop_memory_sampling(self):
        """Stop sampling and store average/peak under the current phase label."""
        if not self.track_memory:
            return

        self._memory_sampling = False
        if self._memory_sample_thread:
            self._memory_sample_thread.join(timeout=1)

        phase = self._current_memory_phase or "training"

        if self._memory_samples:
            avg = sum(self._memory_samples) / len(self._memory_samples)
            peak = max(self._memory_samples)

            self.results["memory"][phase] = {
                "average_mb": round(avg,  3),
                "peak_mb":    round(peak, 3),
            }
        else:
            # Fallback: tracemalloc gives at least something if RSS sampling
            # produced no samples (very fast operations).
            self._tracemalloc_snapshot(phase)

        # Reset thread state so the next start_memory_sampling is clean.
        self._memory_sample_thread = None
        self._current_memory_phase = None

    def _memory_sampler(self, interval):
        process = psutil.Process()
        while self._memory_sampling:
            try:
                mem = process.memory_info().rss / 1e6
                self._memory_samples.append(mem)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(interval)

    # Legacy tracemalloc helpers kept for the fallback path only.
    def start_memory(self):
        if not self.track_memory:
            return
        tracemalloc.start()

    def stop_memory(self):
        if not self.track_memory:
            return
        self._tracemalloc_snapshot("training")

    def _tracemalloc_snapshot(self, phase):
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
        else:
            tracemalloc.start()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        self.results["memory"][phase] = {
            "current_mb": round(current / 1e6, 3),
            "peak_mb":    round(peak    / 1e6, 3),
        }

    # ------------------------------------------------------------------
    # STORAGE PROFILING
    # ------------------------------------------------------------------

    def file_size_mb(self, path):
        p = Path(path)
        if not p.exists():
            return 0
        return round(os.path.getsize(p) / 1e6, 3)

    def log_storage(self, model_path=None, data_path=None):
        if not self.track_storage:
            return

        model_size = self.file_size_mb(model_path) if model_path else None
        data_size  = self.file_size_mb(data_path)  if data_path  else None

        self.results["storage"].update({
            "model_size_mb": model_size,
            "data_size_mb":  data_size,
        })

        if model_size is not None and data_size is not None and data_size > 0:
            self.results["storage"]["model_to_data_ratio"] = round(model_size / data_size, 4)

    def log_storage_extra(self, key: str, value):
        """
        Add an extra key to the storage results via the public API
        instead of callers reaching into self.results["storage"] directly.

        Example:
            profiler.log_storage_extra("compression_ratio", 0.73)
        """
        if not self.track_storage:
            return
        self.results["storage"][key] = value

    # ------------------------------------------------------------------
    # FHE PROFILING
    # ------------------------------------------------------------------

    def log_fhe(self, complexity=None):
        """
        Records FHE-specific metadata.
        """
        if not self.track_fhe:
            return

        self.results["fhe"] = {
            "circuit_complexity": complexity,
        }

    # ------------------------------------------------------------------
    # EXPORT / SAVE
    # ------------------------------------------------------------------

    def export(self) -> dict:
        return copy.deepcopy(self.results)

    def save(self, label: str) -> None:
        """
        Persist the profiling results to disk as JSON.

        The file is written to:
            <output_dir>/<label>.json

        where <label> is supplied by the caller and should uniquely identify
        the run, e.g. "standard__logistic_regression__heart_disease".

        Does nothing when logging.save is False in the config.
        """
        if not self.save_to_disk:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{label}.json"

        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)