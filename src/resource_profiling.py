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
        self.enabled = self.cfg.get("enabled", True)
        self.track_memory = self.cfg.get("memory", {}).get("enabled", True)
        self.track_storage = self.cfg.get("storage", {}).get("enabled", True)
        self.track_time = self.cfg.get("time", {}).get("enabled", True)
        self.track_fhe = self.cfg.get("fhe", {}).get("enabled", True)

        self.reset()

    def reset(self):
        self.results = {
            "training_time": {},
            "inference_time": {},
            "memory": {},
            "storage": {},
            "fhe": {}
        }
        # For memory sampling during operations
        self._memory_samples = []
        self._memory_sampling = False
        self._memory_sample_thread = None

    # -----------------------------
    # TIME PROFILING
    # -----------------------------
    @contextmanager
    def time_block(self, name):
        if not self.track_time:
            yield
            return

        start = time.time()
        yield
        end = time.time()

        self.results["training_time"][name] = round(end - start, 4)

    def log_inference(self, total_time, n_samples):
        if not self.track_time:
            return

        self.results["inference_time"] = {
            "total": round(total_time, 4),
            "per_sample": round(total_time / max(n_samples, 1), 6)
        }

    # -----------------------------
    # MEMORY PROFILING
    # -----------------------------
    def start_memory(self):
        if not self.track_memory:
            return
        tracemalloc.start()

    def stop_memory(self):
        if not self.track_memory:
            return

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.results["memory"] = {
            "current_mb": round(current / 1e6, 3),
            "peak_mb": round(peak / 1e6, 3)
        }

    def start_memory_sampling(self, interval=0.1):
        """Start sampling memory usage at intervals to calculate average."""
        if not self.track_memory:
            return

        self._memory_samples = []
        self._memory_sampling = True
        self._memory_sample_thread = threading.Thread(
            target=self._memory_sampler,
            args=(interval,)
        )
        self._memory_sample_thread.daemon = True
        self._memory_sample_thread.start()

    def stop_memory_sampling(self):
        """Stop sampling and calculate average memory usage."""
        if not self.track_memory:
            return

        self._memory_sampling = False
        if self._memory_sample_thread:
            self._memory_sample_thread.join(timeout=1)

        if self._memory_samples:
            avg_memory = sum(self._memory_samples) / len(self._memory_samples)
            max_memory = max(self._memory_samples) if self._memory_samples else 0

            self.results["memory"] = {
                "average_mb": round(avg_memory, 3),
                "peak_mb": round(max_memory, 3)
            }
        else:
            # Fallback to tracemalloc if no samples
            self.stop_memory()

    def _memory_sampler(self, interval):
        """Background thread to sample memory usage."""
        process = psutil.Process()
        while self._memory_sampling:
            try:
                # Get memory usage in MB
                mem = process.memory_info().rss / 1e6
                self._memory_samples.append(mem)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(interval)

    # -----------------------------
    # STORAGE PROFILING
    # -----------------------------
    def file_size_mb(self, path):
        if not Path(path).exists():
            return 0
        return round(os.path.getsize(path) / 1e6, 3)

    def log_storage(self, model_path=None, data_path=None):
        if not self.track_storage:
            return

        model_size = self.file_size_mb(model_path) if model_path else None
        data_size = self.file_size_mb(data_path) if data_path else None

        self.results["storage"] = {
            "model_size_mb": model_size,
            "data_size_mb": data_size,
        }

        # Add comparison if both sizes are available
        if model_size is not None and data_size is not None and data_size > 0:
            self.results["storage"]["model_to_data_ratio"] = round(model_size / data_size, 4)
        elif model_size is not None and data_size is not None and model_size > 0:
            self.results["storage"]["data_to_model_ratio"] = round(data_size / model_size, 4)

    # -----------------------------
    # FHE PROFILING
    # -----------------------------
    def log_fhe(self, compile_time=None, complexity=None):
        if not self.track_fhe:
            return

        self.results["fhe"] = {
            "compile_time": compile_time,
            "circuit_complexity": complexity
        }

    # -----------------------------
    def export(self):
        return self.results