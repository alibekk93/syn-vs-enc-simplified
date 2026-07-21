import copy
import datetime
import json
import os
import socket
import time
import tracemalloc
import psutil
import threading
from pathlib import Path
from contextlib import contextmanager


# Environment variables that cap the thread pools used by BLAS, OpenMP and the
# Concrete/Rust backend.  Recorded in every profile so a run where the caps were
# not actually applied is identifiable after the fact rather than silently
# comparable to one where they were.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


class _Timer:
    """Handle yielded by timing context managers so callers can read elapsed time."""

    def __init__(self):
        self.elapsed = 0.0
        self.cpu = 0.0


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
        # Sampling starts at min_interval and backs off geometrically to
        # memory_interval, so sub-second phases still yield enough samples to
        # average without flooding the sample list during multi-minute ones.
        self.memory_min_interval = mem_cfg.get("min_interval", 0.01)

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
            # Mirrors training_time keys with CPU (not wall-clock) seconds.  Kept
            # in a separate bucket because the analysis layer computes
            # train_time as sum(training_time.values()) — anything added to
            # training_time itself would be silently folded into that total.
            "training_cpu_time": {},
            "inference_time": {},
            "memory":  {},
            "storage": {},
            "fhe":     {},
            "env":     {},
        }
        self._memory_samples       = []
        self._memory_sampling      = False
        self._memory_sample_thread = None
        self._current_memory_phase = None
        self._phase_baseline_mb    = None

        self.results["env"] = self._env_snapshot()

    # ------------------------------------------------------------------
    # ENVIRONMENT / PROVENANCE
    # ------------------------------------------------------------------

    @staticmethod
    def _rss_mb():
        """Current process RSS in MB, or None if it cannot be read."""
        try:
            return round(psutil.Process().memory_info().rss / 1e6, 3)
        except Exception:
            return None

    def _env_snapshot(self) -> dict:
        """
        Capture where and under what conditions this measurement ran.

        Timing on a shared cluster is only interpretable alongside the load the
        node was under and when the process started: a batch of array tasks that
        all launch simultaneously contend for the filesystem during library
        loading, which inflates every wall-clock measurement taken afterwards.
        Recording this makes such an effect visible in the results rather than
        something that has to be reconstructed from scheduler logs.
        """
        env = {
            "hostname":            socket.gethostname(),
            "pid":                 os.getpid(),
            "profile_start":       datetime.datetime.now().isoformat(timespec="seconds"),
            "cpu_count_logical":   os.cpu_count(),
            "baseline_rss_mb":     self._rss_mb(),
        }

        # SLURM identifiers — absent when running locally.
        for key, var in (
            ("slurm_job_id",            "SLURM_JOB_ID"),
            ("slurm_array_job_id",      "SLURM_ARRAY_JOB_ID"),
            ("slurm_array_task_id",     "SLURM_ARRAY_TASK_ID"),
            ("slurm_cpus_per_task",     "SLURM_CPUS_PER_TASK"),
        ):
            env[key] = os.environ.get(var)

        # Process start time separates "when the task was scheduled" from "when
        # this measurement began" — the gap is import/startup cost.
        try:
            env["process_start"] = datetime.datetime.fromtimestamp(
                psutil.Process().create_time()
            ).isoformat(timespec="seconds")
        except Exception:
            env["process_start"] = None

        # Cores actually available to this process (the real allocation), which
        # can be far smaller than cpu_count under a cgroup/cpuset.
        try:
            env["cpus_available"] = len(psutil.Process().cpu_affinity())
        except Exception:
            env["cpus_available"] = None

        # Load average is Unix-only; absent on Windows.
        try:
            load1, load5, load15 = os.getloadavg()
            env["loadavg_1m"]  = round(load1,  2)
            env["loadavg_5m"]  = round(load5,  2)
            env["loadavg_15m"] = round(load15, 2)
        except (OSError, AttributeError):
            env["loadavg_1m"] = env["loadavg_5m"] = env["loadavg_15m"] = None

        env["thread_env"] = {v: os.environ.get(v) for v in _THREAD_ENV_VARS}

        return env

    # ------------------------------------------------------------------
    # TIME PROFILING
    # ------------------------------------------------------------------

    @contextmanager
    def time_block(self, name):
        """
        Record wall-clock and CPU time for a named block.

        Uses perf_counter rather than time.time so the measurement is immune to
        NTP steps on long-running cluster jobs.  CPU time is recorded alongside:
        a large gap between elapsed and CPU time means the block spent its time
        blocked (I/O, contention) rather than computing.
        """
        timer = _Timer()

        if not self.track_time:
            yield timer
            return

        wall_start = time.perf_counter()
        cpu_start  = time.process_time()
        try:
            yield timer
        finally:
            timer.elapsed = time.perf_counter() - wall_start
            timer.cpu     = time.process_time() - cpu_start
            self.results["training_time"][name]     = round(timer.elapsed, 4)
            self.results["training_cpu_time"][name] = round(timer.cpu,     4)

    @contextmanager
    def inference_block(self, n_samples):
        """
        Time an inference phase and record it under results["inference_time"].

        Yields a handle whose `.elapsed` / `.cpu` are populated on exit, so
        callers can log the duration without re-timing it themselves.
        """
        timer = _Timer()

        if not self.track_time:
            yield timer
            return

        wall_start = time.perf_counter()
        cpu_start  = time.process_time()
        try:
            yield timer
        finally:
            timer.elapsed = time.perf_counter() - wall_start
            timer.cpu     = time.process_time() - cpu_start
            self.log_inference(timer.elapsed, n_samples, cpu_time=timer.cpu)

    def log_inference(self, total_time, n_samples, cpu_time=None):
        if not self.track_time:
            return

        self.results["inference_time"] = {
            "total":      round(total_time, 4),
            "per_sample": round(total_time / max(n_samples, 1), 6),
        }

        if cpu_time is not None:
            self.results["inference_time"]["cpu_total"] = round(cpu_time, 4)
            self.results["inference_time"]["cpu_per_sample"] = round(
                cpu_time / max(n_samples, 1), 6
            )

    # ------------------------------------------------------------------
    # MEMORY PROFILING
    # ------------------------------------------------------------------

    def start_memory_sampling(self, interval=None, phase="training"):
        """
        Start a background thread that samples RSS memory.

        Args:
            interval: Ceiling for the sampling interval in seconds (default from
                      config). Sampling begins at memory_min_interval and backs
                      off geometrically to this value.
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
        self._phase_baseline_mb    = self._rss_mb()
        self._memory_samples       = []
        self._memory_sampling      = True
        self._memory_sample_thread = threading.Thread(
            target=self._memory_sampler,
            args=(effective_interval, self.memory_min_interval),
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

        phase    = self._current_memory_phase or "training"
        baseline = self._phase_baseline_mb

        if self._memory_samples:
            samples = self._memory_samples
            avg     = sum(samples) / len(samples)
            peak    = max(samples)
            source  = "rss_sampling"
            n       = len(samples)
        else:
            # The phase finished inside a single sampling interval.  Take one
            # direct RSS reading rather than falling back to tracemalloc: it
            # keeps the units and meaning identical to the sampled path (whole
            # process RSS, including the native allocations that dominate FHE
            # work and that tracemalloc cannot see).
            single = self._rss_mb()
            avg = peak = single if single is not None else 0.0
            source = "rss_single_sample"
            n      = 0

        entry = {
            "average_mb": round(avg,  3),
            "peak_mb":    round(peak, 3),
            "source":     source,
            "n_samples":  n,
        }

        # Absolute RSS includes the interpreter and every imported library, so
        # it is dominated by a baseline that differs between modes (Concrete ML
        # vs SDV/torch).  The deltas isolate what the phase itself allocated.
        if baseline is not None:
            entry["baseline_mb"]       = baseline
            entry["average_delta_mb"]  = round(avg  - baseline, 3)
            entry["peak_delta_mb"]     = round(peak - baseline, 3)

        self.results["memory"][phase] = entry

        # Reset thread state so the next start_memory_sampling is clean.
        self._memory_sample_thread = None
        self._current_memory_phase = None
        self._phase_baseline_mb    = None

    def _memory_sampler(self, interval, min_interval):
        process = psutil.Process()
        delay   = min(min_interval, interval)
        while self._memory_sampling:
            try:
                self._memory_samples.append(process.memory_info().rss / 1e6)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(delay)
            if delay < interval:
                delay = min(interval, delay * 1.5)

    # Legacy tracemalloc helpers, retained for callers outside the sampling path.
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
            "source":     "tracemalloc",
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

    def log_env_extra(self, key: str, value):
        """
        Record an extra provenance value alongside the env block.

        Used for process-startup costs (library/toolchain warm-up) that are
        deliberately excluded from the timed blocks but still worth keeping,
        both to confirm the warm-up actually ran and to quantify what it
        removed from the measurement.
        """
        self.results["env"][key] = value

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
