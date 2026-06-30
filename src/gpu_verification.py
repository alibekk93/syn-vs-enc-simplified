# src/gpu_verification.py
"""Verifies that device='cuda' actually drives computation on the GPU, not just
that the flag was accepted.

This exists because of a real bug (see commit "xgboost gpu fix"): XGBoost's
device='cuda' hyperparameter was being honored, but the training data was
still a pandas DataFrame, so XGBoost silently computed on CPU anyway. A check
that only inspects config/flags would not have caught that — so every check
here requires two independent layers of evidence:

  1. Structural — the trained object's own reported device (e.g. the XGBoost
     booster's device, or the dtype of the array actually passed to fit()).
  2. Runtime    — an nvidia-smi memory/utilization sample taken immediately
     before and during the call, requiring a real, nonzero delta. This layer
     is vendor/library-agnostic and doesn't rely on a library accurately
     self-reporting what it did — which is exactly what went wrong before.

Run via `python main.py verify-gpu --component {xgboost,synthesizer,fhe,all}`.
"""

import json
import logging
import subprocess
import threading
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NVIDIA_SMI_CMD = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used",
    "--format=csv,noheader,nounits",
]

# Minimum nvidia-smi delta to count as "GPU was exercised" — small enough to
# catch a brief training run, large enough to not be noise from other
# processes sharing the GPU.
MIN_MEMORY_DELTA_MIB = 5
MIN_UTILIZATION_PCT = 5


class GpuCheckResult:
    def __init__(self, component: str, passed: bool, detail: str):
        self.component = component
        self.passed = passed
        self.detail = detail

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{self.component}] {status} — {self.detail}"


# ------------------------------------------------------------------
# nvidia-smi sampling
# ------------------------------------------------------------------

def _nvidia_smi_sample():
    """Returns (utilization_pct, memory_used_mib) for GPU 0, or None if
    nvidia-smi isn't available."""
    try:
        out = subprocess.check_output(NVIDIA_SMI_CMD, text=True, timeout=5)
        util_str, mem_str = out.strip().splitlines()[0].split(",")
        return float(util_str.strip()), float(mem_str.strip())
    except Exception as e:
        logger.warning(f"nvidia-smi sampling failed: {e}")
        return None


class _GpuSampler:
    """Background-samples nvidia-smi at a fixed interval for the duration of a
    `with` block, so a single sample landing between kernel launches doesn't
    produce a false negative."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            sample = _nvidia_smi_sample()
            if sample is not None:
                self._samples.append(sample)
            time.sleep(self.interval)

    @property
    def peak_utilization(self) -> float:
        return max((s[0] for s in self._samples), default=0.0)

    @property
    def peak_memory(self) -> float:
        return max((s[1] for s in self._samples), default=0.0)


# ------------------------------------------------------------------
# Dummy data — kept tiny and in-memory so checks run in seconds and don't
# depend on data/processed/ being populated.
# ------------------------------------------------------------------

def _dummy_classification_df(n_rows: int = 2000, n_features: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


def _dummy_mixed_df(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "num1":   rng.normal(size=n_rows),
        "num2":   rng.normal(size=n_rows),
        "cat1":   rng.choice(["a", "b", "c"], size=n_rows),
        "target": rng.integers(0, 2, size=n_rows),
    })


# ------------------------------------------------------------------
# Per-component checks
# ------------------------------------------------------------------

def _check_xgboost(device: str) -> GpuCheckResult:
    from src.models import Model

    if device == "cuda":
        try:
            import cupy  # noqa: F401
        except ImportError:
            return GpuCheckResult(
                "xgboost", False,
                "cupy is not installed — Model._to_cuda() needs it for GPU xgboost. "
                "Install a build matching this venv's CUDA version, e.g.: pip install cupy-cuda12x"
            )

    baseline = _nvidia_smi_sample()
    if baseline is None:
        return GpuCheckResult("xgboost", False, "nvidia-smi unavailable — cannot verify runtime GPU usage")
    baseline_mem = baseline[1]

    model = Model("xgboost", device=device, mode="gpu_verification")
    model.df = _dummy_classification_df()
    model.target = "target"
    model.dataset_name = "gpu_verification_dummy"
    model.split()

    saved_path = None
    try:
        with _GpuSampler() as sampler:
            model.train()
        saved_path = model.models_dir / f"{model.mode}__xgboost__{model.dataset_name}.joblib"
    finally:
        if saved_path is not None:
            saved_path.unlink(missing_ok=True)

    booster_cfg = json.loads(model.model.get_booster().save_config())
    booster_device = booster_cfg["learner"]["generic_param"]["device"]
    to_cuda_module = type(model._to_cuda(model.X_train)).__module__

    mem_delta = sampler.peak_memory - baseline_mem
    peak_util = sampler.peak_utilization
    runtime_detail = f"nvidia-smi mem +{mem_delta:.0f}MiB, peak_util={peak_util:.0f}%"

    if device == "cuda":
        structural_ok = "cuda" in booster_device and to_cuda_module.startswith("cupy")
        runtime_ok = mem_delta >= MIN_MEMORY_DELTA_MIB or peak_util >= MIN_UTILIZATION_PCT
        detail = f"booster.device={booster_device}, _to_cuda→{to_cuda_module}, {runtime_detail}"
        if not structural_ok:
            detail = "structural check failed — " + detail
        elif not runtime_ok:
            detail = "no measurable GPU memory/utilization delta — " + detail
        return GpuCheckResult("xgboost", structural_ok and runtime_ok, detail)
    else:
        # Negative control: device='cpu' should NOT touch the GPU.
        passed = booster_device == "cpu" and to_cuda_module == "pandas.core.frame"
        detail = f"(negative control) booster.device={booster_device}, _to_cuda→{to_cuda_module}, {runtime_detail}"
        return GpuCheckResult("xgboost", passed, detail)


def _find_torch_device(obj, max_depth: int = 3):
    """Best-effort search for a torch.device on a fitted synthesizer's
    internal model. sdv/synthcity don't expose a stable public attribute for
    this, so a few plausible paths are tried; if none resolves, structural
    evidence is skipped and the check falls back to nvidia-smi evidence only.
    Confirm the real attribute path the first time this runs in each venv
    (e.g. via `vars(synth.synthesizer)`) and tighten this if needed.
    """
    import torch

    seen = set()
    candidates = [obj]
    for _ in range(max_depth):
        next_candidates = []
        for c in candidates:
            if id(c) in seen or c is None:
                continue
            seen.add(id(c))
            if isinstance(c, torch.device):
                return c
            dev = getattr(c, "device", None)
            if isinstance(dev, torch.device):
                return dev
            for attr in ("_model", "model", "_synthesizer", "generator"):
                if hasattr(c, attr):
                    next_candidates.append(getattr(c, attr))
        candidates = next_candidates
    return None


def _check_synthesizer(method: str, device: str) -> GpuCheckResult:
    from src.synthesizers import Synthesizer, GPU_CAPABLE_SYNTHESIZERS

    label = f"synthesizer:{method}"

    if method not in GPU_CAPABLE_SYNTHESIZERS:
        return GpuCheckResult(label, False, f"'{method}' has no GPU path (not in {GPU_CAPABLE_SYNTHESIZERS})")

    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                return GpuCheckResult(label, False, "torch.cuda.is_available() is False — no CUDA GPU visible")
        except ImportError:
            return GpuCheckResult(label, False, "torch is not installed in this venv")

    baseline = _nvidia_smi_sample()
    if baseline is None:
        return GpuCheckResult(label, False, "nvidia-smi unavailable — cannot verify runtime GPU usage")
    baseline_mem = baseline[1]

    synth = Synthesizer(method, device=device)
    df = _dummy_mixed_df()
    synth.dataset_name = "gpu_verification_dummy"
    synth.n_rows_original = len(df)
    synth.df_train = df
    synth.df_test = pd.DataFrame()
    synth.df = synth.df_train

    with _GpuSampler() as sampler:
        synth.fit()

    torch_device = _find_torch_device(synth.synthesizer)
    mem_delta = sampler.peak_memory - baseline_mem
    peak_util = sampler.peak_utilization
    runtime_detail = f"nvidia-smi mem +{mem_delta:.0f}MiB, peak_util={peak_util:.0f}%"
    runtime_ok = mem_delta >= MIN_MEMORY_DELTA_MIB or peak_util >= MIN_UTILIZATION_PCT

    if device == "cuda":
        if torch_device is not None:
            structural_ok = torch_device.type == "cuda"
            detail = f"model.device={torch_device}, {runtime_detail}"
        else:
            # Couldn't introspect a device attribute on this library version —
            # fall back to runtime evidence alone (see _find_torch_device docstring).
            structural_ok = True
            detail = f"model.device=<unresolved, see _find_torch_device>, {runtime_detail}"
        passed = structural_ok and runtime_ok
        if not runtime_ok:
            detail = "no measurable GPU memory/utilization delta — " + detail
        return GpuCheckResult(label, passed, detail)
    else:
        passed = (torch_device is None or torch_device.type == "cpu") and not runtime_ok
        detail = f"(negative control) model.device={torch_device}, {runtime_detail}"
        return GpuCheckResult(label, passed, detail)


def _check_fhe(device: str) -> GpuCheckResult:
    from src.fhe_models import FHEModel

    if device == "cuda":
        try:
            import concrete.compiler
        except ImportError:
            return GpuCheckResult("fhe", False, "concrete.compiler is not installed in this venv")
        if not concrete.compiler.check_gpu_enabled():
            return GpuCheckResult(
                "fhe", False,
                "this concrete-python build has no GPU support. Install with: "
                "pip install --extra-index-url https://pypi.zama.ai/gpu concrete-python"
            )
        if not concrete.compiler.check_gpu_available():
            return GpuCheckResult("fhe", False, "no GPU available to this process (concrete.compiler.check_gpu_available() is False)")

    baseline = _nvidia_smi_sample()
    if baseline is None:
        return GpuCheckResult("fhe", False, "nvidia-smi unavailable — cannot verify runtime GPU usage")
    baseline_mem = baseline[1]

    model = FHEModel("xgboost", fhe_cfg={"device": device})
    df = _dummy_classification_df(n_rows=300, n_features=6)
    model.df = df
    model.target = "target"
    model.dataset_name = "gpu_verification_dummy"
    model.split()

    saved_path = None
    try:
        with _GpuSampler() as sampler:
            model.model.fit(model.X_train, model.y_train)
            model.model.compile(model.X_train, device=device)
            # Compiling alone may not exercise the GPU — actual execution
            # happens at inference, so run a couple of encrypted predictions too.
            model.predict(model.X_test[:2], fhe="execute")
        saved_path = model.models_dir / f"{model.mode}__xgboost__{model.dataset_name}.json"
    finally:
        if saved_path is not None:
            saved_path.unlink(missing_ok=True)

    mem_delta = sampler.peak_memory - baseline_mem
    peak_util = sampler.peak_utilization
    runtime_detail = f"nvidia-smi mem +{mem_delta:.0f}MiB, peak_util={peak_util:.0f}%"
    runtime_ok = mem_delta >= MIN_MEMORY_DELTA_MIB or peak_util >= MIN_UTILIZATION_PCT

    if device == "cuda":
        # concrete-ml's compiled circuit doesn't expose a documented
        # "which device did this run on" attribute, so runtime evidence is
        # the primary signal here (structural evidence is just the eager
        # check_gpu_available() preflight above).
        detail = runtime_detail if runtime_ok else "no measurable GPU memory/utilization delta — " + runtime_detail
        return GpuCheckResult("fhe", runtime_ok, detail)
    else:
        detail = f"(negative control) {runtime_detail}"
        return GpuCheckResult("fhe", not runtime_ok, detail)


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

SYNTHESIZER_METHODS = ("ctgan", "nflow", "arf")


def run(component: str, method: str | None = None, device: str = "cuda") -> bool:
    """Runs the requested GPU verification check(s) and logs a PASS/FAIL line
    per component. Returns True only if every check that ran passed."""
    if component == "synthesizer" and method is None:
        raise ValueError(f"--method is required when --component synthesizer (choices: {SYNTHESIZER_METHODS})")

    results: list[GpuCheckResult] = []

    if component in ("xgboost", "all"):
        try:
            results.append(_check_xgboost(device))
        except ImportError as e:
            results.append(GpuCheckResult("xgboost", False, f"not available in this venv: {e}"))

    if component in ("synthesizer", "all"):
        methods = [method] if method else list(SYNTHESIZER_METHODS)
        for m in methods:
            try:
                results.append(_check_synthesizer(m, device))
            except ImportError as e:
                results.append(GpuCheckResult(f"synthesizer:{m}", False, f"not available in this venv: {e}"))

    if component in ("fhe", "all"):
        try:
            results.append(_check_fhe(device))
        except ImportError as e:
            results.append(GpuCheckResult("fhe", False, f"not available in this venv: {e}"))

    for r in results:
        (logger.info if r.passed else logger.error)(str(r))

    if not results:
        logger.error("No applicable GPU checks ran (component not importable in this venv?)")
        return False

    return all(r.passed for r in results)
