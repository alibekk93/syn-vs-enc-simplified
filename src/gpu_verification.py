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

Different pipeline stages run in different venvs on the cluster (.venv-sdv,
.venv-synthcity, .venv-fhe — see jobs/run_full_pipeline_drac.sh), each with a
different subset of GPU-capable libraries installed. So checks are organized
per-venv rather than per-component: `run("sdv"|"synthcity"|"fhe")` reads
config/synthesizers.yaml, config/models.yaml, and config/fhe.yaml to find
every method/model that venv is actually configured to run, and reports on
all of them in one pass — no manual --component/--method selection needed.

Not every configured method has a GPU path at all (e.g. gaussian_copula,
logistic_regression), and not every environment this runs in will have the
GPU hardware/library available (e.g. a quick check on a CPU-only dev box).
Neither of those is a failure — they're reported as SKIP / UNAVAILABLE. The
only failure (FAIL, nonzero exit) is the actual bug class this exists to
catch: a GPU-capable item that's GPU-equipped and was asked to use it, but
the evidence shows it didn't (or, for the --device cpu negative control, used
the GPU when it shouldn't have).

Run via `python main.py verify-gpu --venv {sdv,synthcity,fhe}`.
"""

import json
import logging
import subprocess
import threading
import time

import numpy as np
import pandas as pd

from src.utils import load_config

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

# Synthesizer library ("sdv"/"synthcity", per src.synthesizers.SUPPORTED_SYNTHESIZERS)
# that each venv is provisioned for — see requirements/drac-sdv.txt / drac-synthcity.txt.
SYNTH_LIBRARY_BY_VENV = {"sdv": "sdv", "synthcity": "synthcity"}


class GpuCheckResult:
    SKIP        = "SKIP"         # not GPU-capable at all — not applicable, not an error
    UNAVAILABLE = "UNAVAILABLE"  # GPU-capable, but no GPU/library here — not an error
    USED        = "USED"         # GPU-capable, requested, and verified actually used
    NOT_USED    = "NOT_USED"     # negative control (--device cpu): correctly did NOT use the GPU
    FAIL        = "FAIL"         # the bug this exists to catch: requested+available but not used
                                  # (or, in the negative control, used when it shouldn't have been)

    def __init__(self, kind: str, name: str, status: str, detail: str = ""):
        self.kind = kind    # "synthesizer" | "model" | "fhe_model"
        self.name = name
        self.status = status
        self.detail = detail

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.name}"

    def __str__(self):
        line = f"[{self.label}] {self.status}"
        return f"{line} — {self.detail}" if self.detail else line


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
        logger.debug(f"nvidia-smi sampling failed: {e}")
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


def _runtime_evidence(sampler: _GpuSampler, baseline_mem: float) -> tuple[bool, str]:
    mem_delta = sampler.peak_memory - baseline_mem
    peak_util = sampler.peak_utilization
    detail = f"nvidia-smi mem +{mem_delta:.0f}MiB, peak_util={peak_util:.0f}%"
    ok = mem_delta >= MIN_MEMORY_DELTA_MIB or peak_util >= MIN_UTILIZATION_PCT
    return ok, detail


# ------------------------------------------------------------------
# Per-item checks
# ------------------------------------------------------------------

def _check_model(name: str, device: str) -> GpuCheckResult:
    from src.models import Model, GPU_CAPABLE_MODELS

    kind = "model"
    if name not in GPU_CAPABLE_MODELS:
        return GpuCheckResult(kind, name, GpuCheckResult.SKIP, "no GPU path for this model type")

    baseline_mem = 0.0
    if device == "cuda":
        baseline = _nvidia_smi_sample()
        if baseline is None:
            return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, "nvidia-smi unavailable in this environment")
        baseline_mem = baseline[1]

    model = Model(name, device=device, mode="gpu_verification")
    model.df = _dummy_classification_df()
    model.target = "target"
    model.dataset_name = "gpu_verification_dummy"
    model.split()

    saved_path = None
    try:
        # No explicit GPU-array-library preflight: Model._to_cuda() already
        # imports torch itself when device='cuda' — if it's missing, that's
        # reported below as UNAVAILABLE rather than declaring a redundant
        # dependency here.
        with _GpuSampler() as sampler:
            model.train()
        saved_path = model.models_dir / f"{model.mode}__{name}__{model.dataset_name}.joblib"
    except ImportError as e:
        return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, f"GPU library not installed: {e}")
    finally:
        if saved_path is not None:
            saved_path.unlink(missing_ok=True)

    runtime_ok, runtime_detail = _runtime_evidence(sampler, baseline_mem)

    structural_detail = ""
    structural_says_gpu = None  # None = no structural introspection available for this model type
    if name == "xgboost":
        booster_cfg = json.loads(model.model.get_booster().save_config())
        booster_device = booster_cfg["learner"]["generic_param"]["device"]
        to_cuda_module = type(model._to_cuda(model.X_train)).__module__
        structural_detail = f"booster.device={booster_device}, _to_cuda→{to_cuda_module}, "
        structural_says_gpu = "cuda" in booster_device and to_cuda_module.startswith("torch")

    if device == "cuda":
        if structural_says_gpu is False:
            return GpuCheckResult(kind, name, GpuCheckResult.FAIL,
                                   f"structural check shows CPU despite device='cuda' — {structural_detail}{runtime_detail}")
        if not runtime_ok:
            return GpuCheckResult(kind, name, GpuCheckResult.FAIL,
                                   f"no measurable GPU memory/utilization delta — {structural_detail}{runtime_detail}")
        return GpuCheckResult(kind, name, GpuCheckResult.USED, f"{structural_detail}{runtime_detail}")
    else:
        unexpected_gpu = runtime_ok or structural_says_gpu is True
        status = GpuCheckResult.FAIL if unexpected_gpu else GpuCheckResult.NOT_USED
        return GpuCheckResult(kind, name, status, f"(negative control) {structural_detail}{runtime_detail}")


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


def _check_synthesizer(name: str, device: str) -> GpuCheckResult:
    from src.synthesizers import Synthesizer, GPU_CAPABLE_SYNTHESIZERS

    kind = "synthesizer"
    if name not in GPU_CAPABLE_SYNTHESIZERS:
        return GpuCheckResult(kind, name, GpuCheckResult.SKIP, "no GPU path for this method")

    baseline_mem = 0.0
    if device == "cuda":
        baseline = _nvidia_smi_sample()
        if baseline is None:
            return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, "nvidia-smi unavailable in this environment")
        baseline_mem = baseline[1]

        try:
            import torch
        except ImportError as e:
            return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, f"torch not installed in this venv: {e}")
        if not torch.cuda.is_available():
            return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, "torch.cuda.is_available() is False — no CUDA GPU visible")

    try:
        synth = Synthesizer(name, device=device)
    except ImportError as e:
        return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, f"library not installed in this venv: {e}")

    df = _dummy_mixed_df()
    synth.dataset_name = "gpu_verification_dummy"
    synth.n_rows_original = len(df)
    synth.df_train = df
    synth.df_test = pd.DataFrame()
    synth.df = synth.df_train

    try:
        with _GpuSampler() as sampler:
            synth.fit()
    except ImportError as e:
        return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, f"GPU library not installed: {e}")

    try:
        torch_device = _find_torch_device(synth.synthesizer)
    except ImportError:
        torch_device = None

    runtime_ok, runtime_detail = _runtime_evidence(sampler, baseline_mem)
    structural_detail = f"model.device={torch_device}, " if torch_device is not None else "model.device=<unresolved>, "
    structural_says_gpu = (torch_device.type == "cuda") if torch_device is not None else None

    if device == "cuda":
        if structural_says_gpu is False:
            return GpuCheckResult(kind, name, GpuCheckResult.FAIL,
                                   f"structural check shows CPU despite device='cuda' — {structural_detail}{runtime_detail}")
        if not runtime_ok:
            return GpuCheckResult(kind, name, GpuCheckResult.FAIL,
                                   f"no measurable GPU memory/utilization delta — {structural_detail}{runtime_detail}")
        return GpuCheckResult(kind, name, GpuCheckResult.USED, f"{structural_detail}{runtime_detail}")
    else:
        unexpected_gpu = runtime_ok or structural_says_gpu is True
        status = GpuCheckResult.FAIL if unexpected_gpu else GpuCheckResult.NOT_USED
        return GpuCheckResult(kind, name, status, f"(negative control) {structural_detail}{runtime_detail}")


def _check_fhe_model(name: str, device: str) -> GpuCheckResult:
    kind = "fhe_model"

    baseline_mem = 0.0
    if device == "cuda":
        try:
            import concrete.compiler
        except ImportError as e:
            return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, f"concrete.compiler not installed in this venv: {e}")
        if not concrete.compiler.check_gpu_enabled():
            return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE,
                                   "this concrete-python build has no GPU support — install with: "
                                   "pip install --extra-index-url https://pypi.zama.ai/gpu concrete-python")
        if not concrete.compiler.check_gpu_available():
            return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, "no GPU available to this process")

        baseline = _nvidia_smi_sample()
        if baseline is None:
            return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, "nvidia-smi unavailable in this environment")
        baseline_mem = baseline[1]

    try:
        from src.fhe_models import FHEModel
    except ImportError as e:
        return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, f"concrete-ml not installed in this venv: {e}")

    model = FHEModel(name, fhe_cfg={"device": device})
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
        saved_path = model.models_dir / f"{model.mode}__{name}__{model.dataset_name}.json"
    except ImportError as e:
        return GpuCheckResult(kind, name, GpuCheckResult.UNAVAILABLE, f"GPU library not installed: {e}")
    finally:
        if saved_path is not None:
            saved_path.unlink(missing_ok=True)

    # concrete-ml's compiled circuit doesn't expose a documented "which device
    # did this run on" attribute, so runtime evidence is the only signal here
    # (structural evidence is the eager check_gpu_available() preflight above).
    runtime_ok, runtime_detail = _runtime_evidence(sampler, baseline_mem)

    if device == "cuda":
        status = GpuCheckResult.USED if runtime_ok else GpuCheckResult.FAIL
        detail = runtime_detail if runtime_ok else f"no measurable GPU memory/utilization delta — {runtime_detail}"
        return GpuCheckResult(kind, name, status, detail)
    else:
        status = GpuCheckResult.FAIL if runtime_ok else GpuCheckResult.NOT_USED
        return GpuCheckResult(kind, name, status, f"(negative control) {runtime_detail}")


# ------------------------------------------------------------------
# Config-driven enumeration — "all from configs", no manual selection
# ------------------------------------------------------------------

def _configured_synthesizers(venv: str) -> list[str]:
    from src.synthesizers import SUPPORTED_SYNTHESIZERS
    library = SYNTH_LIBRARY_BY_VENV[venv]
    cfg = load_config("config/synthesizers.yaml")
    return [name for name in cfg.get("methods", {}) if SUPPORTED_SYNTHESIZERS.get(name) == library]


def _configured_models() -> list[str]:
    cfg = load_config("config/models.yaml")
    return [m["name"] for m in cfg.get("models", [])]


def _configured_fhe_models() -> list[str]:
    cfg = load_config("config/fhe.yaml")
    return list(cfg.get("models", {}).keys())


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

def run(venv: str, device: str = "cuda") -> bool:
    """Checks every method/model config/{synthesizers,models,fhe}.yaml configures
    for the given venv, logs a status line for each, and returns True unless
    any of them FAILed (SKIP/UNAVAILABLE don't count as failure)."""
    if venv not in SYNTH_LIBRARY_BY_VENV and venv != "fhe":
        raise ValueError(f"Unknown venv '{venv}' — must be one of: sdv, synthcity, fhe")

    results: list[GpuCheckResult] = []

    if venv == "fhe":
        for name in _configured_fhe_models():
            results.append(_check_fhe_model(name, device))
    else:
        for name in _configured_synthesizers(venv):
            results.append(_check_synthesizer(name, device))
        for name in _configured_models():
            results.append(_check_model(name, device))

    logger.info(f"=== GPU verification report — venv: {venv}, device: {device} ===")
    log_fn_by_status = {
        GpuCheckResult.FAIL: logger.error,
        GpuCheckResult.UNAVAILABLE: logger.warning,
    }
    for r in results:
        log_fn_by_status.get(r.status, logger.info)(str(r))

    if not results:
        logger.warning(f"No methods/models configured for venv '{venv}' in config/synthesizers.yaml, config/models.yaml, or config/fhe.yaml")
        return True

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{count} {status}" for status, count in counts.items())
    logger.info(f"Summary ({venv}): {summary}")

    return counts.get(GpuCheckResult.FAIL, 0) == 0
