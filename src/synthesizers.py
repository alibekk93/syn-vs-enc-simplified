# src/synthesizers.py
"""Synthesizer class for fitting and sampling synthetic tabular data."""

import contextlib
import io
import logging
import os
import time
import warnings

# transformers logs a torch-version warning on import (pulled in transitively
# by sdv/synthcity's dependency tree, which never actually needs torch) —
# silence it before anything below has a chance to trigger that import.
logging.getLogger("transformers").setLevel(logging.ERROR)

import pandas as pd
from pathlib import Path
from typing import Optional, Union

from sklearn.model_selection import train_test_split

from src.utils import load_config

logger = logging.getLogger(__name__)

# sdv and synthcity are heavy, mutually-independent optional dependencies
# (experiments run them in separate venvs). Each is imported lazily, only
# once a synthesizer actually backed by that library is fit/saved/loaded,
# so an environment with only one of the two libraries installed never
# needs the other to even import this module.
_SDV = None
_SYNTHCITY = None
_NATIVE = None


def _load_sdv() -> dict:
    """Lazily import sdv and cache its constructor classes."""
    global _SDV
    if _SDV is None:
        from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
        from sdv.metadata import Metadata

        for _lib in ("sdv", "rdt", "copulas"):
            logging.getLogger(_lib).setLevel(logging.WARNING)

        _SDV = {
            "classes": {
                "gaussian_copula": GaussianCopulaSynthesizer,
                "ctgan": CTGANSynthesizer,
            },
            "Metadata": Metadata,
        }
    return _SDV


def _load_native() -> dict:
    """Lazily import and cache native synthesizer wrapper classes."""
    global _NATIVE
    if _NATIVE is None:
        from src.synthesizer_wrappers import ARFWrapper, BayesianNetworkWrapper, NFlowWrapper

        _NATIVE = {
            "bayesian_network": BayesianNetworkWrapper,
            "nflow": NFlowWrapper,
            "arf": ARFWrapper,
        }
    return _NATIVE


def _load_synthcity() -> dict:
    """Lazily import synthcity and cache its shared Plugins() registry."""
    global _SYNTHCITY
    if _SYNTHCITY is None:
        # bayesian_network's pgmpy structure search reports MI-score warnings
        # from joblib worker processes, which a local catch_warnings() can't
        # reach. Filter UserWarning process-wide — current process via
        # filterwarnings(), and any joblib workers spawned in a fresh
        # interpreter via PYTHONWARNINGS.
        os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning")
        warnings.filterwarnings("ignore", category=UserWarning)

        # Importing/initializing synthcity transitively imports pykeops and
        # compiles its JIT binder, which prints "[KeOps] Compiling ... OK"
        # straight to stdout. That's a one-time, harmless build step —
        # swallow it rather than surface it.
        with contextlib.redirect_stdout(io.StringIO()):
            import synthcity.logger as synthcity_log
            from synthcity.plugins import Plugins
            from synthcity.utils.serialization import save_to_file, load_from_file

            # synthcity logs through loguru (not stdlib logging) and adds
            # its own CRITICAL stderr sink on import — drop it, we log via
            # `logger` instead.
            synthcity_log.remove()

            # Plugins() re-scans every plugin file (including ones with
            # unmet optional deps, e.g. goggle) on each instantiation —
            # build it once.
            plugins = Plugins()

        _SYNTHCITY = {
            "plugins": plugins,
            "save_to_file": save_to_file,
            "load_from_file": load_from_file,
        }
    return _SYNTHCITY


# Maps synthesizer name -> backing library ("sdv", "native", or "synthcity").
SUPPORTED_SYNTHESIZERS = {
    "gaussian_copula": "sdv",
    "ctgan": "sdv",
    "bayesian_network": "native",
    "nflow": "native",
    "arf": "native",
}

# Of SUPPORTED_SYNTHESIZERS, only these are torch-backed neural models with a
# real GPU path. gaussian_copula is a statistical model (sdv) and
# bayesian_network is pgmpy structure search (synthcity) — both CPU-only,
# no GPU acceleration exists for them.
GPU_CAPABLE_SYNTHESIZERS = {"ctgan", "nflow", "arf"}


class Synthesizer:
    """
    Wraps an SDV synthesizer with fit/sample/save functionality,
    driven by config/synthesizers.yaml.

    Usage:
        synth = Synthesizer("gaussian_copula", cfg="config/synthesizers.yaml")
        synth.load_data("heart_disease")
        synth.fit()
        synthetic_df = synth.sample()
        synth.save()

        # or all at once:
        synthetic_df = synth.run("heart_disease")

        # reload later:
        synth = Synthesizer.load("synthesizers/gaussian_copula__heart_disease.pkl")
        synthetic_df = synth.sample()
    """

    PROCESSED_DIR = Path("data/processed")

    def __init__(self, name: str, cfg: str = "config/synthesizers.yaml", device: str | None = None):
        """
        Args:
            name:   Synthesizer name — must be a key in config and SUPPORTED_SYNTHESIZERS
            cfg:    Path to synthesizers.yaml
            device: "cpu" or "cuda" (default: synthesizers.yaml's `device`, normally
                    cpu). Only applied for GPU_CAPABLE_SYNTHESIZERS (ctgan, nflow,
                    arf) — gaussian_copula and bayesian_network have no GPU path
                    and ignore it.
        """
        all_cfg = load_config(cfg)
        output_cfg             = all_cfg.get("output", {})
        methods_cfg            = all_cfg.get("methods", {})

        if name not in methods_cfg:
            available = [k for k in methods_cfg]
            raise KeyError(f"Synthesizer '{name}' not found in config. Available: {available}")

        self.name              = name
        self.cfg               = all_cfg
        self.synth_cfg         = methods_cfg[name]
        self.split_cfg         = all_cfg.get("split", {})
        self.method            = name
        self.test_size         = self.split_cfg.get("test_size", None)
        self.random_state      = self.split_cfg.get("random_state", 42)
        self.stratify          = self.split_cfg.get("stratify", False)

        if self.method not in SUPPORTED_SYNTHESIZERS:
            raise KeyError(f"Method '{self.method}' is not supported. Supported: {list(SUPPORTED_SYNTHESIZERS)}")

        self.library           = SUPPORTED_SYNTHESIZERS[self.method]
        self.device            = device or all_cfg.get("device", "cpu")

        self.synthetic_dir    = Path(output_cfg.get("synthetic_dir", "data/synthetic"))
        self.synthesizers_dir = Path(output_cfg.get("synthesizers_dir", "synthesizers"))

        # Placeholders — populated in load_data() and fit()
        self.dataset_name: Optional[str]    = None
        self.n_rows_original: Optional[int] = None
        self.synthesizer: Optional[object]  = None
        self.df: Optional[pd.DataFrame]     = None

        logger.debug(f"[{self.name}] Initialized with method='{self.method}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warmup(self) -> float:
        """
        Import this synthesizer's backing library ahead of any timed work.

        fit() calls the relevant _load_*() helper as its first action, so
        without this the library import (sdv, synthcity, torch, pgmpy) is
        charged to the synthesis_fit measurement. That import cost differs
        substantially by backend — a torch-backed nflow pays far more than
        native arf — so leaving it in place adds a per-method constant to
        fit times that reads as a real difference in synthesis cost.

        Idempotent: the _load_*() helpers cache their result, so calling this
        makes fit()'s own call a no-op. Returns the import duration in seconds
        so callers can record it separately.
        """
        start = time.perf_counter()
        if self.library == "sdv":
            _load_sdv()
        elif self.library == "native":
            _load_native()
        else:
            _load_synthcity()
        elapsed = time.perf_counter() - start

        logger.info(
            f"[{self.name}] warmup: '{self.library}' imported in {elapsed:.3f}s "
            f"(excluded from fit measurement)"
        )
        return elapsed

    def load_data(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> None:
        """Load processed dataset and store it for fitting.

        Similar to Model.load_data but does not extract a target column.
        Additionally, splits the dataset into train/test.
        processed_path from the config entry overrides the default data/processed/ location.
        """
        ds_cfg = load_config(dataset_cfg)
        entry = ds_cfg.get(dataset_name, {})
        processed_path = entry.get("processed_path")
        path = Path(processed_path) if processed_path else self.PROCESSED_DIR / f"{dataset_name}.csv"
        logger.debug(f"[{self.name}] Loading data from {path}")
        df = pd.read_csv(path)

        # Store dataset name and original size (BEFORE split)
        self.dataset_name = dataset_name
        self.n_rows_original = len(df)

        # --- Split ---
        if self.test_size:
            test_size = self.test_size
            random_state = self.random_state
            target = entry.get("target")
            stratify_col = df[target] if (self.stratify and target) else None

            df_train, df_test = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
                stratify=stratify_col,
            )

            # Store splits
            self.df_train = df_train.reset_index(drop=True)
            self.df_test  = df_test.reset_index(drop=True)
        
        else:
            self.df_train = df.reset_index(drop=True)
            self.df_test  = pd.DataFrame()  # Empty test set if no split

        # Use TRAIN ONLY for fitting
        self.df = self.df_train

        logger.debug(
            f"[{self.name}] Loaded {self.n_rows_original} rows "
            f"(train={len(self.df_train)}, test={len(self.df_test)})"
        )

    def run(self, dataset_name: str) -> pd.DataFrame:
        """Full pipeline: load_data → fit → sample → save."""
        self.load_data(dataset_name)
        self.fit()
        synthetic_df = self.sample()
        self.save()
        return synthetic_df

    def fit(self) -> None:
        """Fit the synthesizer on the loaded data."""
        if self.df is None:
            raise RuntimeError("Call load_data() before fit()")

        params = dict(self.synth_cfg.get("parameters") or {})

        gpu_requested = self.device == "cuda"
        gpu_capable   = self.method in GPU_CAPABLE_SYNTHESIZERS
        if gpu_requested and not gpu_capable:
            logger.info(f"[{self.name}] device='cuda' requested but '{self.method}' has no GPU support — running on CPU")

        if self.library == "sdv":
            sdv = _load_sdv()
            metadata = sdv["Metadata"].detect_from_dataframe(self.df)
            logger.debug(f"[{self.name}] Metadata detected for {len(self.df.columns)} columns")

            if not params.get("numerical_distributions"):
                params.pop("numerical_distributions", None)

            if self.method == "ctgan":
                params["enable_gpu"] = gpu_requested

            self.synthesizer = sdv["classes"][self.method](metadata, **params)
        elif self.library == "native":
            if gpu_capable:
                params["device"] = "cuda" if gpu_requested else "cpu"

            native = _load_native()
            self.synthesizer = native[self.method](**params)
        else:
            if gpu_capable:
                params["device"] = "cuda" if gpu_requested else "cpu"

            synthcity = _load_synthcity()
            self.synthesizer = synthcity["plugins"].get(
                self.method,
                workspace=self.synthesizers_dir / ".synthcity_workspace",
                **params,
            )

        logger.debug(f"[{self.name}] Fitting on {self.n_rows_original} rows...")
        self.synthesizer.fit(self.df)
        logger.debug(f"[{self.name}] Fitting complete")

    def sample(self, num_rows: Optional[Union[int, str]] = None, synth_scale: int = 100) -> pd.DataFrame:
        """
        Sample synthetic data from the fitted synthesizer.

        Args:
            num_rows: Number of rows to generate. If None, uses value from config.
                      Use "same" to match the size of the original dataset.

        Returns:
            DataFrame of synthetic data
        """
        if self.synthesizer is None:
            raise RuntimeError("Call fit() before sample()")

        num_rows = num_rows or self.synth_cfg.get("num_rows", "same")
        if num_rows == "same":
            num_rows = self.n_rows_original

        logger.debug(f"[{self.name}] Sampling {num_rows} rows...")
        if self.library == "sdv":
            synthetic_df = self.synthesizer.sample(num_rows=num_rows)
        else:
            synthetic_df = self.synthesizer.generate(count=num_rows).dataframe()
        logger.debug(f"[{self.name}] Sampling complete")

        synthetic_df = self._restore_dtypes(synthetic_df)
        self._save_synthetic(synthetic_df, synth_scale=synth_scale)
        return synthetic_df

    def save(self) -> None:
        """Save the fitted synthesizer to disk."""
        if self.synthesizer is None:
            raise RuntimeError("Call fit() before save()")

        self.synthesizers_dir.mkdir(parents=True, exist_ok=True)
        path = self.synthesizers_dir / f"{self.name}__{self.dataset_name}.pkl"

        if self.library == "sdv":
            self.synthesizer._n_rows_original = self.n_rows_original
            self.synthesizer.save(str(path))
        elif self.library == "native":
            import pickle
            with open(path, "wb") as f:
                pickle.dump(
                    {"synthesizer": self.synthesizer, "n_rows_original": self.n_rows_original}, f
                )
        else:
            synthcity = _load_synthcity()
            synthcity["save_to_file"](
                path,
                {"synthesizer": self.synthesizer, "n_rows_original": self.n_rows_original},
            )

        logger.debug(f"[{self.name}] Synthesizer saved → {path}")

    @classmethod
    def load(cls, path: str, cfg: str = "config/synthesizers.yaml") -> "Synthesizer":
        """Load a saved Synthesizer from a .pkl file."""
        stem = Path(path).stem.split("__")
        name = stem[0]

        instance = cls.__new__(cls)
        instance.name         = name
        instance.cfg          = load_config(cfg)
        instance.synth_cfg    = instance.cfg.get("methods", {}).get(name, {})
        instance.method       = name
        instance.library      = SUPPORTED_SYNTHESIZERS[name]
        output_cfg            = instance.cfg.get("output", {})
        instance.synthetic_dir    = Path(output_cfg.get("synthetic_dir", "data/synthetic"))
        instance.synthesizers_dir = Path(output_cfg.get("synthesizers_dir", "synthesizers"))
        instance.dataset_name     = stem[1] if len(stem) > 1 else None

        if instance.library == "sdv":
            sdv = _load_sdv()
            instance.synthesizer     = sdv["classes"][name].load(str(path))
            instance.n_rows_original = getattr(instance.synthesizer, '_n_rows_original', None)
        elif instance.library == "native":
            import pickle
            with open(path, "rb") as f:
                payload = pickle.load(f)
            instance.synthesizer     = payload["synthesizer"]
            instance.n_rows_original = payload["n_rows_original"]
        else:
            synthcity = _load_synthcity()
            payload = synthcity["load_from_file"](path)
            instance.synthesizer     = payload["synthesizer"]
            instance.n_rows_original = payload["n_rows_original"]

        logger.debug(f"[{name}] Loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _restore_dtypes(self, synthetic_df: pd.DataFrame) -> pd.DataFrame:
        """Round-cast synthetic columns back to integer dtype where the training data was integer."""
        if self.df is None:
            return synthetic_df
        result = synthetic_df.copy()
        for col in result.columns:
            if col not in self.df.columns:
                continue
            orig_dtype = self.df[col].dtype
            if pd.api.types.is_integer_dtype(orig_dtype):
                try:
                    col_min = int(self.df[col].min())
                    col_max = int(self.df[col].max())
                    numeric = pd.to_numeric(result[col], errors="coerce")
                    result[col] = numeric.round().clip(col_min, col_max).astype(orig_dtype)
                except (TypeError, ValueError, OverflowError):
                    result[col] = numeric
                    logger.warning(f"[{self.name}] Could not restore integer dtype for column '{col}'")
        return result

    def _save_synthetic(self, df: pd.DataFrame, synth_scale: int = 100) -> None:
        nan_rows = int(df.isna().any(axis=1).sum())
        if nan_rows > 0:
            logger.warning(f"[{self.name}] Dropping {nan_rows} NaN rows from synthetic {self.dataset_name} data")
            df = df.dropna()
        if df.empty:
            raise ValueError(
                f"[{self.name}] Synthetic data for '{self.dataset_name}' is empty after dropping NaN rows — "
                "synthesizer produced no usable samples"
            )
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        path = self.synthetic_dir / f"{self.name}_{synth_scale}__{self.dataset_name}.csv"
        df.to_csv(path, index=False)
        logger.debug(f"[{self.name}] Synthetic data saved → {path}")