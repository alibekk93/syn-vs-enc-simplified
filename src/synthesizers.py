"""Synthesizer class for fitting and sampling synthetic tabular data."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata

from src.utils import load_config

logger = logging.getLogger(__name__)

SUPPORTED_SYNTHESIZERS = {
    "gaussian_copula": GaussianCopulaSynthesizer,
}


class Synthesizer:
    """
    Wraps an SDV synthesizer with fit/sample/save functionality,
    driven by config/synthesizers.yaml.

    Usage:
        synth = Synthesizer("gaussian_copula", cfg="config/synthesizers.yaml")
        synth.fit("heart_disease")
        synthetic_df = synth.sample()
        synth.save()

        # or all at once:
        synthetic_df = synth.run("heart_disease")

        # reload later:
        synth = Synthesizer.load("synthesizers/gaussian_copula__heart_disease.pkl")
        synthetic_df = synth.sample()
    """

    PROCESSED_DIR = Path("data/processed")

    def __init__(self, name: str, cfg: str = "config/synthesizers.yaml"):
        """
        Args:
            name: Synthesizer name — must be a key in config and SUPPORTED_SYNTHESIZERS
            cfg:  Path to synthesizers.yaml
        """
        all_cfg = load_config(cfg)

        if name not in all_cfg:
            available = [k for k in all_cfg if k != "output"]
            raise KeyError(f"Synthesizer '{name}' not found in config. Available: {available}")

        self.name      = name
        self.cfg       = all_cfg
        self.synth_cfg = all_cfg[name]
        self.method    = self.synth_cfg["method"]

        if self.method not in SUPPORTED_SYNTHESIZERS:
            raise KeyError(f"Method '{self.method}' is not supported. Supported: {list(SUPPORTED_SYNTHESIZERS)}")

        output_cfg            = all_cfg.get("output", {})
        self.synthetic_dir    = Path(output_cfg.get("synthetic_dir", "data/synthetic"))
        self.synthesizers_dir = Path(output_cfg.get("synthesizers_dir", "synthesizers"))

        # Placeholders — populated in fit()
        self.dataset_name: Optional[str]    = None
        self.n_rows_original: Optional[int] = None
        self.synthesizer: Optional[object]  = None

        logger.info(f"[{self.name}] Initialized with method='{self.method}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, dataset_name: str) -> pd.DataFrame:
        """Full pipeline: fit → sample → save."""
        self.fit(dataset_name)
        synthetic_df = self.sample()
        self.save()
        return synthetic_df

    def fit(self, dataset_name: str) -> None:
        """Fit the synthesizer on a processed dataset."""
        path = self.PROCESSED_DIR / f"{dataset_name}.csv"
        logger.info(f"[{self.name}] Loading data from {path}")
        df = pd.read_csv(path)

        self.dataset_name    = dataset_name
        self.n_rows_original = len(df)

        metadata = Metadata.detect_from_dataframe(df)
        logger.info(f"[{self.name}] Metadata detected for {len(df.columns)} columns")

        params = self.synth_cfg.get("parameters") or {}
        if not params.get("numerical_distributions"):
            params.pop("numerical_distributions", None)

        self.synthesizer = SUPPORTED_SYNTHESIZERS[self.method](metadata, **params)

        logger.info(f"[{self.name}] Fitting on {self.n_rows_original} rows...")
        self.synthesizer.fit(df)
        logger.info(f"[{self.name}] Fitting complete")

    def sample(self, num_rows: Optional[Union[int, str]] = None) -> pd.DataFrame:
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

        logger.info(f"[{self.name}] Sampling {num_rows} rows...")
        synthetic_df = self.synthesizer.sample(num_rows=num_rows)
        logger.info(f"[{self.name}] Sampling complete")

        self._save_synthetic(synthetic_df)
        return synthetic_df

    def save(self) -> None:
        """Save the fitted synthesizer to disk."""
        if self.synthesizer is None:
            raise RuntimeError("Call fit() before save()")

        self.synthesizers_dir.mkdir(parents=True, exist_ok=True)
        path = self.synthesizers_dir / f"{self.name}__{self.dataset_name}.pkl"
        self.synthesizer.save(str(path))
        logger.info(f"[{self.name}] Synthesizer saved → {path}")

    @classmethod
    def load(cls, path: str, cfg: str = "config/synthesizers.yaml") -> "Synthesizer":
        """Load a saved Synthesizer from a .pkl file."""
        stem = Path(path).stem.split("__")
        name = stem[0]

        instance = cls.__new__(cls)
        instance.name         = name
        instance.cfg          = load_config(cfg)
        instance.synth_cfg    = instance.cfg.get(name, {})
        instance.method       = instance.synth_cfg.get("method", name)
        output_cfg            = instance.cfg.get("output", {})
        instance.synthetic_dir    = Path(output_cfg.get("synthetic_dir", "data/synthetic"))
        instance.synthesizers_dir = Path(output_cfg.get("synthesizers_dir", "synthesizers"))
        instance.dataset_name     = stem[1] if len(stem) > 1 else None
        instance.n_rows_original  = None  # not stored in .pkl — pass num_rows to sample() explicitly if needed
        instance.synthesizer      = SUPPORTED_SYNTHESIZERS[instance.method].load(str(path))

        logger.info(f"[{name}] Loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_synthetic(self, df: pd.DataFrame) -> None:
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        path = self.synthetic_dir / f"{self.name}__{self.dataset_name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"[{self.name}] Synthetic data saved → {path}")