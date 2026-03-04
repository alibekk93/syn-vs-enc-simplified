"""Dataset class for loading, preprocessing, and saving datasets."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

logger = logging.getLogger(__name__)

NUMERIC_STEPS = {
    "impute_median":  lambda df, cols: df[cols].fillna(df[cols].median()),
    "impute_mean":    lambda df, cols: df[cols].fillna(df[cols].mean()),
    "standard_scale": lambda df, cols: pd.DataFrame(
        StandardScaler().fit_transform(df[cols]), columns=cols, index=df.index
    ),
    "minmax_scale":   lambda df, cols: pd.DataFrame(
        MinMaxScaler().fit_transform(df[cols]), columns=cols, index=df.index
    ),
}

CATEGORICAL_STEPS = {
    "impute_mode":   lambda df, cols: df[cols].fillna(df[cols].mode().iloc[0]),
    "onehot_encode": None,  # handled separately — changes column layout
}


class Dataset:
    """
    Handles loading, preprocessing, and saving for a single dataset.

    Usage:
        ds = Dataset("heart_disease", cfg="config/datasets.yaml")
        df = ds.load()        # load raw
        df = ds.preprocess(df)
        ds.save(df)

        # or all at once:
        df = ds.run()
    """

    RAW_DIR       = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")

    def __init__(self, name: str, cfg: str):
        """
        Args:
            name: Dataset name — used to find config entry and resolve file paths
            cfg:  Path to datasets.yaml
        """
        from src.utils import load_config

        all_cfg = load_config(cfg)
        if name not in all_cfg:
            raise KeyError(f"Dataset '{name}' not found in config. Available: {list(all_cfg)}")

        self.name     = name
        self.cfg      = all_cfg[name]
        self.features: list = self.cfg.get("features") or []
        self.target:   str  = self.cfg.get("target")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Load → preprocess → save in one call."""
        df = self.load()
        df = self.preprocess(df)
        self.save(df)
        return df

    def load(self) -> pd.DataFrame:
        """Load raw CSV and assign column names from config."""
        path = self.RAW_DIR / f"{self.name}.csv"
        logger.info(f"[{self.name}] Loading from {path}")
        df = pd.read_csv(path)

        all_cols = self.features + ([self.target] if self.target else [])
        if all_cols and len(all_cols) == len(df.columns):
            df.columns = all_cols

        df.replace("?", np.nan, inplace=True)
        logger.info(f"[{self.name}] Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps from config."""
        df   = df.copy()
        prep = self.cfg.get("preprocessing", {})

        # --- Binarize target ---
        df = self._binarize_target(df, prep.get("binarize_target", {}))

        # --- Numeric ---
        num_cfg  = prep.get("numeric", {})
        num_cols = [c for c in (num_cfg.get("columns") or []) if c in df.columns]
        if num_cols:
            df[num_cols] = df[num_cols].astype(float)
            for step in (num_cfg.get("steps") or []):
                if step not in NUMERIC_STEPS:
                    raise ValueError(f"Unknown numeric step: '{step}'")
                df[num_cols] = NUMERIC_STEPS[step](df, num_cols)
                logger.info(f"[{self.name}] [numeric] {step} → {num_cols}")

        # --- Categorical ---
        cat_cfg  = prep.get("categorical", {})
        cat_cols = [c for c in (cat_cfg.get("columns") or []) if c in df.columns]
        if cat_cols:
            for step in (cat_cfg.get("steps") or []):
                if step == "onehot_encode":
                    df = self._onehot(df, cat_cols)
                    logger.info(f"[{self.name}] [categorical] onehot_encode → {cat_cols}")
                elif step in CATEGORICAL_STEPS:
                    df[cat_cols] = CATEGORICAL_STEPS[step](df, cat_cols)
                    logger.info(f"[{self.name}] [categorical] {step} → {cat_cols}")
                else:
                    raise ValueError(f"Unknown categorical step: '{step}'")

        return df

    def save(self, df: pd.DataFrame) -> None:
        """Save processed DataFrame to data/processed/{name}.csv."""
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = self.PROCESSED_DIR / f"{self.name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"[{self.name}] Saved → {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _binarize_target(self, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """Binarize target column: 0 if value < threshold, 1 if value >= threshold."""
        if not cfg.get("enabled"):
            return df
        if not self.target or self.target not in df.columns:
            logger.warning(f"[{self.name}] Binarize target enabled but '{self.target}' not found")
            return df

        threshold = cfg.get("threshold")
        if threshold is None:
            raise ValueError(f"[{self.name}] binarize_target is enabled but threshold is not set")

        df[self.target] = (df[self.target] >= threshold).astype(int)
        logger.info(f"[{self.name}] Binarized '{self.target}' with threshold={threshold}")
        return df

    def _onehot(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        enc     = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = enc.fit_transform(df[cols])
        enc_df  = pd.DataFrame(
            encoded, columns=enc.get_feature_names_out(cols), index=df.index
        )
        return pd.concat([df.drop(columns=cols), enc_df], axis=1)