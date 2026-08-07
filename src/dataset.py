# src/dataset.py
"""Dataset class for loading, preprocessing, and saving datasets."""

import logging
import pandas as pd
import numpy as np

from src.utils import atomic_path
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

logger = logging.getLogger(__name__)


def _scale(scaler, df, cols, tr):
    """Fit the scaler on the training rows, apply it to every row."""
    scaler.fit(df.loc[tr, cols])
    return pd.DataFrame(scaler.transform(df[cols]), columns=cols, index=df.index)


# Every step takes (df, cols, tr) where `tr` indexes the training rows: constants
# are estimated from df.loc[tr] only and then applied to the whole column. See
# Dataset.preprocess() for how `tr` is kept identical to the split the models draw.
NUMERIC_STEPS = {
    "impute_median":  lambda df, cols, tr: df[cols].fillna(df.loc[tr, cols].median()),
    "impute_mean":    lambda df, cols, tr: df[cols].fillna(df.loc[tr, cols].mean()),
    "standard_scale": lambda df, cols, tr: _scale(StandardScaler(), df, cols, tr),
    "minmax_scale":   lambda df, cols, tr: _scale(MinMaxScaler(), df, cols, tr),
}

CATEGORICAL_STEPS = {
    "impute_mode":   lambda df, cols, tr: df[cols].fillna(df.loc[tr, cols].mode().iloc[0]),
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

    def __init__(self, name: str, cfg: str, models_cfg: str = "config/models.yaml"):
        """
        Args:
            name:       Dataset name — used to find config entry and resolve file paths
            cfg:        Path to datasets.yaml
            models_cfg: Path to models.yaml. Read for the train/test split parameters
                        only. preprocess() has to draw the very same split that
                        Model.split()/FHEModel.split()/Synthesizer.load_data() will
                        draw later, so those parameters live in exactly one file and
                        every splitter reads them from there.
        """
        from src.utils import load_config

        all_cfg = load_config(cfg)
        if name not in all_cfg:
            raise KeyError(f"Dataset '{name}' not found in config. Available: {list(all_cfg)}")

        self.name     = name
        self.cfg      = all_cfg[name]
        self.features: list = self.cfg.get("features") or []
        self.target:   str  = self.cfg.get("target")

        split_cfg      = load_config(models_cfg)
        self.test_size = split_cfg.get("test_size", 0.2)
        self.seed      = split_cfg.get("random_seed", 42)
        self.stratify  = split_cfg.get("stratify", False)

        raw_cfg            = self.cfg.get("raw_path")
        self.raw_path      = Path(raw_cfg) if raw_cfg else self.RAW_DIR / f"{self.name}.csv"
        processed_cfg      = self.cfg.get("processed_path")
        self.processed_path = Path(processed_cfg) if processed_cfg else self.PROCESSED_DIR / f"{self.name}.csv"

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
        path = self.raw_path
        logger.info(f"[{self.name}] Loading from {path}")
        df = pd.read_csv(path)

        all_cols = self.features + ([self.target] if self.target else [])
        if all_cols and len(all_cols) == len(df.columns):
            df.columns = all_cols

        # Keep only the declared features and target.
        #
        # Nothing downstream filters on `features`: Model/FHEModel build their
        # design matrix as df.drop(columns=[target]), so *any* surplus column in
        # the raw CSV silently becomes a predictor. Both pregnancy_outcome and
        # gestational_diabetes ship a row-identifier column that is not declared
        # here, is therefore never scaled or imputed, and — because the rows are
        # ordered by class — predicts the target almost perfectly on its own
        # (PatientID alone: test ROC-AUC 1.00). Dropping undeclared columns at
        # load time makes `features` authoritative.
        #
        # Done before preprocess() so that columns derived later (e.g. one-hot
        # expansions, which are not listed in `features`) are not dropped.
        if all_cols:
            missing = [c for c in all_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    f"[{self.name}] Columns declared in config but absent from {path}: "
                    f"{missing}. Available: {list(df.columns)}"
                )
            extra = [c for c in df.columns if c not in all_cols]
            if extra:
                logger.info(
                    f"[{self.name}] Dropping {len(extra)} undeclared column(s): {extra}"
                )
                df = df[all_cols]

        df.replace("?", np.nan, inplace=True)
        logger.info(f"[{self.name}] Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps from config, fitting every constant on the
        training partition only.

        The processed file is written whole and split downstream, so this method
        reproduces the split rather than storing it: train_test_split derives one
        index permutation from (n_samples, test_size, random_state, stratify
        labels) and applies it to whatever arrays it is handed — feature values
        never enter the draw. Model.split(), FHEModel.split() and
        Synthesizer.load_data() call it on this same row count with these same
        target labels and parameters, so `train_idx` below is exactly the set of
        rows they will call train.

        Two things this relies on, both enforced here: the target is binarized
        before the draw, so the stratify labels match what the splitters see; and
        no step reorders or drops rows, only columns.
        """
        df   = df.copy()
        prep = self.cfg.get("preprocessing", {})

        # --- Binarize target (must precede the draw: it defines the stratify labels) ---
        df = self._binarize_target(df, prep.get("binarize_target", {}))
        df[self.target] = df[self.target].astype(int)

        # --- Reproduce the downstream train/test split ---
        # Split the target Series and keep its index: the returned labels are the
        # row labels the downstream splitters will hand to their own train sets.
        y = df[self.target]
        y_train, y_test = train_test_split(
            y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y if self.stratify else None,
        )
        train_idx, test_idx = y_train.index, y_test.index
        logger.info(
            f"[{self.name}] Fitting preprocessing on {len(train_idx)} train rows "
            f"({len(test_idx)} held out, seed={self.seed}, stratify={self.stratify})"
        )

        # --- Numeric ---
        num_cfg  = prep.get("numeric", {})
        num_cols = [c for c in (num_cfg.get("columns") or []) if c in df.columns]
        if num_cols:
            df[num_cols] = df[num_cols].astype(float)
            for step in (num_cfg.get("steps") or []):
                if step not in NUMERIC_STEPS:
                    raise ValueError(f"Unknown numeric step: '{step}'")
                df[num_cols] = NUMERIC_STEPS[step](df, num_cols, train_idx)
                logger.info(f"[{self.name}] [numeric] {step} → {num_cols}")

        # --- Categorical ---
        cat_cfg  = prep.get("categorical", {})
        cat_cols = [c for c in (cat_cfg.get("columns") or []) if c in df.columns]
        if cat_cols:
            for step in (cat_cfg.get("steps") or []):
                if step == "onehot_encode":
                    df = self._onehot(df, cat_cols, train_idx)
                    logger.info(f"[{self.name}] [categorical] onehot_encode → {cat_cols}")
                elif step in CATEGORICAL_STEPS:
                    df[cat_cols] = CATEGORICAL_STEPS[step](df, cat_cols, train_idx)
                    logger.info(f"[{self.name}] [categorical] {step} → {cat_cols}")
                else:
                    raise ValueError(f"Unknown categorical step: '{step}'")

        return df

    def save(self, df: pd.DataFrame) -> None:
        """Save processed DataFrame to the resolved processed_path.

        Written atomically: every concurrently running job rewrites this same
        path during its preprocessing stage, and a reader must never see a
        half-written file.
        """
        with atomic_path(self.processed_path) as tmp:
            df.to_csv(tmp, index=False)
        logger.info(f"[{self.name}] Saved → {self.processed_path}")

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

    def _onehot(self, df: pd.DataFrame, cols: list, tr) -> pd.DataFrame:
        # Categories are learned from the training rows only; handle_unknown="ignore"
        # encodes a level that appears solely in the test rows as an all-zero block.
        enc     = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc.fit(df.loc[tr, cols])
        encoded = enc.transform(df[cols])
        enc_df  = pd.DataFrame(
            encoded, columns=enc.get_feature_names_out(cols), index=df.index
        )
        return pd.concat([df.drop(columns=cols), enc_df], axis=1)