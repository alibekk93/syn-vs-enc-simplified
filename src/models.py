# src/models.py
"""Model class for initializing, training, evaluating and predicting."""

import logging
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from src.utils import load_config

logger = logging.getLogger(__name__)


def _build_xgboost(hyperparams: dict):
    """Import and instantiate XGBClassifier separately to keep the
    dependency optional — code runs fine without xgboost installed as
    long as the model is not used."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    return XGBClassifier(**hyperparams)


SUPPORTED_MODELS = {
    "logistic_regression": lambda hp: LogisticRegression(**hp),
    "random_forest":       lambda hp: RandomForestClassifier(**hp),
    "xgboost":             lambda hp: _build_xgboost(hp),
    "svm":                 lambda hp: SVC(**hp),
    "mlp":                 lambda hp: MLPClassifier(**{
                               **{k: v for k, v in hp.items() if k != "hidden_layer_sizes"},
                               "hidden_layer_sizes": tuple(hp.get("hidden_layer_sizes", [100])),
                           }),
}

# Of SUPPORTED_MODELS, only xgboost has a GPU path (`device="cuda"`).
# logistic_regression/random_forest/svm/mlp are plain scikit-learn — CPU-only,
# no GPU acceleration exists for them.
GPU_CAPABLE_MODELS = {"xgboost"}

SUPPORTED_METRICS = {
    "accuracy":  lambda y, yp, yprob: accuracy_score(y, yp),
    "precision": lambda y, yp, yprob: precision_score(y, yp, zero_division=0),
    "recall":    lambda y, yp, yprob: recall_score(y, yp, zero_division=0),
    "f1":        lambda y, yp, yprob: f1_score(y, yp, zero_division=0),
    "roc_auc":   lambda y, yp, yprob: roc_auc_score(y, yprob) if yprob is not None else float("nan"),
}


class Model:
    """
    Wraps a single sklearn-compatible model with train/predict/evaluate
    functionality, driven by config/models.yaml.

    Usage:
        model = Model("random_forest", cfg="config/models.yaml")
        model.load_data("heart_disease")
        model.split()
        model.train()
        metrics = model.evaluate()

        # or all at once:
        metrics = model.run("heart_disease")
    """

    PROCESSED_DIR = Path("data/processed")

    def __init__(self, name: str, cfg: str = "config/models.yaml", mode: str = "standard", device: str | None = None):
        """
        Args:
            name:   Model name — must be a key in SUPPORTED_MODELS
            cfg:    Path to models.yaml
            mode:   Mode of operation — 'standard', 'synthesized', etc. Used for saving.
            device: "cpu" or "cuda" (default: models.yaml's `device`, normally cpu).
                    Only applied for GPU_CAPABLE_MODELS (currently xgboost) —
                    other model types have no GPU path and ignore it.
        """
        all_cfg    = load_config(cfg)
        model_cfgs = {m["name"]: m for m in all_cfg.get("models", [])}

        if name not in model_cfgs:
            raise KeyError(f"Model '{name}' not found in config. Available: {list(model_cfgs)}")
        if name not in SUPPORTED_MODELS:
            raise KeyError(f"Model '{name}' is not supported. Supported: {list(SUPPORTED_MODELS)}")

        self.name      = name
        self.cfg       = all_cfg
        self.model_cfg = model_cfgs[name]
        self.mode      = mode
        self.device    = device or all_cfg.get("device", "cpu")

        output_cfg       = all_cfg.get("output", {})
        self.results_dir = Path(output_cfg.get("results_dir", "results"))
        self.models_dir  = Path(output_cfg.get("models_dir", "models"))

        hyperparams = self.model_cfg.get("hyperparameters") or {}
        if self.device == "cuda":
            if name in GPU_CAPABLE_MODELS:
                hyperparams = {**hyperparams, "device": "cuda"}
            else:
                logger.info(f"[{name}] device='cuda' requested but this model has no GPU support — running on CPU")
        self.model  = SUPPORTED_MODELS[name](hyperparams)

        # Data placeholders
        self.df:           Optional[pd.DataFrame] = None
        self.target:       Optional[str]          = None
        self.dataset_name: Optional[str]          = None
        self.X_train:      Optional[pd.DataFrame] = None
        self.X_test:       Optional[pd.DataFrame] = None
        self.y_train:      Optional[pd.Series]    = None
        self.y_test:       Optional[pd.Series]    = None
        # For saving: if set, overrides self.dataset_name in the saved filename/path
        self.save_dataset_name: Optional[str]     = None

        logger.debug(f"[{self.name}] Initialized with hyperparameters: {hyperparams} (mode={self.mode})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load a saved Model instance from a .joblib file."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a Model instance, got {type(model)}")
        logger.debug(f"[{model.name}] Loaded from {path}")
        return model

    def run(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> dict:
        """Full pipeline: load → split → train → evaluate."""
        self.load_data(dataset_name, dataset_cfg)
        self.split()
        self.train()
        return self.evaluate()

    def load_data(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> None:
        """Load processed dataset and resolve target column from datasets.yaml.
        For synthetic datasets (named '<synth>__<original>'), uses the original dataset's config.
        processed_path from the config entry overrides the default data/processed/ location.
        """
        ds_cfg = load_config(dataset_cfg)

        if dataset_name in ds_cfg:
            entry = ds_cfg[dataset_name]
            processed_path = entry.get("processed_path")
            path = Path(processed_path) if processed_path else self.PROCESSED_DIR / f"{dataset_name}.csv"
            self.target       = entry["target"]
            self.dataset_name = dataset_name
        else:
            # Try to interpret as a synthetic dataset: <synthesizer>__<original_dataset>
            parts = dataset_name.split('__')
            if len(parts) == 2:
                base_dataset_name = parts[1]
                if base_dataset_name in ds_cfg:
                    entry = ds_cfg[base_dataset_name]
                    processed_path = entry.get("processed_path")
                    if processed_path:
                        path = Path(processed_path).parent / f"{dataset_name}.csv"
                    else:
                        path = self.PROCESSED_DIR / f"{dataset_name}.csv"
                    self.target       = entry["target"]
                    self.dataset_name = dataset_name
                else:
                    raise KeyError(f"Dataset '{dataset_name}' not found in {dataset_cfg} and base dataset '{base_dataset_name}' not found either.")
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in {dataset_cfg}")

        logger.debug(f"[{self.name}] Loading data from {path}")
        self.df = pd.read_csv(path)
        logger.debug(f"[{self.name}] Loaded {len(self.df)} rows, target='{self.target}'")

    def split(self) -> None:
        """Split loaded data into train and test sets."""
        if self.df is None or self.target is None:
            raise RuntimeError("Call load_data() before split()")

        test_size = self.cfg.get("test_size", 0.2)
        seed      = self.cfg.get("random_seed", 42)
        stratify  = self.cfg.get("stratify", False)

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
            stratify=y if stratify else None
        )

        logger.debug(f"[{self.name}] Split → train={len(self.X_train)}, test={len(self.X_test)}")

    def use_all_as_train(self) -> None:
        """Use the entire loaded dataset as the training set (no test split)."""
        if self.df is None or self.target is None:
            raise RuntimeError("Call load_data() before use_all_as_train()")
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.X_train = X.reset_index(drop=True)
        self.y_train = y.reset_index(drop=True)
        logger.debug(f"[{self.name}] Using all {len(self.X_train)} rows as training set")

    def load_test_data(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> None:
        """Load real processed dataset and set its test split, using the same split params as split()."""
        ds_cfg = load_config(dataset_cfg)
        entry = ds_cfg[dataset_name]
        processed_path = entry.get("processed_path")
        path = Path(processed_path) if processed_path else self.PROCESSED_DIR / f"{dataset_name}.csv"
        df = pd.read_csv(path)
        target = entry["target"]

        test_size = self.cfg.get("test_size", 0.2)
        seed      = self.cfg.get("random_seed", 42)
        stratify  = self.cfg.get("stratify", False)

        X = df.drop(columns=[target])
        y = df[target]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
            stratify=y if stratify else None
        )

        self.X_test = X_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        logger.debug(f"[{self.name}] Loaded real test split: {len(self.X_test)} rows from {path}")

    def train(self) -> None:
        """Fit the model on the training set."""
        if self.X_train is None:
            raise RuntimeError("Call split() before train()")

        logger.debug(f"[{self.name}] Training...")
        self.model.fit(self._to_cuda(self.X_train), self.y_train)
        logger.debug(f"[{self.name}] Training complete")
        self._save_model()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions for X."""
        return self.model.predict(self._to_cuda(X))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Return probability estimates for X, or None if not supported."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(self._to_cuda(X))[:, 1]
        return None

    def _to_cuda(self, X: pd.DataFrame):
        """Convert DataFrame to a cupy array for GPU XGBoost; pass through for all other cases.

        Handing XGBoost's sklearn fit()/predict() a plain pandas DataFrame
        silently falls back to CPU training even with device='cuda' set (see
        commit "xgboost gpu fix"). cupy is xgboost's one reliably-supported
        GPU array type for this path — a torch.Tensor was tried instead (to
        avoid the extra dependency) but xgboost's data dispatch doesn't
        recognize it as GPU-resident and ends up calling tensor.numpy() on a
        CUDA tensor, which raises. Keep cupy here.
        """
        if self.name == "xgboost" and self.device == "cuda":
            import cupy as cp
            return cp.array(X.values)
        return X

    def evaluate(self, on: str = "test") -> dict:
        """
        Compute metrics on the specified split.

        Args:
            on: One of 'train', 'test'

        Returns:
            Dict of metric name → score
        """
        splits = {
            "train": (self.X_train, self.y_train),
            "test":  (self.X_test,  self.y_test),
        }
        if on not in splits:
            raise ValueError(f"'on' must be one of {list(splits)}")

        X, y = splits[on]
        if X is None or y is None:
            raise RuntimeError(f"'{on}' split is not available")

        y_pred  = self.predict(X)
        y_proba = self.predict_proba(X)

        metric_names = self.cfg.get("metrics") or list(SUPPORTED_METRICS)
        results = {}
        for metric in metric_names:
            if metric not in SUPPORTED_METRICS:
                logger.warning(f"[{self.name}] Unknown metric '{metric}', skipping")
                continue
            results[metric] = round(SUPPORTED_METRICS[metric](y, y_pred, y_proba), 4)

        logger.debug(f"[{self.name}] Evaluation ({on}): {results}")
        self._save_results(results, on)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        dataset_to_use = self.save_dataset_name if self.save_dataset_name is not None else self.dataset_name
        path = self.models_dir / f"{self.mode}__{self.name}__{dataset_to_use}.joblib"
        joblib.dump(self, path)
        logger.debug(f"[{self.name}] Model saved → {path}")

    def _save_results(self, results: dict, split: str) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        dataset_to_use = self.save_dataset_name if self.save_dataset_name is not None else self.dataset_name
        path = self.results_dir / f"{self.mode}__{self.name}__{dataset_to_use}__{split}__metrics.json"
        with open(path, "w") as f:
            json.dump({"mode": self.mode, "model": self.name, "dataset": dataset_to_use, "split": split, "metrics": results}, f, indent=2)
        logger.debug(f"[{self.name}] Results saved → {path}")