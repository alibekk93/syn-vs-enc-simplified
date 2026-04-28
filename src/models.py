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

    def __init__(self, name: str, cfg: str = "config/models.yaml"):
        """
        Args:
            name: Model name — must be a key in SUPPORTED_MODELS
            cfg:  Path to models.yaml
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

        output_cfg       = all_cfg.get("output", {})
        self.results_dir = Path(output_cfg.get("results_dir", "results"))
        self.models_dir  = Path(output_cfg.get("models_dir", "models"))

        hyperparams = self.model_cfg.get("hyperparameters") or {}
        self.model  = SUPPORTED_MODELS[name](hyperparams)

        # Data placeholders
        self.df:           Optional[pd.DataFrame] = None
        self.target:       Optional[str]          = None
        self.dataset_name: Optional[str]          = None
        self.X_train:      Optional[pd.DataFrame] = None
        self.X_test:       Optional[pd.DataFrame] = None
        self.y_train:      Optional[pd.Series]    = None
        self.y_test:       Optional[pd.Series]    = None

        logger.info(f"[{self.name}] Initialized with hyperparameters: {hyperparams}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load a saved Model instance from a .joblib file."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a Model instance, got {type(model)}")
        logger.info(f"[{model.name}] Loaded from {path}")
        return model

    def run(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> dict:
        """Full pipeline: load → split → train → evaluate."""
        self.load_data(dataset_name, dataset_cfg)
        self.split()
        self.train()
        return self.evaluate()

    def load_data(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> None:
        """Load processed dataset and resolve target column from datasets.yaml."""
        path = self.PROCESSED_DIR / f"{dataset_name}.csv"
        logger.info(f"[{self.name}] Loading data from {path}")
        self.df = pd.read_csv(path)

        ds_cfg = load_config(dataset_cfg)
        if dataset_name not in ds_cfg:
            raise KeyError(f"Dataset '{dataset_name}' not found in {dataset_cfg}")
        self.target       = ds_cfg[dataset_name]["target"]
        self.dataset_name = dataset_name

        logger.info(f"[{self.name}] Loaded {len(self.df)} rows, target='{self.target}'")

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

        logger.info(f"[{self.name}] Split → train={len(self.X_train)}, test={len(self.X_test)}")

    def train(self) -> None:
        """Fit the model on the training set."""
        if self.X_train is None:
            raise RuntimeError("Call split() before train()")

        logger.info(f"[{self.name}] Training...")
        self.model.fit(self.X_train, self.y_train)
        logger.info(f"[{self.name}] Training complete")
        self._save_model()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions for X."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Return probability estimates for X, or None if not supported."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return None

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

        logger.info(f"[{self.name}] Evaluation ({on}): {results}")
        self._save_results(results, on)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        path = self.models_dir / f"{self.name}__{self.dataset_name}.joblib"
        joblib.dump(self, path)
        logger.info(f"[{self.name}] Model saved → {path}")

    def _save_results(self, results: dict, split: str) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / f"{self.name}__{self.dataset_name}__{split}__metrics.json"
        with open(path, "w") as f:
            json.dump({"model": self.name, "dataset": self.dataset_name, "split": split, "metrics": results}, f, indent=2)
        logger.info(f"[{self.name}] Results saved → {path}")