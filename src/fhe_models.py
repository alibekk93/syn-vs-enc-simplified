"""FHE Model class using Concrete ML for privacy-preserving inference."""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from concrete.ml.sklearn import (
    LogisticRegression,
    RandomForestClassifier,
    XGBClassifier,
    NeuralNetClassifier,
)
from concrete.ml.deployment import FHEModelDev
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from src.utils import load_config

logger = logging.getLogger(__name__)

# Concrete ML equivalents of the models in models.yaml.
# Note: mlp uses NeuralNetClassifier (Concrete ML's QAT-based FCNN, not sklearn MLP)
SUPPORTED_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest":       RandomForestClassifier,
    "xgboost":             XGBClassifier,
    "mlp":                 NeuralNetClassifier,
}

# sklearn-only params to strip before passing to Concrete ML constructors
_SKLEARN_ONLY = {"hidden_layer_sizes", "activation", "max_iter", "random_state"}

SUPPORTED_METRICS = {
    "accuracy":  lambda y, yp, yprob: accuracy_score(y, yp),
    "precision": lambda y, yp, yprob: precision_score(y, yp, zero_division=0),
    "recall":    lambda y, yp, yprob: recall_score(y, yp, zero_division=0),
    "f1":        lambda y, yp, yprob: f1_score(y, yp, zero_division=0),
    "roc_auc":   lambda y, yp, yprob: roc_auc_score(y, yprob) if yprob is not None else float("nan"),
}


class FHEModel:
    """
    Wraps a Concrete ML model with train/compile/predict/evaluate functionality,
    reusing the same config/models.yaml as the standard Model class.

    The pipeline adds a compile() step after fit(), which is required by
    Concrete ML before FHE inference can be performed.

    Usage:
        fhe = FHEModel("logistic_regression", cfg="config/models.yaml")
        fhe.load_data("heart_disease")
        fhe.split()
        fhe.train()       # fits and compiles the FHE circuit
        metrics = fhe.evaluate()

        # or all at once:
        metrics = fhe.run("heart_disease")
    """

    PROCESSED_DIR = Path("data/processed")
    FHE_MODELS_DIR = Path("models/fhe")

    def __init__(self, name: str, cfg: str = "config/models.yaml"):
        """
        Args:
            name: Model name — must be a key in SUPPORTED_MODELS and models.yaml
            cfg:  Path to models.yaml (shared with standard Model class)
        """
        all_cfg    = load_config(cfg)
        model_cfgs = {m["name"]: m for m in all_cfg.get("models", [])}

        if name not in model_cfgs:
            raise KeyError(f"Model '{name}' not found in config. Available: {list(model_cfgs)}")
        if name not in SUPPORTED_MODELS:
            raise KeyError(f"Model '{name}' is not supported for FHE. Supported: {list(SUPPORTED_MODELS)}")

        self.name      = name
        self.cfg       = all_cfg
        self.model_cfg = model_cfgs[name]

        hyperparams = self._prepare_hyperparams(name, self.model_cfg.get("hyperparameters") or {})
        self.model  = SUPPORTED_MODELS[name](**hyperparams)

        # Data placeholders
        self.df:           Optional[pd.DataFrame] = None
        self.target:       Optional[str]          = None
        self.dataset_name: Optional[str]          = None
        self.X_train:      Optional[np.ndarray]   = None
        self.X_test:       Optional[np.ndarray]   = None
        self.y_train:      Optional[np.ndarray]   = None
        self.y_test:       Optional[np.ndarray]   = None

        logger.info(f"[FHE:{self.name}] Initialized with hyperparameters: {hyperparams}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> dict:
        """Full pipeline: load → split → train (fit + compile) → evaluate."""
        self.load_data(dataset_name, dataset_cfg)
        self.split()
        self.train()
        return self.evaluate()

    def load_data(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> None:
        """Load processed dataset and resolve target column from datasets.yaml."""
        path = self.PROCESSED_DIR / f"{dataset_name}.csv"
        logger.info(f"[FHE:{self.name}] Loading data from {path}")
        self.df = pd.read_csv(path)

        ds_cfg = load_config(dataset_cfg)
        if dataset_name not in ds_cfg:
            raise KeyError(f"Dataset '{dataset_name}' not found in {dataset_cfg}")
        self.target       = ds_cfg[dataset_name]["target"]
        self.dataset_name = dataset_name

        logger.info(f"[FHE:{self.name}] Loaded {len(self.df)} rows, target='{self.target}'")

    def split(self) -> None:
        """Split loaded data into train and test sets, returned as numpy arrays."""
        if self.df is None or self.target is None:
            raise RuntimeError("Call load_data() before split()")

        test_size = self.cfg.get("test_size", 0.2)
        seed      = self.cfg.get("random_seed", 42)
        stratify  = self.cfg.get("stratify", False)

        X = self.df.drop(columns=[self.target]).values
        y = self.df[self.target].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
            stratify=y if stratify else None
        )

        logger.info(f"[FHE:{self.name}] Split → train={len(self.X_train)}, test={len(self.X_test)}")

    def train(self) -> None:
        """Fit and compile the FHE model, then save it."""
        if self.X_train is None:
            raise RuntimeError("Call split() before train()")

        logger.info(f"[FHE:{self.name}] Fitting...")
        self.model.fit(self.X_train, self.y_train)

        logger.info(f"[FHE:{self.name}] Compiling FHE circuit...")
        self.model.compile(self.X_train)
        logger.info(f"[FHE:{self.name}] Compilation complete")

        self._save_model()

    def predict(self, X: np.ndarray, fhe: str = "simulate") -> np.ndarray:
        """
        Return class predictions.

        Args:
            X:   Input array
            fhe: One of 'disable' (quantized clear), 'simulate' (FHE simulation),
                 'execute' (real FHE — slow)
        """
        return self.model.predict(X, fhe=fhe)

    def predict_proba(self, X: np.ndarray, fhe: str = "simulate") -> Optional[np.ndarray]:
        """Return probability estimates, or None if not supported by this model."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X, fhe=fhe)[:, 1]
        return None

    def evaluate(self, on: str = "test", fhe: str = "simulate") -> dict:
        """
        Compute metrics on the specified split.

        Args:
            on:  One of 'train', 'test'
            fhe: Inference mode — 'disable', 'simulate', or 'execute'

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

        y_pred  = self.predict(X, fhe=fhe)
        y_proba = self.predict_proba(X, fhe=fhe)

        metric_names = self.cfg.get("metrics") or list(SUPPORTED_METRICS)
        results = {}
        for metric in metric_names:
            if metric not in SUPPORTED_METRICS:
                logger.warning(f"[FHE:{self.name}] Unknown metric '{metric}', skipping")
                continue
            results[metric] = round(SUPPORTED_METRICS[metric](y, y_pred, y_proba), 4)

        logger.info(f"[FHE:{self.name}] Evaluation ({on}, fhe={fhe}): {results}")
        self._save_results(results, on, fhe)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare_hyperparams(self, name: str, hyperparams: dict) -> dict:
        """
        Adapt hyperparams from models.yaml for Concrete ML constructors.
        Strips sklearn-only params and adds n_bits for quantization.
        """
        params = hyperparams.copy()

        # All Concrete ML models accept n_bits for quantization precision
        params.setdefault("n_bits", 8)

        if name == "mlp":
            # Strip sklearn-only params — Concrete ML uses module__* equivalents
            for key in _SKLEARN_ONLY:
                params.pop(key, None)

        return params

    def _save_model(self) -> None:
        """Save compiled FHE model using FHEModelDev (produces client/server files)."""
        out_dir = self.FHE_MODELS_DIR / f"{self.name}__{self.dataset_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        dev = FHEModelDev(path_dir=str(out_dir), model=self.model)
        dev.save()
        logger.info(f"[FHE:{self.name}] Model saved → {out_dir}")

    def _save_results(self, results: dict, split: str, fhe: str) -> None:
        results_dir = Path(self.cfg.get("output", {}).get("results_dir", "results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"fhe__{self.name}__{self.dataset_name}__{split}__metrics.json"
        with open(path, "w") as f:
            json.dump({
                "model":   self.name,
                "dataset": self.dataset_name,
                "split":   split,
                "fhe":     fhe,
                "metrics": results
            }, f, indent=2)
        logger.info(f"[FHE:{self.name}] Results saved → {path}")
