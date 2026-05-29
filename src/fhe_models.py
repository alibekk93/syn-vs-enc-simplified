# src/fhe_models.py
"""FHE Model class using Concrete ML for privacy-preserving inference."""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Concrete ML serialization
from concrete.ml.common.serialization.dumpers import dump as cml_dump
from concrete.ml.common.serialization.loaders import load as cml_load

from concrete.ml.sklearn import (
    LogisticRegression,
    RandomForestClassifier,
    XGBClassifier,
    NeuralNetClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from src.utils import load_config

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest":       RandomForestClassifier,
    "xgboost":             XGBClassifier,
    "mlp":                 NeuralNetClassifier,
}

_SKLEARN_ONLY = {"hidden_layer_sizes", "activation", "max_iter", "random_state"}

SUPPORTED_METRICS = {
    "accuracy":  lambda y, yp, yprob: accuracy_score(y, yp),
    "precision": lambda y, yp, yprob: precision_score(y, yp, zero_division=0),
    "recall":    lambda y, yp, yprob: recall_score(y, yp, zero_division=0),
    "f1":        lambda y, yp, yprob: f1_score(y, yp, zero_division=0),
    "roc_auc":   lambda y, yp, yprob: roc_auc_score(y, yprob) if yprob is not None else float("nan"),
}


class FHEModel:

    PROCESSED_DIR = Path("data/processed")

    def __init__(
        self,
        name: str,
        cfg: str = "config/models.yaml",
        mode: str = "fhe",
        fhe_cfg: Optional[dict] = None,
    ):
        all_cfg    = load_config(cfg)
        model_cfgs = {m["name"]: m for m in all_cfg.get("models", [])}

        if name not in model_cfgs:
            raise KeyError(f"Model '{name}' not found in config. Available: {list(model_cfgs)}")
        if name not in SUPPORTED_MODELS:
            raise KeyError(f"Model '{name}' is not supported for FHE. Supported: {list(SUPPORTED_MODELS)}")

        self.name      = name
        self.cfg       = all_cfg
        self.model_cfg = model_cfgs[name]
        self.mode      = mode
        self.fhe_cfg   = fhe_cfg or {}

        output_cfg       = all_cfg.get("output", {})
        self.results_dir = Path(output_cfg.get("results_dir", "results"))
        self.models_dir  = Path(output_cfg.get("models_dir", "models"))

        hyperparams = self._prepare_hyperparams(name, self.model_cfg.get("hyperparameters") or {})
        self.model  = SUPPORTED_MODELS[name](**hyperparams)

        # Data placeholders
        self.df: Optional[pd.DataFrame] = None
        self.target: Optional[str] = None
        self.dataset_name: Optional[str] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.save_dataset_name: Optional[str] = None

        logger.info(f"[FHE:{self.name}] Initialized with hyperparameters: {hyperparams} (mode={self.mode})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> dict:
        self.load_data(dataset_name, dataset_cfg)
        self.split()
        self.train()
        return self.evaluate()

    def load_data(self, dataset_name: str, dataset_cfg: str = "config/datasets.yaml") -> None:
        path = self.PROCESSED_DIR / f"{dataset_name}.csv"
        logger.info(f"[FHE:{self.name}] Loading data from {path}")
        self.df = pd.read_csv(path)

        ds_cfg = load_config(dataset_cfg)
        if dataset_name in ds_cfg:
            self.target = ds_cfg[dataset_name]["target"]
            self.dataset_name = dataset_name
        else:
            parts = dataset_name.split('__')
            if len(parts) == 2:
                base_dataset_name = parts[1]
                if base_dataset_name in ds_cfg:
                    self.target = ds_cfg[base_dataset_name]["target"]
                    self.dataset_name = dataset_name
                else:
                    raise KeyError(f"Dataset '{dataset_name}' not found.")
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found.")

        logger.info(f"[FHE:{self.name}] Loaded {len(self.df)} rows, target='{self.target}'")

    def split(self) -> None:
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

    def train(self) -> None:
        if self.X_train is None:
            raise RuntimeError("Call split() before train()")

        self.model.fit(self.X_train, self.y_train)
        self.model.compile(self.X_train)
        self._save_model()

    def predict(self, X: np.ndarray, fhe: str = "simulate") -> np.ndarray:
        return self.model.predict(X, fhe=fhe)

    def predict_proba(self, X: np.ndarray, fhe: str = "simulate") -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X, fhe=fhe)[:, 1]
        return None

    def evaluate(self, on: str = "test", fhe: str = "simulate") -> dict:
        splits = {
            "train": (self.X_train, self.y_train),
            "test":  (self.X_test,  self.y_test),
        }

        X, y = splits[on]
        y_pred  = self.predict(X, fhe=fhe)
        y_proba = self.predict_proba(X, fhe=fhe)

        metric_names = self.cfg.get("metrics") or list(SUPPORTED_METRICS)
        results = {}
        for metric in metric_names:
            if metric in SUPPORTED_METRICS:
                results[metric] = round(SUPPORTED_METRICS[metric](y, y_pred, y_proba), 4)

        self._save_results(results, on, fhe)
        return results

    @classmethod
    def load(cls, path: str) -> "FHEModel":
        path = Path(path)
        with open(path, "r") as f:
            model_obj = cml_load(f)

        instance = cls(name="loaded_fhe_model")
        instance.model = model_obj
        return instance

    def compile(self) -> None:
        if self.X_train is None:
            raise RuntimeError("Need training data")
        self.model.compile(self.X_train)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare_hyperparams(self, name: str, hyperparams: dict) -> dict:
        params = hyperparams.copy()

        # Remove sklearn-only params for MLP
        if name == "mlp":
            for key in _SKLEARN_ONLY:
                params.pop(key, None)

        fhe_params = self.fhe_cfg.get("models", {}).get(name, {})

        if name != "mlp":
            if "n_bits" in fhe_params:
                params["n_bits"] = fhe_params["n_bits"]

        else:
            mapping = {
                "n_w_bits": "module__n_w_bits",
                "n_a_bits": "module__n_a_bits",
                "n_accum_bits": "module__n_accum_bits",
                "n_layers": "n_layers",
            }
            for yaml_key, model_key in mapping.items():
                if yaml_key in fhe_params:
                    params[model_key] = fhe_params[yaml_key]

        return params

    def _save_model(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)

        dataset_to_use = self.save_dataset_name or self.dataset_name
        path = self.models_dir / f"{self.mode}__{self.name}__{dataset_to_use}.json"

        with open(path, "w") as f:
            cml_dump(self.model, f)

    def _save_results(self, results: dict, split: str, fhe: str) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

        path = self.results_dir / f"fhe__{self.name}__{self.dataset_name}__{split}__metrics.json"
        with open(path, "w") as f:
            json.dump({
                "mode": self.mode,
                "model": self.name,
                "dataset": self.dataset_name,
                "split": split,
                "fhe": fhe,
                "metrics": results
            }, f, indent=2)