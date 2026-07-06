# src/synthesizer_wrappers.py
"""Native synthesizer wrappers — no synthcity dependency.

Each class exposes the same interface expected by Synthesizer in
src/synthesizers.py:

    wrapper.fit(df: pd.DataFrame) -> None
    wrapper.generate(count: int)  -> GeneratedData   (.dataframe() method)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeneratedData:
    """Thin wrapper satisfying the .dataframe() protocol used by Synthesizer.sample()."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def dataframe(self) -> pd.DataFrame:
        return self._df


# ---------------------------------------------------------------------------
# Bayesian Network  (pgmpy)
# ---------------------------------------------------------------------------


class BayesianNetworkWrapper:
    """Bayesian network synthesizer backed by pgmpy.

    Continuous columns are discretised with KBinsDiscretizer before structure
    learning (pgmpy handles only discrete states). Intra-bin noise is added on
    inverse-transform so samples are continuous, not bin-centre point masses.
    """

    def __init__(
        self,
        struct_learning_n_iter: int = 100,
        struct_learning_search_method: str = "tree_search",
        struct_learning_score: str = "k2",
        struct_max_indegree: int = 4,
        encoder_max_clusters: int = 10,
        encoder_noise_scale: float = 0.1,
        **kwargs,
    ) -> None:
        self.n_iter = struct_learning_n_iter
        self.search_method = struct_learning_search_method
        self.score = struct_learning_score
        self.max_indegree = struct_max_indegree
        self.n_bins = encoder_max_clusters
        self.noise_scale = encoder_noise_scale

        self._model = None
        self._cont_cols: list = []
        self._cat_cols: list = []
        self._label_encoders: dict = {}
        self._bin_discretizers: dict = {}
        self._bin_edges: dict = {}
        self._columns: list = []

    def fit(self, df: pd.DataFrame) -> None:
        import importlib as _il
        from pgmpy.estimators import MaximumLikelihoodEstimator
        from pgmpy.models import DiscreteBayesianNetwork

        def _pgmpy_score(*names):
            for mod_path in ("pgmpy.estimators", "pgmpy.structure_score"):
                try:
                    mod = _il.import_module(mod_path)
                except ImportError:
                    continue
                for name in names:
                    cls = getattr(mod, name, None)
                    if cls is not None:
                        return cls
            raise ImportError(f"none of pgmpy score classes {names!r} found")

        K2Score   = _pgmpy_score("K2Score", "K2")
        BDeuScore = _pgmpy_score("BDeuScore", "BDeu")
        BicScore  = _pgmpy_score("BicScore", "BIC")
        try:
            BDsScore = _pgmpy_score("BDsScore", "BDs")
        except ImportError:
            BDsScore = K2Score
        from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

        self._columns = list(df.columns)

        # Columns with more unique values than n_bins are treated as continuous
        self._cont_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > self.n_bins
        ]
        self._cat_cols = [c for c in df.columns if c not in self._cont_cols]

        encoded = df.copy()

        for col in self._cont_cols:
            n_bins = min(self.n_bins, df[col].nunique())
            disc = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
            )
            encoded[col] = disc.fit_transform(df[[col]]).astype(int).ravel()
            self._bin_discretizers[col] = disc
            self._bin_edges[col] = disc.bin_edges_[0]

        for col in self._cat_cols:
            le = LabelEncoder()
            encoded[col] = le.fit_transform(df[col].astype(str))
            self._label_encoders[col] = le

        encoded = encoded.astype(int)

        _score_map = {
            "k2": K2Score, "bdeu": BDeuScore, "bic": BicScore, "bds": BDsScore,
        }
        scoring = _score_map.get(self.score, K2Score)(encoded)

        dag = self._learn_structure(encoded, scoring)
        edges = list(dag.edges())

        bn = DiscreteBayesianNetwork(edges)
        for col in self._columns:
            if col not in bn.nodes():
                bn.add_node(col)

        bn.fit(encoded, estimator=MaximumLikelihoodEstimator)
        self._model = bn

        logger.debug("[BayesianNetworkWrapper] DAG learned: %d edges", len(edges))

    def _learn_structure(self, encoded: pd.DataFrame, scoring):
        from pgmpy.estimators import HillClimbSearch, MmhcEstimator, PC, TreeSearch

        m = self.search_method

        if m == "hillclimb":
            return HillClimbSearch(encoded).estimate(
                scoring_method=scoring,
                max_indegree=self.max_indegree,
                max_iter=self.n_iter,
            )
        if m == "tree_search":
            return TreeSearch(encoded, root_node=self._columns[0]).estimate(
                estimator_type="chow-liu"
            )
        if m == "pc":
            return PC(encoded).estimate(return_type="dag", significance_level=0.05)
        if m == "mmhc":
            return MmhcEstimator(encoded).estimate(
                tabu_length=10,
                significance_level=0.05,
                scoring_method=scoring,
            )
        # exhaustive or unknown: fall back to hillclimb
        return HillClimbSearch(encoded).estimate(
            scoring_method=scoring,
            max_indegree=self.max_indegree,
            max_iter=self.n_iter,
        )

    def generate(self, count: int) -> GeneratedData:
        samples = self._model.simulate(n_samples=count, show_progress=False)[self._columns]

        result = samples.copy().astype(object)

        for col in self._cont_cols:
            edges = self._bin_edges[col]
            bin_idx = samples[col].clip(0, len(edges) - 2).astype(int).values
            lo, hi = edges[bin_idx], edges[bin_idx + 1]
            noise = np.random.uniform(-0.5, 0.5, size=len(samples))
            # Midpoint + scaled intra-bin noise
            result[col] = (lo + hi) / 2.0 + noise * (hi - lo) * self.noise_scale

        for col in self._cat_cols:
            le = self._label_encoders[col]
            idx = np.clip(
                np.round(samples[col].values).astype(int),
                0,
                len(le.classes_) - 1,
            )
            result[col] = le.inverse_transform(idx)

        return GeneratedData(result)


# ---------------------------------------------------------------------------
# Normalizing Flows  (nflows)
# ---------------------------------------------------------------------------


class NFlowWrapper:
    """Normalizing flows synthesizer backed by nflows + PyTorch.

    Categorical columns are one-hot encoded; numeric columns are standardised.
    All features are modelled jointly as a single normalizing flow over the
    combined encoded space.
    """

    # Config string → nflows transform class name
    _AUTOREGRESSIVE = {
        "rq-autoregressive":        "MaskedPiecewiseRationalQuadraticAutoregressiveTransform",
        "quadratic-autoregressive": "MaskedPiecewiseQuadraticAutoregressiveTransform",
        "affine-autoregressive":    "MaskedAffineAutoregressiveTransform",
    }
    _COUPLING = {
        "rq-coupling":       "PiecewiseRationalQuadraticCouplingTransform",
        "quadratic-coupling":"PiecewiseQuadraticCouplingTransform",
        "affine-coupling":   "AffineCouplingTransform",
    }

    def __init__(
        self,
        n_iter: int = 100,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        batch_size: int = 200,
        dropout: float = 0.1,
        lr: float = 0.001,
        base_transform_type: str = "rq-autoregressive",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self.n_iter = n_iter
        self.n_layers = n_layers_hidden
        self.n_units = n_units_hidden
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.transform_type = base_transform_type
        self.device = device

        self._flow = None
        self._cont_cols: list = []
        self._cat_cols: list = []
        self._scaler = None
        self._ohe_encoders: dict = {}
        self._columns: list = []
        self._d: int = 0

    # ---- encoding / decoding -----------------------------------------------

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self._cont_cols:
            parts.append(self._scaler.transform(df[self._cont_cols].values))
        for col in self._cat_cols:
            parts.append(self._ohe_encoders[col].transform(df[[col]]))
        if not parts:
            return np.empty((len(df), 0), dtype=np.float32)
        return np.hstack(parts).astype(np.float32)

    def _decode(self, arr: np.ndarray) -> pd.DataFrame:
        result = {}
        idx = 0

        if self._cont_cols:
            n = len(self._cont_cols)
            cont = self._scaler.inverse_transform(arr[:, idx:idx + n])
            for i, col in enumerate(self._cont_cols):
                values = cont[:, i]
                if col in self._binary_cols:
                    result[col] = np.clip(np.round(values).astype(int), 0, 1)
                else:
                    result[col] = values
            idx += n

        for col in self._cat_cols:
            enc = self._ohe_encoders[col]
            n_cats = len(enc.categories_[0])
            cat_idx = np.argmax(arr[:, idx:idx + n_cats], axis=1)
            cat_idx = np.clip(cat_idx, 0, n_cats - 1)
            result[col] = enc.categories_[0][cat_idx]
            idx += n_cats

        return pd.DataFrame(result)[self._columns]

    # ---- flow construction -------------------------------------------------

    def _build_flow(self, d: int):
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import ReversePermutation

        transforms = []
        for layer_idx in range(self.n_layers):
            transforms.append(self._build_single_transform(d, layer_idx))
            transforms.append(ReversePermutation(features=d))

        return Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([d]),
        )

    def _build_single_transform(self, d: int, layer_idx: int):
        t = self.transform_type

        if t in self._AUTOREGRESSIVE:
            return self._build_autoregressive(self._AUTOREGRESSIVE[t], d)
        if t in self._COUPLING:
            return self._build_coupling(self._COUPLING[t], d, layer_idx)

        # Unknown type: fall back to rq-autoregressive
        logger.warning(
            "[NFlowWrapper] Unknown base_transform_type '%s', using rq-autoregressive", t
        )
        return self._build_autoregressive(
            "MaskedPiecewiseRationalQuadraticAutoregressiveTransform", d
        )

    def _build_autoregressive(self, cls_name: str, d: int):
        import nflows.transforms.autoregressive as ar_mod

        TransformClass = getattr(ar_mod, cls_name)
        kw = dict(features=d, hidden_features=self.n_units, dropout_probability=self.dropout)
        if "RationalQuadratic" in cls_name or "Quadratic" in cls_name:
            kw.update(num_bins=8, tails="linear", tail_bound=3.0)
        return TransformClass(**kw)

    def _build_coupling(self, cls_name: str, d: int, layer_idx: int):
        import torch
        import nflows.transforms.coupling as cp_mod
        from nflows.nn.nets import ResidualNet

        TransformClass = getattr(cp_mod, cls_name)

        # Alternating masks: even layers keep even-index dims, odd layers keep odd-index dims
        mask = torch.zeros(d)
        mask[layer_idx % 2::2] = 1

        n_units = self.n_units
        dropout = self.dropout

        def create_net(in_features, out_features):
            return ResidualNet(
                in_features, out_features,
                hidden_features=n_units,
                num_blocks=2,
                dropout_probability=dropout,
            )

        kw: dict = dict(mask=mask, transform_net_create_fn=create_net)
        if "RationalQuadratic" in cls_name:
            kw.update(num_bins=8, tails="linear", tail_bound=3.0)
        return TransformClass(**kw)

    # ---- fit / generate ----------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        import torch
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        self._columns = list(df.columns)
        self._cont_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self._cat_cols = [c for c in df.columns if c not in self._cont_cols]
        self._binary_cols = {
            c for c in self._cont_cols if set(df[c].dropna().unique()).issubset({0, 1})
        }

        if self._cont_cols:
            self._scaler = StandardScaler()
            self._scaler.fit(df[self._cont_cols].values)

        for col in self._cat_cols:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc.fit(df[[col]])
            self._ohe_encoders[col] = enc

        X = self._encode(df)
        self._d = X.shape[1]

        self._flow = self._build_flow(self._d).to(self.device)

        dataset = torch.tensor(X, dtype=torch.float32)
        optimizer = torch.optim.Adam(self._flow.parameters(), lr=self.lr)

        self._flow.train()
        n = len(X)
        for step in range(self.n_iter):
            idx = np.random.randint(0, n, size=min(self.batch_size, n))
            batch = dataset[idx].to(self.device)
            optimizer.zero_grad()
            loss = -self._flow.log_prob(batch).mean()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._flow.parameters(), max_norm=5.0)
                optimizer.step()

        self._flow.eval()
        logger.debug(
            "[NFlowWrapper] Training complete — %d steps, encoded dim=%d", self.n_iter, self._d
        )

    def generate(self, count: int) -> GeneratedData:
        import torch

        self._flow.eval()
        with torch.no_grad():
            samples = self._flow.sample(count).cpu().numpy()

        df = self._decode(samples)
        return GeneratedData(df)


# ---------------------------------------------------------------------------
# Adversarial Random Forests  (arfpy)
# ---------------------------------------------------------------------------


class ARFWrapper:
    """ARF synthesizer backed by arfpy.

    arfpy handles mixed-type (categorical + continuous) tabular data natively,
    so no preprocessing is required here.
    """

    def __init__(
        self,
        num_trees: int = 10,
        delta: int = 0,
        max_iters: int = 10,
        early_stop: bool = True,
        verbose: bool = False,
        min_node_size: int = 5,
        **kwargs,
    ) -> None:
        self.num_trees = num_trees
        self.delta = delta
        self.max_iters = max_iters
        self.early_stop = early_stop
        self.verbose = verbose
        self.min_node_size = min_node_size
        self._model = None

    def fit(self, df: pd.DataFrame) -> None:
        import numpy as _np
        if not hasattr(_np, "in1d"):
            _np.in1d = _np.isin  # removed in NumPy 2.0; arfpy still uses it
        from arfpy.arf import arf

        self._model = arf(
            df,
            num_trees=self.num_trees,
            delta=self.delta,
            max_iters=self.max_iters,
            early_stop=self.early_stop,
            verbose=self.verbose,
            min_node_size=self.min_node_size,
        )
        self._model.forde()
        logger.debug("[ARFWrapper] ARF fitted — %d trees", self.num_trees)

    def generate(self, count: int) -> GeneratedData:
        return GeneratedData(self._model.forge(n=count))
