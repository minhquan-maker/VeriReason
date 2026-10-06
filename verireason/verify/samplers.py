"""Reference samplers Q_g(. | x_-g) for group interventions (proposal §2.1, §4.3).

All samplers return S alternative values for the RAW features of one group; derived features are
recomputed by the scoring model (h). Reported references (sensitivity analysis, §4.3):
  knn       conditional / on-manifold: values of g from the k nearest reference applicants in the
            space of the remaining raw features (cf. Aas et al., 2021)
  marginal  interventional: values of g from random reference applicants (cf. Janzing et al., 2020)
  median    population median (numeric) / mode (categorical); deterministic, S = 1
  approved  marginal over approved applicants in the reference population (f(x) < tau)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from verireason.data.schema import FeatureSchema


class Sampler(ABC):
    name: str
    deterministic: bool = False

    def __init__(self, schema: FeatureSchema, reference: pd.DataFrame):
        self.schema = schema
        self.reference = reference[schema.raw_features].reset_index(drop=True)

    @abstractmethod
    def sample(self, x: pd.Series, group: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Return `n` rows (or 1 if deterministic) with the raw features of `group`."""


class MarginalSampler(Sampler):
    name = "marginal"

    def sample(self, x, group, n, rng):
        cols = self.schema.groups[group].features
        idx = rng.integers(0, len(self.reference), size=n)
        return self.reference.loc[idx, cols].reset_index(drop=True)


class ApprovedSampler(MarginalSampler):
    name = "approved"

    def __init__(self, schema, reference, risk: np.ndarray, tau: float):
        super().__init__(schema, reference)
        keep = np.asarray(risk) < tau
        if keep.sum() == 0:
            raise ValueError("no approved applicants in the reference population")
        self.reference = self.reference[keep].reset_index(drop=True)


class MedianSampler(Sampler):
    name = "median"
    deterministic = True

    def __init__(self, schema, reference):
        super().__init__(schema, reference)
        self._center = {
            f: (self.reference[f].median() if spec.type == "numeric"
                else self.reference[f].mode().iloc[0])
            for f, spec in schema.features.items()
        }

    def sample(self, x, group, n, rng):
        cols = self.schema.groups[group].features
        return pd.DataFrame([{c: self._center[c] for c in cols}])


class KNNConditionalSampler(Sampler):
    """Plausibility-constrained conditional sampler: neighbours in x_-g, their values of g."""

    name = "knn"

    def __init__(self, schema, reference, k: int = 50, max_rows: int | None = None, seed: int = 0):
        super().__init__(schema, reference)
        if max_rows and len(self.reference) > max_rows:
            self.reference = self.reference.sample(n=max_rows, random_state=seed).reset_index(
                drop=True)
        self.k = min(k, len(self.reference))
        self._index: dict[str, tuple[ColumnTransformer, NearestNeighbors]] = {}

    def _fit_group(self, group: str):
        if group in self._index:
            return self._index[group]
        in_group = set(self.schema.groups[group].features)
        others = [f for f in self.schema.raw_features if f not in in_group]
        num = [f for f in others if self.schema.features[f].type == "numeric"]
        cat = [f for f in others if self.schema.features[f].type == "categorical"]
        enc = ColumnTransformer(
            [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler())]), num),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
        Z = enc.fit_transform(self._as_str(self.reference))
        nn = NearestNeighbors(n_neighbors=self.k).fit(Z)
        self._index[group] = (enc, nn)
        return enc, nn

    def _as_str(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.schema.categorical_columns:
            df[c] = df[c].astype(str)
        return df

    def sample(self, x, group, n, rng):
        enc, nn = self._fit_group(group)
        z = enc.transform(self._as_str(x.to_frame().T[self.schema.raw_features]))
        _, idx = nn.kneighbors(z)
        pick = rng.choice(idx[0], size=n, replace=True)
        cols = self.schema.groups[group].features
        return self.reference.loc[pick, cols].reset_index(drop=True)


def build_sampler(name: str, schema: FeatureSchema, reference: pd.DataFrame, *,
                  risk: np.ndarray | None = None, tau: float | None = None,
                  k: int = 50, max_rows: int | None = None, seed: int = 0) -> Sampler:
    if name == "marginal":
        return MarginalSampler(schema, reference)
    if name == "median":
        return MedianSampler(schema, reference)
    if name == "approved":
        return ApprovedSampler(schema, reference, risk=risk, tau=tau)
    if name == "knn":
        return KNNConditionalSampler(schema, reference, k=k, max_rows=max_rows, seed=seed)
    raise ValueError(f"unknown sampler {name!r}")
