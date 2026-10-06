"""Scoring model f: raw applicant features -> P(default).

`ScoringModel` owns h (derived-feature recomputation) and the encoder, so verifiers can intervene
on raw features and always get an internally consistent profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from verireason.data.schema import FeatureSchema


def _encoder(schema: FeatureSchema, scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(num_steps), schema.numeric_columns),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
          schema.categorical_columns)],
        remainder="drop", verbose_feature_names_out=False,
    )


@dataclass
class ScoringModel:
    name: str
    kind: str                       # "xgboost" | "logistic"
    schema: FeatureSchema
    pipeline: Pipeline
    tau: float = 0.5
    meta: dict = field(default_factory=dict)

    # --- construction ---------------------------------------------------------------------
    @classmethod
    def build(cls, name: str, kind: str, schema: FeatureSchema, params: dict,
              seed: int = 0) -> ScoringModel:
        if kind == "logistic":
            clf = LogisticRegression(max_iter=5000, random_state=seed, **params)
            enc = _encoder(schema, scale=True)
        elif kind == "xgboost":
            from xgboost import XGBClassifier
            clf = XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1, **params)
            enc = _encoder(schema, scale=False)
        else:
            raise ValueError(f"unknown model type {kind!r}")
        return cls(name=name, kind=kind, schema=schema,
                   pipeline=Pipeline([("enc", enc), ("clf", clf)]))

    def fit(self, X_raw: pd.DataFrame, y: np.ndarray) -> ScoringModel:
        self.pipeline.fit(self._prep(X_raw), y)
        return self

    # --- scoring --------------------------------------------------------------------------
    def _prep(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = self.schema.add_derived(X_raw[self.schema.raw_features])
        for c in self.schema.categorical_columns:
            X[c] = X[c].astype(str)
        return X

    def predict_risk(self, X_raw: pd.DataFrame) -> np.ndarray:
        """f(x): predicted probability of default."""
        return self.pipeline.predict_proba(self._prep(X_raw))[:, 1]

    def encode(self, X_raw: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.named_steps["enc"].transform(self._prep(X_raw)), float)

    # --- attributions (baseline B2 and generator input) --------------------------------------
    def _encoded_columns(self) -> list[str]:
        """Raw/derived column behind every encoded column (same order as `encode`)."""
        enc: ColumnTransformer = self.pipeline.named_steps["enc"]
        cols: list[str] = []
        for name, trans, cs in enc.transformers_:
            if name == "num":
                cols += list(cs)
            elif name == "cat":
                for c, cats in zip(cs, trans.categories_, strict=True):
                    cols += [c] * len(cats)
        return cols

    def encoded_column_groups(self) -> list[str]:
        col2group = self.schema.column_to_group()
        return [col2group[c] for c in self._encoded_columns()]

    def _contributions(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Per encoded column, logit (margin) scale.

        XGBoost: TreeSHAP via xgboost's pred_contribs (path-dependent, no background data).
        Logistic: exact interventional SHAP w_j (z_j - E_R[z_j]) with R the reference population
        (its encoded mean is stored in meta at training time).
        """
        Z = self.encode(X_raw)
        if self.kind == "xgboost":
            import xgboost as xgb
            booster = self.pipeline.named_steps["clf"].get_booster()
            return booster.predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]  # drop bias
        w = self.pipeline.named_steps["clf"].coef_[0]
        return (Z - np.asarray(self.meta["encoded_reference_mean"])) * w

    def group_attributions(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """phi_g = sum of contributions over each group's encoded columns (logit scale)."""
        contrib = pd.DataFrame(self._contributions(X_raw))
        groups = np.array(self.encoded_column_groups())
        return contrib.T.groupby(groups).sum().T.reindex(columns=list(self.schema.groups))

    def feature_attributions(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Per raw/derived feature (one-hot columns summed); this is what generators see."""
        contrib = pd.DataFrame(self._contributions(X_raw))
        return contrib.T.groupby(np.array(self._encoded_columns())).sum().T

    # --- persistence ----------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> ScoringModel:
        return joblib.load(path)
