"""Feature schema: model inputs, derived features (h), and reason groups G.

The schema is the single place where domain knowledge enters the pipeline. It is loaded from
configs/schemas/<dataset>.yaml and validated so that:
  * groups partition the model's raw features (no feature in two groups, none left out);
  * every derived feature's sources lie in exactly one group, which then owns the derived feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass
class Feature:
    name: str
    type: str                       # "numeric" | "categorical"
    values: dict[str, str] = field(default_factory=dict)   # code -> human label (categorical)
    unit: str | None = None
    immutable: bool = False


@dataclass
class Derived:
    name: str
    op: str
    sources: list[str]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if self.op == "ratio":
            num, den = (df[s].astype(float) for s in self.sources)
            return num / den.replace(0, np.nan)
        raise ValueError(f"unknown derived op {self.op!r}")


@dataclass
class Group:
    name: str
    label: str
    features: list[str]             # raw features replaced together by an intervention
    derived: list[str] = field(default_factory=list)
    immutable: bool = False
    actionable: bool = True
    aliases: list[str] = field(default_factory=list)

    @property
    def model_columns(self) -> list[str]:
        return self.features + self.derived


@dataclass
class FeatureSchema:
    target: str
    features: dict[str, Feature]
    derived: dict[str, Derived]
    groups: dict[str, Group]
    dropped: list[str] = field(default_factory=list)

    # --- loading -------------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> FeatureSchema:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FeatureSchema:
        feats = {
            n: Feature(name=n, type=spec["type"], values={str(k): v for k, v in
                                                         (spec.get("values") or {}).items()},
                       unit=spec.get("unit"), immutable=bool(spec.get("immutable", False)))
            for n, spec in raw["features"].items()
        }
        derived = {
            n: Derived(name=n, op=spec["op"], sources=[spec["num"], spec["den"]])
            for n, spec in (raw.get("derived") or {}).items()
        }
        groups = {
            n: Group(name=n, label=spec.get("label", n), features=list(spec["features"]),
                     immutable=bool(spec.get("immutable", False)),
                     actionable=bool(spec.get("actionable", not spec.get("immutable", False))),
                     aliases=list(spec.get("aliases", [])))
            for n, spec in raw["groups"].items()
        }
        schema = cls(target=raw["target"], features=feats, derived=derived, groups=groups,
                     dropped=list(raw.get("dropped", [])))
        schema._assign_derived()
        schema.validate()
        return schema

    def _assign_derived(self) -> None:
        owner = self.feature_to_group()
        for d in self.derived.values():
            owners = {owner.get(s) for s in d.sources}
            if len(owners) != 1 or None in owners:
                raise ValueError(
                    f"derived feature {d.name!r}: sources {d.sources} must all belong to one group "
                    f"(found {owners}); otherwise group interventions are not self-contained")
            self.groups[owners.pop()].derived.append(d.name)

    def validate(self) -> None:
        seen: dict[str, str] = {}
        for g in self.groups.values():
            for f in g.features:
                if f not in self.features:
                    raise ValueError(f"group {g.name!r} references unknown feature {f!r}")
                if f in seen:
                    raise ValueError(f"feature {f!r} is in groups {seen[f]!r} and {g.name!r}")
                seen[f] = g.name
        missing = set(self.features) - set(seen)
        if missing:
            raise ValueError(f"features not assigned to any group: {sorted(missing)}")
        for f in self.dropped:
            if f in self.features:
                raise ValueError(f"dropped feature {f!r} is also listed as a model feature")

    # --- views ---------------------------------------------------------------------------
    def feature_to_group(self) -> dict[str, str]:
        return {f: g.name for g in self.groups.values() for f in g.features}

    def column_to_group(self) -> dict[str, str]:
        """Model column (raw or derived) -> group."""
        return {c: g.name for g in self.groups.values() for c in g.model_columns}

    @property
    def raw_features(self) -> list[str]:
        return list(self.features)

    @property
    def model_columns(self) -> list[str]:
        return self.raw_features + list(self.derived)

    @property
    def numeric_columns(self) -> list[str]:
        return [f for f, s in self.features.items() if s.type == "numeric"] + list(self.derived)

    @property
    def categorical_columns(self) -> list[str]:
        return [f for f, s in self.features.items() if s.type == "categorical"]

    def add_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """h: recompute every derived feature from the (possibly intervened) raw features."""
        out = df.copy()
        for d in self.derived.values():
            out[d.name] = d.compute(out)
        return out

    def describe_value(self, feature: str, value: Any) -> str:
        spec = self.features[feature]
        if spec.type == "categorical":
            return spec.values.get(str(value), str(value))
        if isinstance(value, float) and not np.isnan(value):
            value = round(value, 3)
        return f"{value} {spec.unit}" if spec.unit else str(value)
