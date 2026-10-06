import pandas as pd
import pytest

from tests.conftest import REPO
from verireason.data import german
from verireason.data.schema import FeatureSchema


def test_german_schema_partitions_features():
    s = FeatureSchema.from_yaml(REPO / "configs/schemas/german.yaml")
    owned = [f for g in s.groups.values() for f in g.features]
    assert sorted(owned) == sorted(s.raw_features)
    assert s.groups["loan_terms"].derived == ["monthly_payment"]
    assert s.groups["age"].immutable and not s.groups["age"].actionable


def test_home_credit_schema_is_valid():
    s = FeatureSchema.from_yaml(REPO / "configs/schemas/home_credit.yaml")
    assert "CREDIT_INCOME_RATIO" in s.groups["loan_and_affordability"].derived


def test_derived_sources_must_share_a_group():
    raw = {"target": "y",
           "features": {"a": {"type": "numeric"}, "b": {"type": "numeric"}},
           "derived": {"r": {"op": "ratio", "num": "a", "den": "b"}},
           "groups": {"ga": {"features": ["a"]}, "gb": {"features": ["b"]}}}
    with pytest.raises(ValueError, match="one group"):
        FeatureSchema.from_dict(raw)


def test_feature_in_two_groups_rejected():
    raw = {"target": "y", "features": {"a": {"type": "numeric"}},
           "groups": {"g1": {"features": ["a"]}, "g2": {"features": ["a"]}}}
    with pytest.raises(ValueError):
        FeatureSchema.from_dict(raw)


def test_parse_uci_german_format(tmp_path):
    syn = german.synthesize(n=20, seed=1)
    cols = german.UCI_ORDER[:-1]
    uci = syn[cols].copy()
    uci["class"] = syn["default"].map({0: 1, 1: 2})
    path = tmp_path / "german.data"
    uci.to_csv(path, sep=" ", header=False, index=False)
    df = german.load(path)
    assert len(df) == 20
    assert (df["default"].to_numpy() == syn["default"].to_numpy()).all()
    assert pd.api.types.is_numeric_dtype(df["credit_amount"])
