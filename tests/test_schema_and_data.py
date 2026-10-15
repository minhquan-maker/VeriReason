import pandas as pd
import pytest

from tests.conftest import REPO
from verireason.data import german, home_credit
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
    assert s.groups["outstanding_debt"].derived == ["BUREAU_UTIL"]
    assert "INST_LATE_RATIO" in s.groups["payment_history"].features
    assert set(home_credit.BUREAU_COLUMNS + home_credit.INSTALMENT_COLUMNS) <= set(s.raw_features)


def test_home_credit_app_only_schema_is_valid():
    s = FeatureSchema.from_yaml(REPO / "configs/schemas/home_credit_app_only.yaml")
    assert "CREDIT_INCOME_RATIO" in s.groups["loan_and_affordability"].derived
    assert "payment_history" not in s.groups


def _write_home_credit_fixture(d):
    app = {c: [0, 0, 0] for c in home_credit._RAW_COLUMNS}
    app.update({"SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0], "DAYS_BIRTH": [-10000] * 3,
                "DAYS_EMPLOYED": [-1000, 365243, -500], "DAYS_ID_PUBLISH": [-100] * 3,
                "DAYS_LAST_PHONE_CHANGE": [-50] * 3})
    for c in ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
              "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "FLAG_OWN_CAR",
              "FLAG_OWN_REALTY", "CODE_GENDER"]:
        app[c] = ["a"] * 3
    pd.DataFrame(app).to_csv(d / "application_train.csv", index=False)
    pd.DataFrame({  # applicant 3 has no bureau record; applicant 99 is not an applicant
        "SK_ID_CURR": [1, 1, 2, 99], "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active"],
        "CREDIT_DAY_OVERDUE": [0, 30, 5, 400], "AMT_CREDIT_SUM": [1000.0, 500.0, 200.0, 9.0],
        "AMT_CREDIT_SUM_DEBT": [400.0, None, 100.0, 9.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 10.0, 0.0, 0.0]}).to_csv(d / "bureau.csv", index=False)
    pd.DataFrame({  # applicant 1: 4 instalments, 2 late (3 and 7 days), one early; 3 has none
        "SK_ID_CURR": [1, 1, 1, 1, 2, 2], "DAYS_INSTALMENT": [-100, -70, -40, -10, -90, -60],
        "DAYS_ENTRY_PAYMENT": [-103, -67, -33, None, -90, -60]}
    ).to_csv(d / "installments_payments.csv", index=False)


def test_home_credit_aggregates(tmp_path):
    _write_home_credit_fixture(tmp_path)
    df = home_credit.load(tmp_path / "application_train.csv",
                          bureau_path=tmp_path / "bureau.csv",
                          installments_path=tmp_path / "installments_payments.csv")
    assert list(df["applicant_id"]) == ["hc1", "hc2", "hc3"]
    a1, a2, a3 = (df.iloc[i] for i in range(3))
    assert (a1.BUREAU_OVERDUE_MAX, a1.BUREAU_DEBT_SUM, a1.BUREAU_CREDIT_SUM,
            a1.BUREAU_ACTIVE_CNT, a1.HAS_BUREAU_HISTORY) == (30, 400.0, 1500.0, 1, 1)
    assert a1.BUREAU_OVERDUE_AMT == 10.0
    # unpaid instalment ignored; days late = 0, 3, 7 -> ratio 2/3 (the early one is clipped to 0)
    assert a1.INST_LATE_RATIO == pytest.approx(2 / 3)
    assert a1.INST_DAYS_LATE_MEAN == pytest.approx(10 / 3)
    assert a1.INST_DAYS_LATE_MAX == 7
    assert a2.INST_LATE_RATIO == 0 and a2.HAS_INST_HISTORY == 1
    assert a3.HAS_BUREAU_HISTORY == 0 and a3.HAS_INST_HISTORY == 0
    assert a3.BUREAU_DEBT_SUM == 0 and a3.INST_DAYS_LATE_MAX == 0
    assert pd.isna(a2.YEARS_EMPLOYED)          # 365243 sentinel -> NaN
    schema = FeatureSchema.from_yaml(REPO / "configs/schemas/home_credit.yaml")
    assert set(schema.raw_features) <= set(df.columns)
    util = schema.add_derived(df)["BUREAU_UTIL"]
    assert util.iloc[0] == pytest.approx(400 / 1500) and pd.isna(util.iloc[2])


def test_home_credit_application_only_has_no_aggregates(tmp_path):
    _write_home_credit_fixture(tmp_path)
    df = home_credit.load(tmp_path / "application_train.csv")
    assert not set(home_credit.BUREAU_COLUMNS) & set(df.columns)


def test_home_credit_missing_secondary_file(tmp_path):
    _write_home_credit_fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        home_credit.load(tmp_path / "application_train.csv", bureau_path=tmp_path / "nope.csv")


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
