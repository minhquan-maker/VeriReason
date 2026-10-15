"""70/15/15 split and tau set on the validation split (pilot plan Step 4)."""

from pathlib import Path

from verireason.data.schema import FeatureSchema
from verireason.extract.extractor import KeywordExtractor
from verireason.models.train import load_model, load_split
from verireason.records import NOT_IN_MODEL, Explanation

REPO = Path(__file__).resolve().parent.parent


def test_split_sizes_and_disjoint(trained):
    parts = {s: load_split(trained, s) for s in ("reference", "validation", "test")}
    n = sum(len(p) for p in parts.values())
    assert abs(len(parts["reference"]) / n - 0.70) < 0.01
    assert abs(len(parts["validation"]) / n - 0.15) < 0.01
    assert abs(len(parts["test"]) / n - 0.15) < 0.01
    ids = [set(p["applicant_id"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_tau_is_set_on_validation(trained):
    model = load_model(trained, "xgb")
    val = load_split(trained, "validation")
    rate = float((model.predict_risk(val) >= model.tau).mean())
    assert abs(rate - trained["decision"]["denial_rate"]) < 0.02


def test_home_credit_schemas_exclude_sensitive_attributes():
    for name in ("home_credit", "home_credit_app_only"):
        schema = FeatureSchema.from_yaml(REPO / "configs" / "schemas" / f"{name}.yaml")
        for f in ("AGE_YEARS", "NAME_FAMILY_STATUS", "CNT_CHILDREN", "CNT_FAM_MEMBERS"):
            assert f not in schema.features
        assert not any(g.immutable for g in schema.groups.values())
        assert {"age", "children", "marital"} <= set(schema.sensitive_aliases)


def test_age_and_family_mentions_are_not_in_model():
    schema = FeatureSchema.from_yaml(REPO / "configs" / "schemas" / "home_credit.yaml")
    ex = KeywordExtractor(schema)

    def groups(text):
        exp = Explanation(explanation_id="e", applicant_id="a", model="xgb", generator="g",
                          llm_model="g", setup="raw", sample=0, prompt="", text=text)
        return {c.group for c in ex.extract(exp)}

    assert NOT_IN_MODEL in groups("Your young age counted against you.")
    assert NOT_IN_MODEL in groups("Having three children raised your risk.")
    assert NOT_IN_MODEL not in groups("Your agency fees were high.")
