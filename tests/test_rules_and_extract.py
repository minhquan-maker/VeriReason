from tests.conftest import REPO
from verireason.data.schema import FeatureSchema
from verireason.extract.extractor import KeywordExtractor
from verireason.label.rules import label_claim, label_explanation
from verireason.records import NOT_IN_MODEL, ApplicantVerdict, Claim, Explanation
from verireason.verify.intervention import verdicts_from_effects

SCHEMA = FeatureSchema.from_yaml(REPO / "configs/schemas/german.yaml")


def _verdict() -> ApplicantVerdict:
    effects = {g: (0.0, 0.0, 10) for g in SCHEMA.groups}
    effects.update(checking_account=(0.30, 0.0, 10), loan_terms=(0.20, 0.0, 10),
                   credit_history=(0.10, 0.0, 10), savings=(0.05, 0.0, 10),
                   employment=(0.03, 0.0, 10), property_and_housing=(-0.10, 0.0, 10))
    groups, principal = verdicts_from_effects(effects, z=1.96, min_effect=0.01, top_k=4)
    return ApplicantVerdict(applicant_id="a", model="xgb", sampler="test", risk=0.8, tau=0.5,
                            top_k=4, groups=groups, principal=principal)


def _claim(group, direction="raises_risk", prominence="contributing", action=None):
    return Claim(claim_id="c", explanation_id="e", quote="q", group=group, direction=direction,
                 prominence=prominence, action=action)


def test_taxonomy_rules():
    v = _verdict()
    lab = lambda c: label_claim(c, v, SCHEMA).verdict  # noqa: E731
    assert lab(_claim("checking_account", prominence="principal")) == "supported"
    assert lab(_claim(None)) == "E5"
    assert lab(_claim(NOT_IN_MODEL)) == "E1"
    assert lab(_claim("telephone")) == "E1"                           # zero effect
    assert lab(_claim("property_and_housing")) == "E2"                # model: favourable
    assert lab(_claim("checking_account", direction="lowers_risk")) == "E2"
    assert lab(_claim("employment", prominence="principal")) == "E3"  # rank 5 > k
    assert lab(_claim("checking_account", prominence="minor")) == "E3"
    assert lab(_claim("age", direction="neutral", action="become younger")) == "E6"


def test_omission_is_response_level():
    v = _verdict()
    exp = Explanation(explanation_id="e", applicant_id="a", model="xgb", generator="g",
                      setup="raw", sample=0, prompt="", text="")
    out = label_explanation(exp, [_claim("checking_account")], v, SCHEMA)
    assert out.omissions == ["loan_terms", "credit_history", "savings"]


def test_keyword_extractor_types_claims():
    text = ("The main reason is your checking account status, which raised the assessed risk. "
            "Your savings helped your application. Overall, your profile was too risky. "
            "Your gender also counted against you.")
    exp = Explanation(explanation_id="e", applicant_id="a", model="xgb", generator="g",
                      setup="raw", sample=0, prompt="", text=text)
    claims = KeywordExtractor(SCHEMA).extract(exp)
    got = {(c.group, c.direction, c.prominence) for c in claims}
    assert ("checking_account", "raises_risk", "principal") in got
    assert ("savings", "lowers_risk", "contributing") in got
    assert any(c.group is None for c in claims)
    assert any(c.group == NOT_IN_MODEL for c in claims)
    for c in claims:
        assert text[c.start:c.end] == c.quote


def test_llm_extractor_parses_and_locates_spans():
    from verireason.extract.extractor import LLMExtractor
    from verireason.llm.base import LLM, Completion

    text = "Your checking account balance raised your risk. Your phone number helped."

    class FakeLLM(LLM):
        name = "fake"

        def complete(self, system, user, *, max_tokens, temperature=None):
            assert "checking_account" in user and text in user
            return Completion(model="fake", text='```json\n{"claims": ['
                              '{"quote": "Your checking account balance raised your risk.", '
                              '"group": "checking_account", "direction": "raises_risk", '
                              '"prominence": "principal", "action": null}, '
                              '{"quote": "Your phone number helped.", "group": "unknown_group", '
                              '"direction": "lowers_risk", "prominence": "minor"}]}\n```')

    exp = Explanation(explanation_id="e", applicant_id="a", model="xgb", generator="g",
                      setup="raw", sample=0, prompt="", text=text)
    claims = LLMExtractor(SCHEMA, FakeLLM()).extract(exp)
    assert [c.group for c in claims] == ["checking_account", None]
    assert text[claims[0].start:claims[0].end] == claims[0].quote
