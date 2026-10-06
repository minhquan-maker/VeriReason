"""Error taxonomy E1-E6 (proposal §2.2) as deterministic rules over claims and verdicts.

Claim-level (one verdict per claim; E5, then E1 for non-inputs, then E6 for recourse claims,
then E1 / E2 / E3):
  E5  non-specific        group is None (cannot be mapped to a reason group)
  E1  unsupported factor  group is not a model input, or stated direction != neutral but
                          delta_hat_g = 0
  E2  wrong direction     delta_hat_g != 0 and stated direction != delta_hat_g
                          (a 'neutral' statement about a non-zero group is also E2, as in §2.2)
  E3  wrong prominence    stated principal but g not in Top_k(G+),
                          or stated minor but g in Top_k(G+)
  E6  invalid recourse    recommended action targets an immutable group (the flip test
                          f(h(x')) < tau needs x' parsed from the action — TODO, see README)
Response-level:
  E4  omission            g in Top_k(G+) never mentioned by any claim
"""

from __future__ import annotations

from verireason.data.schema import FeatureSchema
from verireason.records import (
    NOT_IN_MODEL,
    ApplicantVerdict,
    Claim,
    ClaimLabel,
    Explanation,
    ExplanationLabel,
)

_STATED_SIGN = {"raises_risk": "+", "lowers_risk": "-", "neutral": "0"}


def label_claim(claim: Claim, verdict: ApplicantVerdict, schema: FeatureSchema) -> ClaimLabel:
    def mk(v: str, reason: str, gv=None) -> ClaimLabel:
        return ClaimLabel(claim_id=claim.claim_id, explanation_id=claim.explanation_id,
                          group=claim.group, verdict=v, reason=reason,
                          verified_direction=gv.direction if gv else None,
                          verified_role=gv.role if gv else None)

    if claim.group is None:
        return mk("E5", "claim cannot be mapped to a reason group")
    if claim.group == NOT_IN_MODEL or claim.group not in schema.groups:
        return mk("E1", "factor is not an input of the scoring model")

    gv = verdict.by_group()[claim.group]
    stated = _STATED_SIGN[claim.direction]
    if claim.action and schema.groups[claim.group].immutable:
        return mk("E6", "recommended action targets an immutable attribute", gv)
    if gv.direction == "0" and stated != "0":
        return mk("E1", f"no meaningful model effect (Delta={gv.delta:+.4f}, eps={gv.eps:.4f})", gv)
    if gv.direction != "0" and stated != gv.direction:
        return mk("E2", f"stated {claim.direction}, model effect {gv.direction} "
                        f"(Delta={gv.delta:+.4f})", gv)
    if claim.prominence == "principal" and gv.role != "principal":
        return mk("E3", f"presented as principal but verified role is {gv.role}", gv)
    if claim.prominence == "minor" and gv.role == "principal":
        return mk("E3", f"presented as minor but verified rank is {gv.rank}", gv)
    return mk("supported", "consistent with the intervention verdict", gv)


def label_explanation(exp: Explanation, claims: list[Claim], verdict: ApplicantVerdict,
                      schema: FeatureSchema) -> ExplanationLabel:
    labels = [label_claim(c, verdict, schema) for c in claims]
    mentioned = {c.group for c in claims if c.group}
    omissions = [g for g in verdict.principal if g not in mentioned]
    n_err = sum(lab.verdict != "supported" for lab in labels) + len(omissions)
    return ExplanationLabel(explanation_id=exp.explanation_id, applicant_id=exp.applicant_id,
                            generator=exp.generator, setup=exp.setup, claim_labels=labels,
                            omissions=omissions, n_errors=n_err)
