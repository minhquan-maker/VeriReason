"""Typed records exchanged between pipeline stages (stored as JSONL under runs/<run_name>/).

Notation follows proposal §2.1:
  Delta_g(x) = f(x) - E[f(h(x~_g, x_-g))]   (positive => applicant's values in g raise risk)
  verified direction  delta_hat_g in {+, 0, -}
  G+ = {g : Delta_g > eps}, ranked by signed Delta_g; principal reasons = Top_k(G+)
  claim c_j = (g_j, delta_j, mu_j) + answer span
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Sign = Literal["+", "0", "-"]
Role = Literal["principal", "contributing", "minor"]
StatedDirection = Literal["raises_risk", "lowers_risk", "neutral"]
ErrorType = Literal["E1", "E2", "E3", "E4", "E5", "E6"]

NOT_IN_MODEL = "not_in_model"   # extractor group label for factors that are not model inputs


class GroupVerdict(BaseModel):
    group: str
    delta: float                    # Delta_g(x), probability scale
    se: float                       # Monte Carlo standard error
    ci_low: float
    ci_high: float
    eps: float                      # tolerance actually used for this group
    direction: Sign                 # delta_hat_g
    rank: int | None = None         # 1-based rank inside G+ (None if not in G+)
    role: Role
    n_samples: int


class ApplicantVerdict(BaseModel):
    """Audit record for one applicant under one scoring model and one reference sampler."""

    applicant_id: str
    model: str
    sampler: str
    risk: float                     # f(x)
    tau: float
    top_k: int
    groups: list[GroupVerdict]
    principal: list[str]            # Top_k(G+)
    shap_group: dict[str, float] = Field(default_factory=dict)   # phi_g, logit scale (baseline B2)

    def by_group(self) -> dict[str, GroupVerdict]:
        return {g.group: g for g in self.groups}


class Explanation(BaseModel):
    explanation_id: str
    applicant_id: str
    model: str                      # scoring model the explanation is about
    generator: str                  # generator name from the config
    llm_model: str | None = None    # provider model id actually requested
    setup: Literal["raw", "shap", "shap_groups"]
    sample: int
    prompt: str
    text: str
    meta: dict = Field(default_factory=dict)


class Claim(BaseModel):
    claim_id: str
    explanation_id: str
    quote: str                      # verbatim span from the explanation
    start: int = -1                 # char offsets into Explanation.text (-1 if not located)
    end: int = -1
    group: str | None = None        # reason group, NOT_IN_MODEL, or None (non-specific)
    direction: StatedDirection = "neutral"
    prominence: Role = "contributing"
    action: str | None = None       # recommended action (recourse claim), if any


class ClaimLabel(BaseModel):
    claim_id: str
    explanation_id: str
    group: str | None
    verdict: Literal["supported", "E1", "E2", "E3", "E5", "E6"]
    reason: str
    verified_direction: Sign | None = None
    verified_role: Role | None = None


class ExplanationLabel(BaseModel):
    explanation_id: str
    applicant_id: str
    generator: str
    setup: str
    claim_labels: list[ClaimLabel]
    omissions: list[str]            # E4: groups in Top_k(G+) never mentioned
    n_errors: int

    @property
    def has_error(self) -> bool:
        return self.n_errors > 0
