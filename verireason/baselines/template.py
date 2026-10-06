"""B1 — deterministic reason-code template from SHAP groups (industry-style baseline).

Near-zero error by construction against SHAP groups; shows the accuracy/readability trade-off.
"""

from __future__ import annotations

from verireason.data.schema import FeatureSchema
from verireason.records import ApplicantVerdict


def template_explanation(v: ApplicantVerdict, schema: FeatureSchema, top_k: int = 4,
                         eps: float = 0.0) -> str:
    risky = sorted(((g, phi) for g, phi in v.shap_group.items() if phi > eps),
                   key=lambda t: -t[1])[:top_k]
    if not risky:
        return ("Your application was denied based on the overall assessment of the "
                "information provided.")
    reasons = "; ".join(f"{i + 1}. {schema.groups[g].label}" for i, (g, _) in enumerate(risky))
    return f"Principal reasons for the decision, most important first: {reasons}."
