"""Selective claim-level repair (proposal §4.5) — weeks 6-8, after the go/no-go gate.

Algorithm per explanation:
  1. For each flagged claim build a fact sheet from verified evidence (group, verified direction,
     rank, Delta_g and CI)                                                 -> `fact_sheet` (done)
  2. Regenerate ONLY the offending span (or insert one sentence for an E4 omission), instructing
     the LLM to leave the surrounding text unchanged                       -> TODO
  3. Re-extract and re-verify the edited claim; up to `max_attempts`, then fall back to a
     templated sentence built from the fact sheet                          -> TODO
  4. Preservation check by text diff: supported claims must remain verbatim, and the number of
     edited spans must equal the number of verified errors                 -> `preserved` (done)
Baselines to compare: B5 RARR-style revision, B6 full regeneration with verified facts.
"""

from __future__ import annotations

from verireason.data.schema import FeatureSchema
from verireason.records import ApplicantVerdict, Claim, ClaimLabel

_DIR_TEXT = {"+": "increased the assessed risk", "-": "decreased the assessed risk",
             "0": "had no meaningful effect on the assessed risk"}


def fact_sheet(label: ClaimLabel, verdict: ApplicantVerdict, schema: FeatureSchema) -> dict:
    g = label.group
    if g is None or g not in schema.groups:
        return {"error": label.verdict, "instruction": "remove or replace with a specific, "
                "verified reason"}
    gv = verdict.by_group()[g]
    return {
        "error": label.verdict,
        "group": schema.groups[g].label,
        "verified_effect": _DIR_TEXT[gv.direction],
        "role": gv.role,
        "rank_among_risk_increasing": gv.rank,
        "delta": round(gv.delta, 4),
        "ci": [round(gv.ci_low, 4), round(gv.ci_high, 4)],
    }


def templated_sentence(fs: dict) -> str:
    if "group" not in fs:
        return ""
    prefix = "A principal reason is" if fs["role"] == "principal" else "A further factor is"
    return f"{prefix} your {fs['group'].lower()}, which {fs['verified_effect']}."


def preserved(repaired: str, supported: list[Claim]) -> bool:
    """Every originally supported claim span must appear verbatim in the repaired text."""
    return all(c.quote in repaired for c in supported if c.start >= 0)
