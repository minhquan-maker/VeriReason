"""Claim extraction and typing (proposal §4.2).

Each explanation is decomposed into typed claims c_j = (g_j, delta_j, mu_j) plus the verbatim
span, so repair can be local. Claims that cannot be mapped to a reason group are kept with
group=None (E5) instead of being dropped; factors that are not model inputs get NOT_IN_MODEL (E1).

Two extractors:
  KeywordExtractor  offline heuristic (aliases + cue words). Cheap baseline and smoke tests.
  LLMExtractor      fixed JSON schema via any LLM backend. Its quality is measured against
                    manual annotation on a subset (proposal §4.2).
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from verireason.data.schema import FeatureSchema
from verireason.llm.base import LLM
from verireason.records import NOT_IN_MODEL, Claim, Explanation

_SENT = re.compile(r"[^.!?]+[.!?]?")
_RAISE = re.compile(r"\b(rais|increas|hurt|harm|against|negative|weak|high(er)? risk|concern|"
                    r"lack|insufficient|too (high|long|large|short|low)|contribut)\w*", re.I)
_LOWER = re.compile(r"\b(lower(ed|s)?|decreas\w*|reduc\w*|help(ed|s)?|positive\w*|favou?rabl\w*|"
                    r"in your favou?r|strength\w*|offset\w*|mitigat\w*)\b", re.I)
_NEUTRAL = re.compile(r"\b(no (significant |meaningful )?(effect|impact)|did not (affect|matter)|"
                      r"neutral)\b", re.I)
_PRINCIPAL = re.compile(r"\b(main|primary|principal|most important|key|major|biggest|largest)\b",
                        re.I)
_MINOR = re.compile(r"\b(minor|slight(ly)?|small(er)?|to a lesser extent)\b", re.I)
_ACTION = re.compile(r"\b(you could|you can|consider|we recommend|we suggest|try to|improv\w* "
                     r"your|reduc\w* your|increas\w* your)\b", re.I)
_GENERIC = re.compile(r"\b(reason|risk|profile|denied|declin|assess)\w*", re.I)


def _locate(text: str, quote: str) -> tuple[int, int]:
    i = text.find(quote)
    if i < 0:
        i = text.lower().find(quote.lower())
    return (i, i + len(quote)) if i >= 0 else (-1, -1)


class KeywordExtractor:
    name = "keyword"

    def __init__(self, schema: FeatureSchema):
        self.schema = schema
        self.patterns = {
            g.name: re.compile(r"\b(" + "|".join(re.escape(a) for a in
                                                 [g.label.lower(), *g.aliases]) + r")", re.I)
            for g in schema.groups.values()
        }
        dropped_terms = [d.replace("_", " ") for d in schema.dropped] + [
            "gender", "sex", "nationality", "foreign", "marital", "race", "religion"]
        self.not_in_model = re.compile(r"\b(" + "|".join(map(re.escape, dropped_terms)) + r")",
                                       re.I)

    def extract(self, exp: Explanation) -> list[Claim]:
        claims: list[Claim] = []
        for m in _SENT.finditer(exp.text):
            sent = m.group(0).strip()
            if not sent:
                continue
            start = exp.text.find(sent, m.start())
            groups = [g for g, p in self.patterns.items() if p.search(sent)]
            if self.not_in_model.search(sent):
                groups.append(NOT_IN_MODEL)
            action = sent if _ACTION.search(sent) else None
            if _NEUTRAL.search(sent):
                direction = "neutral"
            elif _LOWER.search(sent):
                direction = "lowers_risk"
            elif _RAISE.search(sent):
                direction = "raises_risk"
            else:
                direction = "neutral"
            prominence = ("principal" if _PRINCIPAL.search(sent)
                          else "minor" if _MINOR.search(sent) else "contributing")
            if not groups:
                if action or not _GENERIC.search(sent):
                    continue
                groups = [None]                     # non-specific reason (E5)
            for g in groups:
                claims.append(Claim(
                    claim_id=f"{exp.explanation_id}#{len(claims)}",
                    explanation_id=exp.explanation_id, quote=sent, start=start,
                    end=start + len(sent), group=g, direction=direction,
                    prominence=prominence, action=action))
        return claims


EXTRACT_SYSTEM = """You decompose a credit denial explanation into atomic, checkable claims.

Return ONLY a JSON object of the form
{"claims": [{"quote": str, "group": str|null, "direction": "raises_risk"|"lowers_risk"|"neutral",
             "prominence": "principal"|"contributing"|"minor", "action": str|null}]}

Rules:
- One claim per (factor, statement). "quote" must be copied verbatim from the explanation.
- "group" must be one of the reason-group ids listed by the user, or "not_in_model" if the
  factor is not among them, or null if the statement is too generic to map to any factor.
- "direction" is what the text says the factor did to the applicant's risk of default.
- "prominence": "principal" if presented as a main/primary reason or among the first stated
  reasons for denial, "minor" if presented as small or secondary, else "contributing".
- "action": the suggested action if the claim recommends one, else null.
"""


class _RawClaim(BaseModel):
    quote: str
    group: str | None = None
    direction: str = "neutral"
    prominence: str = "contributing"
    action: str | None = None


class LLMExtractor:
    def __init__(self, schema: FeatureSchema, llm: LLM, max_tokens: int = 4096):
        self.schema, self.llm, self.max_tokens = schema, llm, max_tokens
        self.name = f"llm:{llm.name}"

    def _user_prompt(self, exp: Explanation) -> str:
        lines = ["Reason groups (id: label — features — typical wording):"]
        for g in self.schema.groups.values():
            lines.append(f"- {g.name}: {g.label} — {', '.join(g.model_columns)} — "
                         f"{', '.join(g.aliases)}")
        lines += ["", "Explanation:", '"""', exp.text, '"""']
        return "\n".join(lines)

    def extract(self, exp: Explanation) -> list[Claim]:
        c = self.llm.complete(EXTRACT_SYSTEM, self._user_prompt(exp), max_tokens=self.max_tokens,
                              temperature=0.0)
        payload = c.text.strip()
        payload = re.sub(r"^```(json)?|```$", "", payload, flags=re.M).strip()
        try:
            items = json.loads(payload)["claims"]
        except (json.JSONDecodeError, KeyError, TypeError) as err:
            raise ValueError(f"extractor returned invalid JSON for {exp.explanation_id}") from err
        valid = set(self.schema.groups) | {NOT_IN_MODEL}
        claims = []
        for item in items:
            try:
                r = _RawClaim.model_validate(item)
            except ValidationError:
                continue
            group = r.group if r.group in valid else None
            start, end = _locate(exp.text, r.quote)
            claims.append(Claim(
                claim_id=f"{exp.explanation_id}#{len(claims)}", explanation_id=exp.explanation_id,
                quote=r.quote, start=start, end=end, group=group,
                direction=r.direction if r.direction in ("raises_risk", "lowers_risk", "neutral")
                else "neutral",
                prominence=r.prominence if r.prominence in ("principal", "contributing", "minor")
                else "contributing",
                action=r.action))
        return claims
