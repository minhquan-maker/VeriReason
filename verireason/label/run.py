"""Stage `label`: apply E1-E6 to every explanation and report error rates (RQ1, Gate #1)."""

from __future__ import annotations

from collections import Counter, defaultdict

from verireason.config import RunConfig
from verireason.data.prepare import load_schema
from verireason.extract.run import load_claims
from verireason.generate.run import explanations_path
from verireason.io import read_jsonl, write_json, write_jsonl
from verireason.label.rules import label_explanation
from verireason.records import ApplicantVerdict, Explanation, ExplanationLabel
from verireason.verify.run import load_verdicts

ERROR_TYPES = ("E1", "E2", "E3", "E4", "E5", "E6")
GATE1_TYPES = ("E2", "E3", "E4")      # direction, prominence, omission (proposal §5.7)


def summarize(labels: list[ExplanationLabel]) -> dict:
    by_cond: dict[str, list[ExplanationLabel]] = defaultdict(list)
    for lab in labels:
        by_cond[f"{lab.generator}|{lab.setup}"].append(lab)
        by_cond["ALL"].append(lab)
    out = {}
    for cond, labs in sorted(by_cond.items()):
        claim_counts = Counter(cl.verdict for lab in labs for cl in lab.claim_labels)
        n_claims = sum(claim_counts.values())
        gate1 = [lab for lab in labs
                 if lab.omissions or any(cl.verdict in GATE1_TYPES for cl in lab.claim_labels)]
        out[cond] = {
            "n_explanations": len(labs),
            "n_claims": n_claims,
            "claim_error_rate": {t: claim_counts.get(t, 0) / n_claims if n_claims else 0.0
                                 for t in ERROR_TYPES if t != "E4"},
            "omission_rate_per_explanation": sum(len(lab.omissions) for lab in labs) / len(labs),
            "share_with_any_error": sum(lab.has_error for lab in labs) / len(labs),
            "share_with_direction_prominence_or_omission_error": len(gate1) / len(labs),
        }
    return out


def run(cfg: RunConfig, model_name: str, sampler: str | None = None,
        verdicts: dict[str, ApplicantVerdict] | None = None, tag: str | None = None) -> dict:
    sampler = sampler or cfg["verifier"]["samplers"][0]
    schema = load_schema(cfg)
    verdicts = verdicts if verdicts is not None else load_verdicts(cfg, model_name, sampler)
    claims = load_claims(cfg, model_name)
    labels = []
    for exp in read_jsonl(explanations_path(cfg, model_name), Explanation):
        v = verdicts.get(exp.applicant_id)
        if v is None:
            continue
        labels.append(label_explanation(exp, claims.get(exp.explanation_id, []), v, schema))
    tag = tag or sampler
    write_jsonl(cfg.run_dir / "labels" / f"{model_name}__{tag}.jsonl", labels)
    summary = summarize(labels)
    write_json(cfg.run_dir / "labels" / f"{model_name}__{tag}__summary.json", summary)
    return summary
