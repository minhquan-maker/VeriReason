"""B2 — SHAP-sign verifier (proposal §4.3, §5.3).

Uses group attributions phi_g = sum_{i in g} phi_i (logit scale, the same quantity generators see)
in place of Delta_g, with the same ranking and Top-k rule, so the two verifiers can be compared
claim by claim (disagreement analysis split by feature correlation, §5.4).
"""

from __future__ import annotations

from verireason.config import RunConfig
from verireason.io import write_jsonl
from verireason.label import run as label_run
from verireason.records import ApplicantVerdict
from verireason.verify.intervention import verdicts_from_effects
from verireason.verify.run import load_verdicts


def shap_verdict(v: ApplicantVerdict, eps: float, top_k: int) -> ApplicantVerdict:
    effects = {g: (phi, 0.0, 0) for g, phi in v.shap_group.items()}
    groups, principal = verdicts_from_effects(effects, z=0.0, min_effect=eps, top_k=top_k)
    return v.model_copy(update={"sampler": "shap_sign", "groups": groups, "principal": principal})


def run(cfg: RunConfig, model_name: str, eps: float = 0.01) -> dict:
    """Label all explanations with SHAP-sign verdicts (eps in log-odds units)."""
    base = load_verdicts(cfg, model_name, cfg["verifier"]["samplers"][0])
    top_k = int(cfg["verifier"]["top_k"])
    verdicts = {a: shap_verdict(v, eps, top_k) for a, v in base.items()}
    write_jsonl(cfg.run_dir / "verdicts" / f"{model_name}__shap_sign.jsonl", verdicts.values())
    return label_run.run(cfg, model_name, verdicts=verdicts, tag="shap_sign")
