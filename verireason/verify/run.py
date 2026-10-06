"""Stage `verify` (audit records per applicant) and `validate-lr` (Gate #3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from verireason.config import RunConfig
from verireason.io import read_jsonl, write_json, write_jsonl
from verireason.models.train import denied_path, load_model, load_split
from verireason.records import ApplicantVerdict
from verireason.verify.analytic import agreement_report
from verireason.verify.intervention import verify_applicant
from verireason.verify.samplers import build_sampler


def verdicts_path(cfg: RunConfig, model: str, sampler: str):
    return cfg.run_dir / "verdicts" / f"{model}__{sampler}.jsonl"


def load_verdicts(cfg: RunConfig, model: str, sampler: str) -> dict[str, ApplicantVerdict]:
    return {v.applicant_id: v for v in read_jsonl(verdicts_path(cfg, model, sampler),
                                                   ApplicantVerdict)}


def run(cfg: RunConfig, model_name: str, samplers: list[str] | None = None) -> dict:
    vcfg = cfg["verifier"]
    samplers = samplers or vcfg["samplers"]
    model = load_model(cfg, model_name)
    ref = load_split(cfg, "reference")
    denied = denied_path(cfg, model_name)
    X = pd.read_parquet(denied)
    shap = model.group_attributions(X)
    summary = {}
    for s_name in samplers:
        sampler = build_sampler(s_name, model.schema, ref, risk=model.predict_risk(ref),
                                tau=model.tau, k=int(vcfg.get("knn_k", 50)),
                                max_rows=vcfg.get("reference_max_rows"), seed=cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        out = []
        for i, (_, x) in enumerate(X.iterrows()):
            out.append(verify_applicant(
                model, x, sampler, n_samples=int(vcfg["n_samples"]), z=float(vcfg["z"]),
                min_effect=float(vcfg["min_effect"]), top_k=int(vcfg["top_k"]), rng=rng,
                shap_group={k: float(v) for k, v in shap.iloc[i].items()}))
        write_jsonl(verdicts_path(cfg, model_name, s_name), out)
        summary[s_name] = {
            "n_applicants": len(out),
            "mean_principal_reasons": float(np.mean([len(v.principal) for v in out])),
            "share_no_principal_reason": float(np.mean([not v.principal for v in out])),
        }
    write_json(cfg.run_dir / "verdicts" / f"{model_name}__summary.json", summary)
    return summary


def validate_lr(cfg: RunConfig, model_name: str = "lr",
                samplers: tuple[str, ...] = ("marginal", "knn")) -> dict:
    """Gate #3: on logistic regression, verifier directions should agree with the analytic
    logit-space reference almost always (proposal §5.7, e.g. >= 95%)."""
    model = load_model(cfg, model_name)
    X = pd.read_parquet(denied_path(cfg, model_name))
    report = {}
    for s in samplers:
        verdicts = list(load_verdicts(cfg, model_name, s).values())
        if not verdicts:
            raise FileNotFoundError(f"run `verify --model {model_name}` with sampler {s} first")
        report[s] = agreement_report(model, X, verdicts)
    write_json(cfg.run_dir / "gates" / "gate3_lr_agreement.json", report)
    return report
