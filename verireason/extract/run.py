"""Stage `extract`: explanations -> typed claims (cached per extractor)."""

from __future__ import annotations

from verireason.config import RunConfig
from verireason.data.prepare import load_schema
from verireason.extract.extractor import KeywordExtractor, LLMExtractor
from verireason.generate.run import explanations_path
from verireason.io import append_jsonl, read_jsonl
from verireason.llm.base import build_llm
from verireason.records import Claim, Explanation


def claims_path(cfg: RunConfig, model: str):
    return cfg.run_dir / "claims" / f"{model}.jsonl"


def build_extractor(cfg: RunConfig):
    ecfg = cfg["extraction"]
    schema = load_schema(cfg)
    if ecfg["backend"] == "keyword":
        return KeywordExtractor(schema)
    return LLMExtractor(schema, build_llm(ecfg["backend"], ecfg.get("model")),
                        max_tokens=int(ecfg.get("max_tokens", 4096)))


def run(cfg: RunConfig, model_name: str) -> dict:
    extractor = build_extractor(cfg)
    out = claims_path(cfg, model_name)
    ledger = out.with_suffix(".done.jsonl")      # also records explanations with zero claims
    done = {r["explanation_id"] for r in read_jsonl(ledger)}
    n_exp = n_claims = 0
    for exp in read_jsonl(explanations_path(cfg, model_name), Explanation):
        if exp.explanation_id in done:
            continue
        claims = extractor.extract(exp)
        for c in claims:
            append_jsonl(out, c)
        append_jsonl(ledger, {"explanation_id": exp.explanation_id, "n_claims": len(claims),
                              "extractor": extractor.name})
        n_exp += 1
        n_claims += len(claims)
    return {"extractor": extractor.name, "explanations": n_exp, "claims": n_claims}


def load_claims(cfg: RunConfig, model_name: str) -> dict[str, list[Claim]]:
    by_exp: dict[str, list[Claim]] = {}
    for c in read_jsonl(claims_path(cfg, model_name), Claim):
        by_exp.setdefault(c.explanation_id, []).append(c)
    return by_exp
