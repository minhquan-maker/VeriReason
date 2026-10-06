"""Stage `generate`: LLM explanations for denied applicants (baseline B0 when evaluated as-is).

Outputs are cached in runs/<run>/explanations/<model>.jsonl; existing explanation ids are skipped
so an interrupted run can be resumed without paying for the same calls twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from verireason.config import RunConfig
from verireason.data.schema import FeatureSchema
from verireason.generate.prompts import SYSTEM, build_prompt
from verireason.io import append_jsonl, read_jsonl
from verireason.llm.base import build_llm
from verireason.models.train import denied_path, load_model
from verireason.records import Explanation


def explanations_path(cfg: RunConfig, model: str):
    return cfg.run_dir / "explanations" / f"{model}.jsonl"


def dummy_explanation(schema: FeatureSchema, group_attr: pd.Series, rng: np.random.Generator
                      ) -> str:
    """OFFLINE SMOKE-TEST generator: templated text from group attributions with injected errors
    (direction flips, a spurious factor, an omission) so that every later stage has something to
    detect. Its error rates mean nothing; never report them."""
    ranked = group_attr.sort_values(ascending=False)
    picks = list(ranked.index[:3])
    if rng.random() < 0.3:                       # omission of the top reason
        picks = picks[1:]
    sentences = []
    for i, g in enumerate(picks):
        label = schema.groups[g].label.lower()
        flip = rng.random() < 0.2
        verb = "lowered" if flip else "raised"
        lead = "The main reason" if i == 0 else "Another reason"
        sentences.append(f"{lead} is your {label}, which {verb} the assessed risk.")
    if rng.random() < 0.3:                       # factor with (likely) no effect
        g = ranked.index[-1]
        sentences.append(f"Your {schema.groups[g].label.lower()} also counted against you.")
    if rng.random() < 0.2:
        sentences.append("Overall, your credit profile was not strong enough.")
    return " ".join(sentences)


def run(cfg: RunConfig, model_name: str, limit: int | None = None) -> dict:
    gcfg = cfg["generation"]
    model = load_model(cfg, model_name)
    X = pd.read_parquet(denied_path(cfg, model_name))
    if limit:
        X = X.head(limit)
    feat_attr = model.feature_attributions(X)
    group_attr = model.group_attributions(X)
    out_path = explanations_path(cfg, model_name)
    done = {e["explanation_id"] for e in read_jsonl(out_path)}
    rng = np.random.default_rng(cfg.seed)
    counts: dict[str, int] = {}

    for gen in gcfg["backends"]:
        llm = None if gen["backend"] == "dummy" else build_llm(
            gen["backend"], gen.get("model"), **gen.get("options", {}))
        for setup in gcfg["setups"]:
            for i, (_, x) in enumerate(X.iterrows()):
                prompt = build_prompt(setup, model.schema, x, float(x["risk"]), model.tau,
                                      feat_attr.iloc[i])
                for s in range(int(gcfg["samples_per_applicant"])):
                    eid = f"{x['applicant_id']}|{model_name}|{gen['name']}|{setup}|{s}"
                    if eid in done:
                        continue
                    if llm is None:
                        text, served, meta = (dummy_explanation(model.schema, group_attr.iloc[i],
                                                                rng), "dummy", {})
                    else:
                        c = llm.complete(SYSTEM, prompt, max_tokens=int(gcfg["max_tokens"]),
                                         temperature=gcfg.get("temperature"))
                        text, served = c.text.strip(), c.model
                        meta = {"usage": c.usage, "stop_reason": c.stop_reason}
                    append_jsonl(out_path, Explanation(
                        explanation_id=eid, applicant_id=str(x["applicant_id"]), model=model_name,
                        generator=gen["name"], llm_model=served, setup=setup, sample=s,
                        prompt=prompt, text=text, meta=meta))
                    counts[gen["name"]] = counts.get(gen["name"], 0) + 1
    return {"new_explanations": counts, "path": str(out_path)}
