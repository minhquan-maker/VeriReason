"""Dependency-aware intervention verifier (proposal §2.1, §4.3).

For applicant x and reason group g:
    Delta_g(x) = f(x) - E[ f(h(x~_g, x_-g)) ],  x~_g ~ Q_g(. | x_-g)
estimated from S samples with its Monte Carlo standard error. Verdicts:
    eps_g          = max(z * SE_g, min_effect)
    delta_hat_g    = '+' if Delta_g > eps_g, '-' if Delta_g < -eps_g, else '0'
    G+             = {g : delta_hat_g = '+'}, ranked by signed Delta_g (largest first)
    principal      = Top_k(G+); other G+ members are contributing; the rest are minor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from verireason.models.scoring import ScoringModel
from verireason.records import ApplicantVerdict, GroupVerdict
from verireason.verify.samplers import Sampler


def group_effects(model: ScoringModel, x: pd.Series, sampler: Sampler, n_samples: int,
                  rng: np.random.Generator) -> tuple[float, dict[str, tuple[float, float, int]]]:
    """Return f(x) and {group: (Delta_g, SE_g, S_used)}; one batched model call per applicant."""
    feats = model.schema.raw_features
    base_row = x[feats].to_frame().T
    blocks, owners = [], []
    for g in model.schema.groups:
        alt = sampler.sample(x, g, n_samples, rng)
        block = pd.concat([base_row] * len(alt), ignore_index=True)
        for c in alt.columns:
            block[c] = alt[c].to_numpy()
        blocks.append(block)
        owners += [g] * len(alt)
    batch = pd.concat([base_row, *blocks], ignore_index=True)
    for c in model.schema.numeric_columns:
        if c in batch:
            batch[c] = pd.to_numeric(batch[c])
    risk = model.predict_risk(batch)
    base, alt_risk = float(risk[0]), risk[1:]
    owners = np.array(owners)

    out = {}
    for g in model.schema.groups:
        diffs = base - alt_risk[owners == g]
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        out[g] = (float(diffs.mean()), se, len(diffs))
    return base, out


def verdicts_from_effects(effects: dict[str, tuple[float, float, int]], *, z: float,
                          min_effect: float, top_k: int) -> tuple[list[GroupVerdict], list[str]]:
    rows = []
    for g, (delta, se, n) in effects.items():
        eps = max(z * se, min_effect)
        direction = "+" if delta > eps else "-" if delta < -eps else "0"
        rows.append(dict(group=g, delta=delta, se=se, ci_low=delta - z * se,
                         ci_high=delta + z * se, eps=eps, direction=direction, n_samples=n))
    g_plus = sorted((r for r in rows if r["direction"] == "+"), key=lambda r: -r["delta"])
    rank = {r["group"]: i + 1 for i, r in enumerate(g_plus)}
    verdicts = []
    for r in rows:
        rk = rank.get(r["group"])
        role = "minor" if rk is None else ("principal" if rk <= top_k else "contributing")
        verdicts.append(GroupVerdict(rank=rk, role=role, **r))
    principal = [r["group"] for r in g_plus[:top_k]]
    return verdicts, principal


def verify_applicant(model: ScoringModel, x: pd.Series, sampler: Sampler, *, n_samples: int,
                     z: float, min_effect: float, top_k: int, rng: np.random.Generator,
                     shap_group: dict[str, float] | None = None) -> ApplicantVerdict:
    base, effects = group_effects(model, x, sampler, n_samples, rng)
    verdicts, principal = verdicts_from_effects(effects, z=z, min_effect=min_effect, top_k=top_k)
    return ApplicantVerdict(applicant_id=str(x["applicant_id"]), model=model.name,
                            sampler=sampler.name, risk=base, tau=model.tau, top_k=top_k,
                            groups=verdicts, principal=principal, shap_group=shap_group or {})
