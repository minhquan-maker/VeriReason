"""Analytic reference for logistic regression (proposal §4.6) — the Gate #3 check.

For f(x) = sigmoid(w'z(x) + b), the group effect in logit space under a marginal (interventional)
reference is   a_g(x) = w_g' (z_g(x) - E_R[z_g]),
so the true direction and ranking of each group are known exactly. The sampled verifier works on
the probability scale (and optionally with a conditional sampler), so it is not closed-form; we
measure how often its verdicts agree with a_g and attribute disagreements to sampling,
nonlinearity, and the conditional reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from verireason.models.scoring import ScoringModel
from verireason.records import ApplicantVerdict


def analytic_group_effects(model: ScoringModel, X: pd.DataFrame) -> pd.DataFrame:
    if model.kind != "logistic":
        raise ValueError("analytic reference only exists for the logistic model")
    return model.group_attributions(X)


def agreement_report(model: ScoringModel, X: pd.DataFrame, verdicts: list[ApplicantVerdict],
                     analytic_tol: float = 1e-3) -> dict:
    """Direction agreement on groups the verifier calls non-zero, plus Top-k overlap."""
    A = analytic_group_effects(model, X)
    A.index = X["applicant_id"].astype(str).to_numpy()
    n_dir = n_agree = n_zero = 0
    jacc = []
    for v in verdicts:
        a = A.loc[v.applicant_id]
        for gv in v.groups:
            if gv.direction == "0":
                n_zero += 1
                continue
            if abs(a[gv.group]) < analytic_tol:
                continue
            n_dir += 1
            n_agree += int(np.sign(a[gv.group]) == (1 if gv.direction == "+" else -1))
        pos = a[a > analytic_tol].sort_values(ascending=False)
        ref_top = set(pos.index[: v.top_k])
        got = set(v.principal)
        if ref_top or got:
            jacc.append(len(ref_top & got) / len(ref_top | got))
    return {
        "n_applicants": len(verdicts),
        "n_nonzero_group_verdicts": n_dir,
        "n_zero_group_verdicts": n_zero,
        "direction_agreement": n_agree / n_dir if n_dir else float("nan"),
        "topk_jaccard_mean": float(np.mean(jacc)) if jacc else float("nan"),
    }
