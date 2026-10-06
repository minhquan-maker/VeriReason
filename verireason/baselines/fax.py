"""B4 — FAX-style verify-and-filter, re-implemented for credit (Kim et al., 2026; arXiv:2605.27879).

Central to the go/no-go decision (Gate #2): does general claim verification already catch the
direction / prominence / omission errors that the intervention verifier finds?

TODO (weeks 1-2, branch `feat/fax-baseline`) — re-read the FAX paper in full before coding:
  1. Draft explanation -> claims (reuse verireason.extract so both systems see the same claims,
     plus FAX's own decomposition prompt as an ablation).
  2. Cross-check each claim with FAX's "inherently faithful tools" — for tabular data this is
     the model's own attributions/predictions exposed as tools; mirror exactly what FAX exposes.
  3. Filter unsupported / contradictory claims, then regenerate the final explanation from the
     surviving claims (FAX filters; it does not repair).
  4. Output: flagged claims (for detection P/R/F1 vs. our labels) and the final text (for
     residual error rate after re-labelling with our verifier).
"""

from __future__ import annotations

from verireason.config import RunConfig


def run(cfg: RunConfig, model_name: str) -> dict:
    raise NotImplementedError("FAX-style baseline (B4) not implemented yet — see module docstring")
