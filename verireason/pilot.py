"""Two-week pilot (proposal §5.7) as one command: every stage that exists today, plus a gate report.

Gate #1  problem exists: share of explanations with a direction/prominence/omission error
Gate #2  gap exists: FAX-style baseline misses those errors          (B4 not implemented yet)
Gate #3  verifier sound: LR direction agreement with the analytic reference
Thresholds are illustrative (proposal §5.7) and must be agreed with the supervisor.
"""

from __future__ import annotations

from verireason.config import RunConfig
from verireason.io import write_json

GATE1_MIN = 0.10
GATE3_MIN = 0.95


def run(cfg: RunConfig, synthetic: bool = False, limit: int | None = None) -> dict:
    from verireason.baselines import shap_sign
    from verireason.data import prepare
    from verireason.extract import run as extract
    from verireason.generate import run as generate
    from verireason.label import run as label
    from verireason.models import train
    from verireason.verify import run as verify

    prepare.run(cfg, synthetic=synthetic)
    train.run(cfg)
    verify.run(cfg, "lr", ["marginal", "knn"])
    gate3 = verify.validate_lr(cfg)
    verify.run(cfg, "xgb")
    generate.run(cfg, "xgb", limit)
    extract.run(cfg, "xgb")
    lab = label.run(cfg, "xgb")
    shap_lab = shap_sign.run(cfg, "xgb")

    g1 = lab["ALL"]["share_with_direction_prominence_or_omission_error"]
    g3 = gate3["marginal"]["direction_agreement"]
    report = {
        "synthetic_data": synthetic,
        "generators": [g["name"] for g in cfg["generation"]["backends"]],
        "gate1_problem_exists": {"value": g1, "threshold": GATE1_MIN, "pass": g1 >= GATE1_MIN},
        "gate2_gap_exists": {"value": None, "pass": None, "note": "B4 (FAX-style) not implemented"},
        "gate3_verifier_sound": {"value": g3, "threshold": GATE3_MIN, "pass": g3 >= GATE3_MIN,
                                 "detail": gate3},
        "intervention_vs_shap_sign_any_error": {
            "intervention": lab["ALL"]["share_with_any_error"],
            "shap_sign": shap_lab["ALL"]["share_with_any_error"]},
    }
    if any(g["backend"] == "dummy" for g in cfg["generation"]["backends"]):
        report["warning"] = ("dummy generator in use: error rates are injected by construction "
                             "and are NOT results")
    write_json(cfg.run_dir / "gates" / "pilot_report.json", report)
    return report
