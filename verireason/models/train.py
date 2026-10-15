"""Stage `train`: fit each configured scoring model, set tau on validation, select denied
test applicants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from verireason.config import RunConfig
from verireason.data.prepare import load_schema
from verireason.io import write_json
from verireason.models.scoring import ScoringModel


def model_path(cfg: RunConfig, name: str):
    return cfg.run_dir / "models" / f"{name}.joblib"


def load_model(cfg: RunConfig, name: str) -> ScoringModel:
    return ScoringModel.load(model_path(cfg, name))


def load_split(cfg: RunConfig, split: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.run_dir / "data" / f"{split}.parquet")


def denied_path(cfg: RunConfig, name: str):
    return cfg.run_dir / "data" / f"denied_{name}.parquet"


def run(cfg: RunConfig) -> dict:
    schema = load_schema(cfg)
    ref, val, test = (load_split(cfg, s) for s in ("reference", "validation", "test"))
    denial_rate = float(cfg["decision"]["denial_rate"])
    n_denied = int(cfg["pilot"]["n_denied"])
    report: dict = {}

    for name, spec in cfg["models"].items():
        model = ScoringModel.build(name, spec["type"], schema, spec.get("params", {}), cfg.seed)
        model.fit(ref, ref["default"].to_numpy())
        if model.kind == "logistic":
            model.meta["encoded_reference_mean"] = model.encode(ref).mean(axis=0).tolist()

        p_val, p_test = model.predict_risk(val), model.predict_risk(test)
        # tau: simulated denial policy, set on the validation split so that `denial_rate` of it is
        # denied (neither dataset has real decisions; stated in the paper). Same rule for all.
        model.tau = float(np.quantile(p_val, 1 - denial_rate))
        model.meta.update(auc_validation=float(roc_auc_score(val["default"], p_val)),
                          auc_test=float(roc_auc_score(test["default"], p_test)),
                          auc_reference=float(roc_auc_score(ref["default"],
                                                            model.predict_risk(ref))))
        model.save(model_path(cfg, name))

        denied = test.assign(risk=p_test)[p_test >= model.tau]
        denied = denied.sample(n=min(n_denied, len(denied)), random_state=cfg.seed)
        denied.reset_index(drop=True).to_parquet(denied_path(cfg, name))
        report[name] = {"tau": model.tau, "auc_validation": model.meta["auc_validation"],
                        "auc_test": model.meta["auc_test"],
                        "denial_rate_validation": float((p_val >= model.tau).mean()),
                        "denial_rate_test": float((p_test >= model.tau).mean()),
                        "auc_reference": model.meta["auc_reference"], "n_denied": len(denied)}

    write_json(cfg.run_dir / "models" / "report.json", report)
    return report
