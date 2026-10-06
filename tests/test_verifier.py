import numpy as np
import pandas as pd

from verireason.models.train import denied_path, load_model, load_split
from verireason.verify import run as verify
from verireason.verify.intervention import verdicts_from_effects
from verireason.verify.samplers import build_sampler


def test_verdict_rules_rank_by_signed_effect():
    effects = {"a": (0.30, 0.0, 10), "b": (0.10, 0.0, 10), "c": (-0.50, 0.0, 10),
               "d": (0.001, 0.0, 10), "e": (0.05, 0.0, 10)}
    groups, principal = verdicts_from_effects(effects, z=1.96, min_effect=0.01, top_k=2)
    by = {g.group: g for g in groups}
    assert principal == ["a", "b"]                      # strongly favourable c never a reason
    assert by["c"].direction == "-" and by["c"].role == "minor"
    assert by["d"].direction == "0"
    assert by["e"].role == "contributing" and by["e"].rank == 3


def test_ci_widens_epsilon():
    groups, principal = verdicts_from_effects({"a": (0.05, 0.03, 10)}, z=1.96, min_effect=0.01,
                                              top_k=4)
    assert groups[0].direction == "0" and principal == []


def test_samplers_return_group_columns(trained):
    model = load_model(trained, "xgb")
    ref = load_split(trained, "reference")
    x = ref.iloc[0]
    rng = np.random.default_rng(0)
    for name in ("marginal", "knn", "median", "approved"):
        s = build_sampler(name, model.schema, ref, risk=model.predict_risk(ref), tau=model.tau,
                          k=20)
        alt = s.sample(x, "loan_terms", 15, rng)
        assert list(alt.columns) == model.schema.groups["loan_terms"].features
        assert len(alt) == (1 if s.deterministic else 15)


def test_gate3_lr_agrees_with_analytic_reference(trained):
    verify.run(trained, "lr", ["marginal"])
    report = verify.validate_lr(trained, "lr", ("marginal",))
    assert report["marginal"]["direction_agreement"] >= 0.95


def test_shap_group_attributions_sum_to_margin(trained):
    model = load_model(trained, "xgb")
    X = pd.read_parquet(denied_path(trained, "xgb")).head(5)
    phi = model.group_attributions(X).sum(axis=1).to_numpy()
    import xgboost as xgb
    booster = model.pipeline.named_steps["clf"].get_booster()
    margin = booster.predict(xgb.DMatrix(model.encode(X)), output_margin=True)
    bias = booster.predict(xgb.DMatrix(model.encode(X)), pred_contribs=True)[:, -1]
    assert np.allclose(phi + bias, margin, atol=1e-4)
