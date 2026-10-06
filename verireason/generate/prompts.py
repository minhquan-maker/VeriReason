"""Generator prompts for the three setups of proposal §5.2.

  raw          applicant features + model decision only
  shap         + per-feature TreeSHAP attributions (logit scale)
  shap_groups  + per-feature attributions + the reason-group dictionary
"""

from __future__ import annotations

import pandas as pd

from verireason.data.schema import FeatureSchema

SYSTEM = (
    "You are a credit officer writing the statement of reasons for an applicant whose credit "
    "application was denied by a scoring model. Write directly to the applicant in plain English, "
    "in one short paragraph of 80 to 160 words. State the principal reasons for the denial, most "
    "important first, and say for each factor whether it increased or decreased the assessed risk. "
    "Do not mention factors that are not supported by the information you are given. You may end "
    "with one concrete suggestion the applicant could act on. Output only the explanation text."
)


def _fmt_value(schema: FeatureSchema, col: str, value) -> str:
    if col in schema.features:
        return schema.describe_value(col, value)
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _feature_lines(schema: FeatureSchema, x: pd.Series) -> list[str]:
    xd = schema.add_derived(x[schema.raw_features].to_frame().T).iloc[0]
    return [f"- {c}: {_fmt_value(schema, c, xd[c])}" for c in schema.model_columns]


def build_prompt(setup: str, schema: FeatureSchema, x: pd.Series, risk: float, tau: float,
                 feature_attr: pd.Series | None = None) -> str:
    parts = [
        "Decision: DENIED.",
        f"Predicted probability of default: {risk:.3f} (applications are denied at {tau:.3f} "
        "or above).",
        "",
        "Applicant information:",
        *_feature_lines(schema, x),
    ]
    if setup in ("shap", "shap_groups"):
        if feature_attr is None:
            raise ValueError(f"setup {setup!r} needs feature attributions")
        parts += ["", "Feature contributions to the model's risk score (TreeSHAP, log-odds; "
                  "positive values increase the predicted risk of default, negative values "
                  "decrease it), largest magnitude first:"]
        for c, v in feature_attr.reindex(feature_attr.abs().sort_values(ascending=False).index
                                         ).items():
            parts.append(f"- {c}: {v:+.3f}")
    if setup == "shap_groups":
        parts += ["", "Reason groups used by the lender (refer to reasons at this level):"]
        for g in schema.groups.values():
            parts.append(f"- {g.label}: {', '.join(g.model_columns)}")
    if setup not in ("raw", "shap", "shap_groups"):
        raise ValueError(f"unknown setup {setup!r}")
    parts += ["", "Write the explanation now."]
    return "\n".join(parts)
