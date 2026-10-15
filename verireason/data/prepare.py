"""Stage `prepare`: load a dataset, keep schema columns, split into train (reference R),
validation and test."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from verireason.config import RunConfig
from verireason.data import german, home_credit
from verireason.data.schema import FeatureSchema
from verireason.io import write_json


def load_schema(cfg: RunConfig) -> FeatureSchema:
    return FeatureSchema.from_yaml(cfg.resolve(cfg["schema"]))


def load_dataset(cfg: RunConfig, synthetic: bool = False) -> pd.DataFrame:
    ds = cfg["dataset"]
    if synthetic:
        if ds["name"] != "german":
            raise ValueError("--synthetic is only implemented for the german schema")
        return german.synthesize(n=1000, seed=cfg.seed)
    if ds["name"] == "german":
        return german.load(cfg.resolve(ds["path"]), url=ds.get("url"))
    if ds["name"] == "home_credit":
        def opt(key):
            return cfg.resolve(ds[key]) if ds.get(key) else None
        return home_credit.load(cfg.resolve(ds["path"]), max_rows=ds.get("max_rows"), seed=cfg.seed,
                                bureau_path=opt("bureau_path"),
                                installments_path=opt("installments_path"))
    raise ValueError(f"unknown dataset {ds['name']!r}")


def run(cfg: RunConfig, synthetic: bool = False) -> dict:
    schema = load_schema(cfg)
    df = load_dataset(cfg, synthetic=synthetic)
    df = df.rename(columns={schema.target: "default"}) if schema.target != "default" else df
    keep = ["applicant_id", *schema.raw_features, "default"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing schema columns: {missing}")
    df = df[keep]

    # 70/15/15 by default (pilot plan, Step 4): train (= reference population R), validation
    # (sets tau), test (denied applicants are drawn from here). Stratified on the target.
    test_size, val_size = float(cfg["dataset"]["test_size"]), float(cfg["dataset"]["val_size"])
    rest, test = train_test_split(df, test_size=test_size, stratify=df["default"],
                                  random_state=cfg.seed)
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size),
                                  stratify=rest["default"], random_state=cfg.seed)
    out = cfg.run_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    train.reset_index(drop=True).to_parquet(out / "reference.parquet")
    val.reset_index(drop=True).to_parquet(out / "validation.parquet")
    test.reset_index(drop=True).to_parquet(out / "test.parquet")
    info = {"n_reference": len(train), "n_validation": len(val), "n_test": len(test),
            "default_rate": float(df["default"].mean()), "synthetic": synthetic,
            "dataset": cfg["dataset"]["name"]}
    write_json(out / "info.json", info)
    return info
