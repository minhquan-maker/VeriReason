"""Shared fixtures. Everything runs offline on SYNTHETIC German-format data."""

from __future__ import annotations

from pathlib import Path

import pytest

from verireason.config import load_config

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    c = load_config(REPO / "configs" / "german.yaml", runs_dir=tmp_path_factory.mktemp("runs"))
    c.raw["run_name"] = "test"
    c.raw["pilot"]["n_denied"] = 30
    c.raw["verifier"]["n_samples"] = 60
    c.raw["generation"]["backends"] = [{"name": "dummy", "backend": "dummy"}]
    c.raw["extraction"]["backend"] = "keyword"
    return c


@pytest.fixture(scope="session")
def trained(cfg):
    from verireason.data import prepare
    from verireason.models import train
    prepare.run(cfg, synthetic=True)
    train.run(cfg)
    return cfg
