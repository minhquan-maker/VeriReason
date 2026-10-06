"""Experiment config loading and run-directory layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


@dataclass
class RunConfig:
    raw: dict[str, Any]
    path: Path
    root: Path          # paths in the config are relative to this (the repo root by default)
    runs_dir: Path

    @property
    def name(self) -> str:
        return self.raw["run_name"]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    @property
    def run_dir(self) -> Path:
        d = self.runs_dir / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve(self, path: str | Path) -> Path:
        return _resolve(path, self.root)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def load_config(path: str | Path, root: str | Path | None = None,
                runs_dir: str | Path | None = None) -> RunConfig:
    path = Path(path).resolve()
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    root_p = Path(root).resolve() if root else REPO_ROOT
    runs_p = Path(runs_dir).resolve() if runs_dir else root_p / "runs"
    return RunConfig(raw=raw, path=path, root=root_p, runs_dir=runs_p)
