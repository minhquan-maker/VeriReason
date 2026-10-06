"""Small JSONL helpers. Every stage writes append-friendly JSONL so LLM outputs are cached."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: Path, records: Iterable[BaseModel | dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for r in records:
            fh.write((r.model_dump_json() if isinstance(r, BaseModel) else json.dumps(r)) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, record: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write((record.model_dump_json() if isinstance(record, BaseModel) else json.dumps(record))
                 + "\n")


def read_jsonl(path: Path, model: type[T] | None = None) -> Iterator[T | dict]:
    if not path.exists():
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield model.model_validate(obj) if model else obj


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=float)
