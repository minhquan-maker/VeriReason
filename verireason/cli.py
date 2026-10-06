"""Command-line entry point: `python -m verireason <stage> --config configs/german.yaml`.

Stages follow the experimental protocol (proposal §5.5):
  prepare -> train -> verify -> validate-lr (Gate #3) -> generate -> extract -> label (Gate #1)
  -> shap-sign (B2) -> fax (B4, Gate #2) -> repair (after the gate)
"""

from __future__ import annotations

import argparse
import json
import sys

from verireason.config import load_config


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=float))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verireason")
    ap.add_argument("stage", choices=["prepare", "train", "verify", "validate-lr", "generate",
                                      "extract", "label", "shap-sign", "fax", "pilot"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-dir", default=None, help="override output root (default: runs/)")
    ap.add_argument("--model", default="xgb", help="scoring model name from the config")
    ap.add_argument("--sampler", action="append", help="verifier reference(s); repeatable")
    ap.add_argument("--synthetic", action="store_true",
                    help="prepare: use SYNTHETIC German-format data (offline smoke tests only)")
    ap.add_argument("--limit", type=int, default=None, help="generate: first N applicants only")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    cfg = load_config(args.config, runs_dir=args.runs_dir)
    if args.stage == "prepare":
        from verireason.data.prepare import run
        _print(run(cfg, synthetic=args.synthetic))
    elif args.stage == "train":
        from verireason.models.train import run
        _print(run(cfg))
    elif args.stage == "verify":
        from verireason.verify.run import run
        _print(run(cfg, args.model, args.sampler))
    elif args.stage == "validate-lr":
        from verireason.verify.run import validate_lr
        _print(validate_lr(cfg, "lr", tuple(args.sampler or ("marginal", "knn"))))
    elif args.stage == "generate":
        from verireason.generate.run import run
        _print(run(cfg, args.model, args.limit))
    elif args.stage == "extract":
        from verireason.extract.run import run
        _print(run(cfg, args.model))
    elif args.stage == "label":
        from verireason.label.run import run
        _print(run(cfg, args.model, (args.sampler or [None])[0]))
    elif args.stage == "shap-sign":
        from verireason.baselines.shap_sign import run
        _print(run(cfg, args.model))
    elif args.stage == "fax":
        from verireason.baselines.fax import run
        _print(run(cfg, args.model))
    elif args.stage == "pilot":
        from verireason.pilot import run
        _print(run(cfg, synthetic=args.synthetic, limit=args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
