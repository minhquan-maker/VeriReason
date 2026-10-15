"""End-to-end smoke test with the dummy generator and keyword extractor (no network, no keys)."""

from verireason.cli import main
from verireason.io import read_jsonl


def test_cli_stages_end_to_end(trained, capsys):
    from verireason.extract import run as extract
    from verireason.generate import run as generate
    from verireason.label import run as label
    from verireason.verify import run as verify

    verify.run(trained, "xgb", ["knn"])
    generate.run(trained, "xgb", limit=10)
    extract.run(trained, "xgb")
    summary = label.run(trained, "xgb", "knn")
    assert summary["ALL"]["n_explanations"] == 10 * 3          # 3 prompt setups
    labels = list(read_jsonl(trained.run_dir / "labels" / "xgb__knn.jsonl"))
    assert labels and all("claim_labels" in lab for lab in labels)

    # re-running generate is a no-op thanks to the cache
    assert generate.run(trained, "xgb", limit=10)["new_explanations"] == {}

    assert main(["prepare", "--config", str(trained.path), "--synthetic",
                 "--runs-dir", str(trained.runs_dir / "cli")]) == 0
    assert '"n_reference"' in capsys.readouterr().out


def test_gate1_bootstrap_ci():
    from verireason.label.run import gate1_bootstrap_ci
    from verireason.records import ExplanationLabel

    def lab(app, bad):      # an omission (E4) counts towards Gate #1
        return ExplanationLabel(explanation_id=f"{app}-{bad}", applicant_id=app, generator="g",
                                setup="raw", claim_labels=[], omissions=["g"] if bad else [],
                                n_errors=int(bad))

    labs = [lab(f"a{i}", i < 20) for i in range(100)]          # true share 0.20
    lo, hi = gate1_bootstrap_ci(labs, n_boot=500, seed=1)
    assert lo < 0.20 < hi and 0.10 < lo and hi < 0.32
    assert gate1_bootstrap_ci(labs, n_boot=500, seed=1) == [lo, hi]   # reproducible
