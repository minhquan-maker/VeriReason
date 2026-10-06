# Contributing

## Branches

`main` is protected. It must always pass `pytest` and `ruff`, and it changes only through pull
requests with at least one review from the other student.

Work happens on short-lived branches cut from `main`. Each branch owns one work package from the
proposal, which keeps diffs small and lets the two of us work in parallel without conflicts. The
"owner" column is only a suggestion.

| Branch | Scope (modules) | Proposal | Weeks | Suggested owner |
|---|---|---|---|---|
| `feat/data-home-credit` | `data/home_credit.py`, `configs/schemas/home_credit.yaml`: finalise reason groups, EXT_SOURCE decision, Kaggle setup | §5.1 | 1 | Quan |
| `feat/generators` | `generate/`, `llm/`: real backends, prompt freeze, cost / latency logging, ≥ 2 open-weight models + 1 proprietary | §5.2 | 1–2 | My |
| `feat/fax-baseline` | `baselines/fax.py` (B4) → **Gate #2** | §5.3, §5.7 | 1–2 | My |
| `feat/verifier` | `verify/`: sensitivity across references, consistency check (Jaccard of Top_k for near-identical applicants), ε calibration, semi-synthetic validation | §4.3–4.6 | 2–5 | Quan |
| `feat/extractor` | `extract/`: LLM extractor prompt, manual annotation subset, extractor P/R | §4.2 | 3–5 | My |
| `feat/recourse-e6` | E6 flip test: parse the action into x′, check f(h(x′)) < τ | §2.2 | 4–5 | Quan |
| `feat/repair` | `repair/` + B5 (RARR-style) + B6 (full regeneration). **Only after the gate passes** | §4.5, §5.3 | 6–8 | both |
| `feat/llm-judge` | B3 LLM-as-judge (with and without model access) | §5.3 | 6–8 | My |
| `exp/<name>` | experiment configs, analysis notebooks and figures; no changes to core modules | §5.4–5.6 | 9–11 | both |
| `paper/<venue>` | LaTeX sources only | — | 12–13 | both |

Rules:
- Branch from an up-to-date `main`: `git switch main && git pull && git switch -c feat/<name>`.
- Keep a branch to one work package. Split it if it grows beyond about 500 changed lines.
- Rebase or merge `main` into your branch before opening a PR, and keep the PR green.
- Tag every experiment that produces a number reported to the supervisor:
  `git tag pilot-<date>`. Also record the config path and the commit in the run's report.

## Before every PR

```bash
ruff check verireason tests
pytest -q
python -m verireason pilot --config configs/german.yaml --synthetic   # end-to-end smoke run
```

## Conventions

- **Config, not code.** Datasets, models, references, generators and thresholds live in
  `configs/*.yaml`, and every number in the paper must trace back to a config and a commit.
- **Stage outputs are JSONL / parquet under `runs/<run_name>/`.** Records are typed in
  `verireason/records.py`. Change a record only together with every stage that reads it.
- **Cache LLM calls.** Generation and extraction skip ids that already exist on disk. Never
  delete a run that cost API money. Copy it instead.
- **No silent model switches.** Record the model id that served each request, as
  `Explanation.llm_model` does. Do not enable automatic fallbacks to other models inside an
  experimental condition.
- **Synthetic data and the `dummy` generator are for tests only.** Their numbers never go into
  reports.
- **Secrets** go in `.env`, which is git-ignored. Datasets go in `data/raw/`, which is also
  git-ignored.
