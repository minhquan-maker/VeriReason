# Contributing

## Branches

`main` is protected. It must always pass `pytest` and `ruff`, and it changes only through pull
requests with at least one review from the other student.

Work happens on short-lived branches. Each branch owns one work package from the proposal, which
keeps diffs small and lets the two of us work in parallel without conflicts.

Branch names are `<stage>/<slug>`. The stage prefixes follow the paper's pipeline
(**Generate → Decompose → Verify → Classify → Repair**), plus `data/`, `baseline/`, `exp/` and
`paper/`, so the branch list reads like the method section. The "owner" column is only a
suggestion.

| Branch | Stage | Scope (modules) | Proposal | Weeks | Suggested owner |
|---|---|---|---|---|---|
| `data/home-credit-reason-groups` | data | `data/home_credit.py`, `configs/schemas/home_credit.yaml`: finalise reason groups, EXT_SOURCE decision, Kaggle setup | §5.1 | 1 | Quan |
| `generate/llm-narrators` | Generate | `generate/`, `llm/`: real backends, prompt freeze, cost / latency logging, ≥ 2 open-weight models + 1 proprietary | §5.2 | 1–2 | My |
| `baseline/fax-gap-test` | baseline | `baselines/fax.py`: FAX-style verify-and-filter (B4) → **Gate #2** | §5.3, §5.7 | 1–2 | My |
| `exp/pilot-go-no-go` | experiment | pilot configs and runs on about 200 denied Home Credit applicants; gate report for the go/no-go meeting | §5.7 | 1–2 | both |
| `decompose/atomic-claims` | Decompose | `extract/`: LLM extractor prompt, manual annotation subset, extractor P/R | §4.2 | 3–5 | My |
| `verify/soundness-suite` | Verify | `verify/`: sensitivity across references, ε calibration, consistency check (Jaccard of Top_k for near-identical applicants), semi-synthetic planted-rule validation | §4.3–4.6 | 2–5 | Quan |
| `classify/recourse-flip-test` | Classify | `label/`: E6 flip test (parse the action into x′, check f(h(x′)) < τ) | §2.2 | 4–5 | Quan |
| `repair/surgical-edits` | Repair | `repair/`: span-level repair + re-verification loop, B5 (RARR-style), B6 (full regeneration). **Only after the gate passes** | §4.5, §5.3 | 6–8 | both |
| `baseline/llm-judge` | baseline | B3 LLM-as-judge (with and without model access) | §5.3 | 6–8 | My |

Patterns for later branches: `exp/<topic>` for experiment configs, notebooks and figures with
no changes to core modules (for example `exp/ablations`, weeks 9–11); `paper/<venue>` for LaTeX
sources only (for example `paper/finnlp-2027`, weeks 12–13); `fix/<what>` for small fixes.

Rules:
- Start new branches from an up-to-date `main`:
  `git switch main && git pull && git switch -c <stage>/<slug>`.
- Keep a branch to one work package. Split it if it grows beyond about 500 changed lines.
- Merge `main` into your branch before opening a PR, and keep the PR green. Do not rebase a
  branch the other person has already pulled.
- Merge PRs with **Create a merge commit**. Squash and rebase merges rewrite commit ids, which
  breaks any branch that was cut from the PR branch.
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
