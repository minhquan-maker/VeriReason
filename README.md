# VeriReason

**Intervention-based claim verification and repair for LLM-generated credit denial explanations.**

Research code for the VeriReason proposal (HCMUT, Faculty of Computer Science and Engineering).
Students: Nguyen Minh Quan, Tran Hai My. Supervisor: Assoc. Prof. Quan Thanh Tho.

The pipeline checks each claim in an LLM-written denial explanation against the scoring model's
behaviour. It uses dependency-aware group interventions, labels errors with a taxonomy (E1–E6)
tied to the principal-reasons requirement, and (after the go/no-go gate) repairs only the failing
claims.

```
Train model → Select denied → Generate → Extract claims → Verify → (Repair) → Evaluate
   train         train         generate     extract         verify     repair      label
```

> **Status: pilot scaffold (Week 0 of the 13-week plan, 10 Oct 2026 – 17 Jan 2027).**
> Pilot weeks 1–2 run 19 Oct – 1 Nov; go/no-go meeting proposed Mon 2 Nov 2026.
> See [docs/ROADMAP.md](docs/ROADMAP.md) for the dated plan, gates and decision matrix.
>
> *Implemented and tested:*
> - data, models and τ
> - intervention verifier (4 references)
> - LR analytic check (Gates #3-a marginal and #3-b kNN)
> - generator prompts P1–P3 (`raw`, `shap`, `shap_groups`) and backends
> - keyword and LLM claim extractors
> - E1–E5 rules (E6 only for immutable groups)
> - SHAP-sign baseline (B2)
> - template baseline (B1)
> - pilot gate report (Gates #1 with bootstrap CI, #3-a, #3-b; Gate #2 threshold set, value pending B4)
>
> *Not yet implemented:*
> - FAX-style baseline (B4, Gate #2)
> - the ε/k sensitivity grid
> - LLM-judge baseline (B3)
> - selective repair and baselines B5/B6
> - the E6 flip test
> - consistency check
> - semi-synthetic validation
>
> *Aligned with the Weeks 1–2 pilot plan (15 Oct 2026):*
> - 70/15/15 split (`dataset.test_size`, `dataset.val_size`), τ set on the validation split for
>   about 15% denials, seed 2026, kNN K = 50 with S = 50 replacement rows, temperature 0 and one
>   sample per applicant. All three configs use the same values.
> - Gender, age, family status, children and family size are excluded from the Home Credit models;
>   a claim that mentions any of them is E1 (`sensitive_aliases` in the schema). German Credit keeps
>   `age`, because it is used only as a code check.
>
> *Still open before 19 Oct:*
> - Home Credit now has bureau and instalment aggregates (`payment_history` and
>   `outstanding_debt` groups, `configs/home_credit.yaml`). It is tested on small fixtures only; the
>   groups still need expert review and a run on the real files.
> - Applicant selection is a random draw of `pilot.n_denied`; the plan's stratification by risk
>   decile and top group, plus 30–40 hard cases, is not implemented yet.
> - XGBoost has no early stopping (fixed `n_estimators`); the plan suggests early stopping on validation.
> - `pilot/PREREGISTRATION.md` is final for the thresholds and settings listed there, to be committed
>   on 19 Oct. Items under "Open at freeze" in that file still need a value.
> - There is no `pilot/config.yaml`: the pilot is configured by `configs/*.yaml` (`pilot.n_denied`,
>   `decision`, `verifier`, `generation`). `pilot/` holds only the pre-registration and, later, the report.

---

## 1. Setup

Requirements:
- Python ≥ 3.10.
- A CPU is enough for the verifier. A GPU is needed only if you self-host open-weight generators.

```bash
git clone https://github.com/minhquan-maker/VeriReason.git
cd VeriReason
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + tests
pip install -e ".[dev,llm]"        # + Claude / OpenAI-compatible clients for real generators
cp .env.example .env               # then fill in only the keys you use
pytest -q                          # offline tests, a few seconds
# macOS: XGBoost needs OpenMP -> brew install libomp
```

### Data

| Dataset | Role (proposal §5.1) | How to get it |
|---|---|---|
| Statlog German Credit (UCI, id 144) | fast iteration, FAX comparability | downloaded automatically from UCI on first `prepare`, or place `german.data` in `data/raw/`. Do **not** use the Kaggle `uciml/german-credit` copy: it is a reduced re-encoding with different columns |
| Home Credit Default Risk | primary | needs a Kaggle account and accepting the competition rules (the API returns 403 until you do). Pilot files: `application_train.csv`, `bureau.csv`, `installments_payments.csv`; also keep `HomeCredit_columns_description.csv` to check column meanings. `kaggle competitions download -c home-credit-default-risk -p data/raw/home-credit`, then unzip there |
| FICO HELOC (Explainable ML Challenge) | replication (optional) | request access through the FICO form and accept its licence; copies on Kaggle and OpenML exist, but use the FICO source. Special negative codes (−7, −8, −9) need handling. Not wired yet |
| Lending Club (2007–2018) | secondary replication | no longer published by Lending Club; the Kaggle copy (`wordsforthewise/lending-club`) is user-uploaded, so check its licence. Post-origination columns leak the outcome and must be removed. Not wired yet |
| Freddie Mac Single-Family Loan-Level | secondary replication, if time allows | register at Clarity Data Intelligence. Free for academic use; redistribution of the data or derived products that could recreate it is not allowed. Not wired yet |

`data/raw/` and `runs/` are git-ignored. Never commit datasets, API keys or LLM outputs.

### LLM backends

Generators and the extractor are configured in the YAML file (`generation.backends`,
`extraction`):

| backend | used for | credentials |
|---|---|---|
| `dummy` | offline smoke tests. **Errors are injected on purpose, so its rates are not results.** | none |
| `keyword` (extraction only) | offline heuristic extractor | none |
| `anthropic` | proprietary generator / extractor (default `claude-opus-5-5`) | `ANTHROPIC_API_KEY` |
| `openai_compat` | open-weight models via vLLM, Together, Groq, … | `OPENAI_BASE_URL`, `OPENAI_API_KEY` |

To serve a small open-weight model locally:
`vllm serve Qwen/Qwen2.5-7B-Instruct`, then set `OPENAI_BASE_URL=http://localhost:8000/v1`.

---

## 2. Running

Every stage reads one config and writes to `runs/<run_name>/`. Stages are resumable: generation
and extraction skip ids already on disk, so an interrupted run never pays twice.

```bash
# offline smoke run of the whole pilot on SYNTHETIC German-format data (~1 min)
python -m verireason pilot --config configs/german.yaml --synthetic

# real German Credit, stage by stage
python -m verireason prepare      --config configs/german.yaml
python -m verireason train        --config configs/german.yaml
python -m verireason verify       --config configs/german.yaml --model lr --sampler marginal --sampler knn
python -m verireason validate-lr  --config configs/german.yaml          # Gate #3
python -m verireason verify       --config configs/german.yaml --model xgb
python -m verireason generate     --config configs/german.yaml --model xgb --limit 20
python -m verireason extract      --config configs/german.yaml --model xgb
python -m verireason label        --config configs/german.yaml --model xgb   # RQ1, Gate #1
python -m verireason shap-sign    --config configs/german.yaml --model xgb   # baseline B2
```

| Stage | Output under `runs/<run>/` | Proposal |
|---|---|---|
| `prepare` | `data/reference.parquet` (reference population R), `data/test.parquet` | §5.1 |
| `train` | `models/{xgb,lr}.joblib`, `models/report.json` (AUC, τ), `data/denied_<model>.parquet` | §5.1, §5.5 |
| `verify` | `verdicts/<model>__<sampler>.jsonl`: one audit record per applicant (Δg, CI, ε, direction, rank, role, φg) | §2.1, §4.3, §4.4 |
| `validate-lr` | `gates/gate3_lr_agreement.json` | §4.6, §5.7 |
| `generate` | `explanations/<model>.jsonl` (prompt, text, served model id, usage) | §5.2 |
| `extract` | `claims/<model>.jsonl`: typed claims with character spans | §4.2 |
| `label` | `labels/<model>__<sampler>.jsonl` + `__summary.json` (error rates per generator × setup) | §2.2, §5.4 |
| `shap-sign` | the same, using SHAP-sign verdicts (B2) | §5.3 |
| `pilot` | `gates/pilot_report.json` (Gates #1–#3) | §5.7 |

---

## 3. How the code maps to the proposal

```
verireason/
  data/schema.py          feature schema, derived features h, reason groups G (validated)
  data/{german,home_credit}.py  loaders (Home Credit: bureau / instalment aggregates; synthetic German-format data for tests)
  models/scoring.py       f(x) on raw features; owns h and the encoder; TreeSHAP group attributions
  models/train.py         XGBoost (primary) + logistic regression (validation), τ, denied set
  verify/samplers.py      Q_g: knn (conditional), marginal, median, approved references
  verify/intervention.py  Δg(x) with Monte Carlo SE → δ̂g, G+, Top_k(G+), roles
  verify/analytic.py      LR logit-space reference w_gᵀ(z_g − E_R z_g)  (Gate #3)
  generate/prompts.py     3 prompt setups: raw | shap | shap_groups
  extract/extractor.py    typed claims (g, δ, μ, action, span): keyword + LLM extractors
  label/rules.py          taxonomy E1–E6 as deterministic rules
  baselines/              B1 template, B2 SHAP-sign, B4 FAX-style (stub; FAX removes or rewrites refuted claims)
  repair/selective.py     selective repair (fact sheet + preservation check; loop TODO)
  pilot.py                go/no-go gate report
configs/                  experiment configs + schemas/ (reason-group definitions)
tests/                    offline tests (synthetic data, fake LLM)
```

Key definitions (proposal §2.1), as implemented:

- **Δg(x) = f(x) − E[f(h(x̃g, x₋g))]** on the probability scale.
  - Estimated from S samples (`verifier.n_samples`).
  - Whole groups are replaced, and derived ratios are recomputed by `ScoringModel` on every call.
- **ε_g = max(z·SE_g, min_effect)**. A group counts as non-zero only if its CI excludes 0 *and*
  it passes the minimum effect size.
- **G+** is ranked by **signed** Δg, so a strongly favourable group is never a reason for denial.
  Principal reasons are the first k groups in this ranking (k ≤ 4).
- Each derived feature must sit in the same group as all of its sources. The schema loader
  enforces this, so group interventions stay self-contained and the LR analytic reference stays
  exact.

---

## 4. Open decisions (agree with the supervisor before the pilot)

1. **Denial policy.** Neither dataset contains real decisions. τ is set so that
   `decision.denial_rate` of the validation split is denied, and this must be stated in the paper.
2. **`EXT_SOURCE_1/2/3` (Home Credit).** These opaque bureau scores dominate the model. For now
   they are kept as a non-actionable `external_scores` group. The alternatives are to drop them
   or to discuss them as a limitation.
3. **Protected attributes.** `personal_status_sex`, `foreign_worker` and `CODE_GENDER` are
   dropped from the models. For Home Credit, age, family status, children and family size are
   also excluded (pilot plan), so that any LLM mention is an unambiguous E1. Decided 15 Oct 2026.
4. **ε and k.** `min_effect` is a placeholder. Fix it on a development split and report
   sensitivity to ε and k.
5. **Attribution scale.** TreeSHAP values given to generators are in log-odds, while Δg is in
   probability. This mismatch is a possible error source to analyse (§5.1).
6. **Gate thresholds.** The values in `pilot.py` (Gate #1 ≥ 10%, Gate #2 ≥ 30%, Gate #3-a ≥ 99%,
   Gate #3-b ≥ 95%) are illustrative.
   The 30% for Gate #2 is the team's proposal. All three are written into `pilot/PREREGISTRATION.md`, which
   must be committed before any result exists.
7. **FAX re-implementation.** Re-read the full FAX text before building B4 (check the authors'
   GitHub first). Its backbone is Qwen3-32B, so using it as the large open-weight generator makes
   the comparison fair. All positioning-table entries marked "n/r" must be checked against full texts.
8. **Bureau aggregates.** `configs/home_credit.yaml` uses them. `configs/home_credit_app_only.yaml`
   is the fallback without the secondary tables. Check each aggregate against
   `HomeCredit_columns_description.csv` and have the group table reviewed by someone with credit
   knowledge. `previous_application.csv` (prior refusals, approval rate) is optional and unused.

## 5. Planning documents

The project timeline and the Weeks 1–2 pilot plan are the source of the dates, gates and branch
plan. [docs/ROADMAP.md](docs/ROADMAP.md) mirrors them as a checklist.

## 6. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch layout, conventions and the check that must
pass before every PR.
