# Roadmap (proposal §6.2)

Follows *VeriReason Project Timeline* (v. 10 Oct 2026) and *Weeks 1–2 Pilot Plan*. Dates are the
real calendar: Week 0 starts Saturday 10 Oct 2026 and Week 13 ends Sunday 17 Jan 2027.

Each item names the branch it belongs to (see CONTRIBUTING.md for the naming scheme).
Roles: **A** = models and verification (Quan), **B** = LLMs and language (My).

| Phase | Dates | Output |
|---|---|---|
| Week 0: preparation | 10 – 18 Oct 2026 | related-work table, working environment, confirmed venue deadline |
| Weeks 1–2: pilot | 19 Oct – 1 Nov 2026 | pilot report; **go/no-go meeting, proposed Mon 2 Nov** |
| Weeks 3–5: core components | 2 – 22 Nov | validated extractor, taxonomy and verifier |
| Weeks 6–8: repair and baselines | 23 Nov – 13 Dec | end-to-end system |
| Weeks 9–11: experiments | 14 Dec – 3 Jan 2027 | main results (long jobs run over Christmas and New Year) |
| Weeks 12–13: writing | 4 – 17 Jan 2027 | submission draft |

Working rhythm: short daily check-in; weekly supervisor meeting with a one-page note sent in
advance; every reported number tied to a commit hash and seed.

## Week 0 (10 – 18 Oct): preparation

- [ ] Supervisor meeting: direction, target venue and real deadline (FinNLP, ARR for ACL 2027), GPU and API budget, credit-expert reviewer
- [ ] Read FAX in full (arXiv 2605.27879): faithful tools, claim labels, remove vs. rewrite, tabular setup, backbone, code release
- [ ] Read Zandi, Geng, AlMarri (cite 2510.25701 or 2512.00163, whichever is used), LEXMA; rebuild the positioning table with page or section references in place of every "n/r"
- [ ] Background reading, groups B then C (CONTRIBUTING and timeline §4)
- [x] Data terms saved in `docs/licenses/` (German Credit, Home Credit)
- [x] Private repo, CI, environment (`pip install -e ".[dev]"`)
- [ ] Experiment tracker (MLflow or W&B), API keys, GPU host for open-weight LLMs, token budget estimate
- [ ] Roles and meeting schedule agreed (above)

Done when: related-work table has a source for every cell; Home Credit and German Credit are
downloaded; one open-weight and one proprietary LLM can be called from a script; the venue
deadline is in the shared calendar.

Slide corrections found while reading sources (timeline §1.1), to apply before the next
presentation:
- FAX removes *or rewrites* refuted claims. The positioning cell becomes "Partial (remove or rewrite; no preservation constraint)".
- FAX's main benchmark is CRAFTER-XAI-Bench; German Credit is one of three TalkToModel tabular sets.
- Siegel et al.: current version covers 75 models from 13 families.
- AlMarri: two papers (2510.25701, 2512.00163); cite the one used.
- Cite Regulation B as the basis. CFPB Circular 2022-03 was withdrawn on 12 May 2025.

## Weeks 1–2 (19 Oct – 1 Nov): pilot and go/no-go gate

Setting: about 200 denied Home Credit applicants, 3 LLMs × 3 prompts, the minimal verifier, and the
FAX-style baseline B4.

- [x] Scaffold: data → model → verify → generate → extract → label, with offline tests
- [x] Gate #3-a check implemented (`validate-lr`, marginal sampler)
- [ ] **Mon 19 Oct:** finalise `pilot/PREREGISTRATION.md` (draft exists; resolve the "to confirm" items) (G1–G3, ε, k, references, sensitivity grid) and commit it before any result exists
- [x] Home Credit loader with bureau and instalment aggregates, groups `payment_history` and `outstanding_debt` (draft, fixture tests only) — `data/home-credit-reason-groups`
- [ ] Home Credit: download the three files, check aggregates against the column description, finalise reason groups with expert review, decide on EXT_SOURCE and on excluding age / family status / children — `data/home-credit-reason-groups` (A, Tue 20 Oct)
- [ ] Train XGBoost and LR, fix τ and denial rate (A, Wed 21 Oct)
- [ ] TreeSHAP and group SHAP; select about 200 denied applicants (A, Thu 22 Oct) — `exp/pilot-go-no-go`
- [ ] Choose 3 generators (small open-weight ~7–9B, Qwen3-32B or Llama 3.3 70B, one proprietary), freeze P1–P3 — `generate/llm-narrators` (B, Mon 19 – Wed 21 Oct)
- [ ] Dry run on 10 applicants, then full generation over the weekend of 24–25 Oct — `generate/llm-narrators` (B, Thu 22 – Fri 23 Oct)
- [ ] Run German Credit end to end with one real LLM (sanity check on prompts and extractor) — `generate/llm-narrators`
- [ ] Verifier v0 + G3 preliminary (A, Fri 23 Oct); XGBoost verdicts for all 200 (A, Mon 26 Oct)
- [ ] LLM extractor and JSON schema; hand-check about 30 extracted claims (B, Mon 26 Oct) — `decompose/atomic-claims`
- [ ] Verdicts E1–E5 for all claims (A, Tue 27 Oct)
- [ ] FAX-style baseline B4 → Gate #2 (B, Tue 27 – Wed 28 Oct) — `baseline/fax-gap-test`
- [ ] Annotate 50 gold items independently, compute Cohen's κ (both, Wed 28 Oct)
- [ ] Gate table with 95% bootstrap CIs and sensitivity runs (A, Thu 29 Oct); qualitative analysis of 20 error cases (B)
- [ ] Pilot report sent to the supervisor by 18:00 Fri 30 Oct; five-slide summary over the weekend — `exp/pilot-go-no-go`
- [ ] **Go/no-go meeting with the supervisor, Mon 2 Nov** (`runs/<run>/gates/pilot_report.json`); decision recorded in meeting notes

Gates (proposal §5.7; thresholds are illustrative and must be fixed before seeing results):

| Gate | Question | Implemented by | Pass condition |
|---|---|---|---|
| #1 | Problem exists | `label` | ≥ 10% of explanations contain ≥ 1 E2/E3/E4 error. E5 is reported separately, not counted |
| #2 | Gap left by FAX | `baselines/fax.py` (TODO) | B4 leaves ≥ 30% of the verifier's errors undetected or unrepaired. 30% is the team's proposal, not from the slides. Gap = (errors still present + new E4 after B4) / \|L\| |
| #3-a | Implementation correct | `validate-lr` | Marginal sampler, logit-space Δ, no cross-group derived features: sign agreement close to 100%. A mismatch is a bug |
| #3-b | Slide criterion | `validate-lr` (TODO) | Conditional kNN sampler, probability-space Δ: sign agreement ≥ 95%. Disagreement in correlated groups is evidence, not failure |

Required sensitivity: ε ∈ {0.002, 0.005, 0.01}; k ∈ {2, 4}; two references (kNN and marginal);
SHAP-sign comparison. A pass is reliable only if it holds in most configurations.

Decision matrix:

| Outcome | Decision |
|---|---|
| G1, G2, G3 pass | Go: start Weeks 3–5 |
| G1, G3 pass; G2 fails | Adjust: lead with taxonomy, benchmark and the empirical study, or focus on correlated features |
| G1 fails overall, passes for P1 | Adjust: narrow the scope to explanations written without attributions |
| G1 fails for every prompt | Stop this direction; consider recourse (E6) or conversational explanations |
| G3-a fails | Fix the verifier; extend the pilot by up to one week; draw no conclusion from G1 or G2 |
| G3-a passes, G3-b below 95% in correlated groups | Not a failure: record as early evidence for contribution 2 |

If a gate fails, reconsider the direction rather than forcing it.

Pilot report outline (3–5 pages plus appendix): setup (data, groups, models with AUC, τ and denial
rate, LLMs, prompts); gate table with CIs and sensitivity; error distribution by type × LLM ×
prompt; five examples (one each of E2, E3, E4, a case B4 got wrong, a case where SHAP-sign and
intervention disagree); recommended decision and changes to Weeks 3–5.

## Weeks 3–5 (2 – 22 Nov): claim extractor, taxonomy, validated verifier

- [ ] Fix claim schema (group, direction, prominence, optional action, span); 2–3 page annotation guide — `decompose/atomic-claims`
- [ ] Annotate 150–200 explanations, 50 shared for Cohen's κ; extractor P/R/F1 per field (target F1 ≥ 0.85 on group and direction, to agree with the supervisor); carry extractor error into the ablations — `decompose/atomic-claims`
- [ ] Finalise E1–E6 decision rules: ε, k ≤ 4, multi-group claims, vague claims — `verify/soundness-suite`
- [ ] Full verifier: kNN and conditional generative sampler; references conditional / population median / approved; audit record per verdict (Δ_g, rank, samples, seed) — `verify/soundness-suite`
- [ ] Sensitivity of verdicts across references (knn / marginal / median / approved) — `verify/soundness-suite`
- [ ] ε calibration on a development split; sensitivity to ε and k — `verify/soundness-suite`
- [ ] Consistency check: Jaccard overlap of Top_k for applicants within distance r — `verify/soundness-suite`
- [ ] Semi-synthetic credit model with planted rules, correlated features, interactions; verifier recovers planted reasons more often than SHAP-sign for strongly correlated groups — `verify/soundness-suite`
- [ ] SHAP-sign vs. intervention disagreement, split by feature correlation — `verify/soundness-suite`
- [ ] E6 flip test — `classify/recourse-flip-test`

## Weeks 6–8 (23 Nov – 13 Dec): selective repair and remaining baselines

- [ ] Selective repair: fact sheet per flagged claim, regenerate only the offending span (insert a sentence for E4), re-verify, template fallback after a fixed number of attempts (e.g. 3), text diff confirming supported claims are unchanged — `repair/surgical-edits`
- [ ] B3 LLM judge, with and without model access — `baseline/llm-judge`
- [ ] B5 RARR-style revision and B6 full regeneration from verified facts — `repair/surgical-edits`
- [ ] B1 template and B2 SHAP-sign exist; B0 and B4 come from the pilot
- [ ] Normalise every method's output to one format for evaluation
- [ ] End-to-end run on 50 applicants; one command runs any dataset × LLM × prompt × method

## Weeks 9–11 (14 Dec – 3 Jan): full experiments

- [ ] 3–4 LLMs × 3 prompts on Home Credit and German Credit (plus Lending Club or Freddie Mac if time allows), several samples per applicant, fixed decoding
- [ ] Ablations (§5.6): dependency-aware vs. marginal sampling, group vs. single-feature, intervention vs. SHAP-sign, selective vs. full repair with and without re-verification; sensitivity to reference, ε and extractor errors
- [ ] Expert annotation of an XGBoost subset by at least two annotators with credit background; report κ
- [ ] Readability: LLM judge plus a small human check
- [ ] Log cost: model queries, LLM calls, latency
- [ ] Tables and figures answering RQ1–RQ3 with bootstrap CIs

## Weeks 12–13 (4 – 17 Jan): writing

- [ ] Draft in the venue template — `paper/<venue>`
- [ ] Qualitative error analysis and limitations (the reference is a definition, not ground truth; dependence on the extractor; no legal-compliance claim)
- [ ] Final full-text related-work check, including papers posted after October 2026
- [ ] Release code and benchmark within licence terms. Freddie Mac data cannot be redistributed, so release only scripts that rebuild it
