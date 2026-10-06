# Roadmap (proposal §6.2)

Each item names the branch it belongs to (see CONTRIBUTING.md).

## Weeks 1–2: pilot and go/no-go gate

Setting: about 200 denied Home Credit applicants, 3 LLMs, the verifier, and the FAX-style
baseline.

- [x] Scaffold: data → model → verify → generate → extract → label, with offline tests
- [x] Gate #3 check implemented (`validate-lr`)
- [ ] Run German Credit end to end with one real LLM (sanity check on prompts and extractor) — `feat/generators`
- [ ] Home Credit: download, finalise reason groups, decide on EXT_SOURCE — `feat/data-home-credit`
- [ ] Choose 3 generators (small open-weight, large open-weight, proprietary) and freeze prompts — `feat/generators`
- [ ] FAX-style baseline B4 → Gate #2 — `feat/fax-baseline`
- [ ] Hand-check about 30 extracted claims before trusting Gate #1 numbers — `feat/extractor`
- [ ] **Go/no-go meeting with the supervisor** (`runs/<run>/gates/pilot_report.json`)

Gates (proposal §5.7; the thresholds are illustrative):

| Gate | Question | Implemented by | Threshold (placeholder) |
|---|---|---|---|
| #1 | Problem exists: share of explanations with ≥ 1 E2/E3/E4 error | `label` | ≥ 10% |
| #2 | Gap exists: FAX-style misses a substantial share of those errors | `fax` (TODO) | to agree |
| #3 | Verifier is sound: LR direction agreement with the analytic reference | `validate-lr` | ≥ 95% |

If the gate fails, reconsider the direction (for example, explanation drift) rather than forcing
it.

## Weeks 3–5: claim extractor, taxonomy, validated verifier

- [ ] LLM extractor; manual annotation subset; extractor P/R; carry extractor error into the ablations
- [ ] Sensitivity of verdicts across references (knn / marginal / median / approved)
- [ ] ε calibration on a development split; sensitivity to ε and k
- [ ] Consistency check: Jaccard overlap of Top_k for applicants within distance r
- [ ] Semi-synthetic credit model with planted rules (correlated features, interactions)
- [ ] E6 flip test

## Weeks 6–8: selective repair and remaining baselines

- [ ] Span-level regeneration, re-verification loop, templated fallback, preservation diff
- [ ] B3 LLM judge, B5 RARR-style revision, B6 full regeneration

## Weeks 9–11: full experiments

- [ ] All datasets × LLMs × setups × samples
- [ ] Ablations (§5.6): dependency-aware vs. marginal sampling, group vs. single-feature interventions, intervention vs. SHAP-sign, selective repair vs. full regeneration, with and without re-verification
- [ ] Expert annotation of a small XGBoost subset

## Weeks 12–13: writing

- [ ] Re-check related work against full texts (FAX, LLM-credit studies)
