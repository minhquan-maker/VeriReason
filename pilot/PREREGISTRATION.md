# Pilot pre-registration

Status: **final text, frozen 15 Oct 2026**, before any pilot result exists (no Home Credit model has
been trained and no LLM explanation has been generated). After this commit, thresholds and settings
change only through a dated amendment at the bottom of this file, never by editing the text above it.

Source: *VeriReason Weeks 1–2 Pilot Plan* (Steps 3–13), proposal §5.7, and the repository state at
the commit that contains this file. Settings are read from `configs/home_credit.yaml` (primary),
`configs/schemas/home_credit.yaml` and `configs/german.yaml` (code check only); the config values
below were aligned with the plan on the same date.

Supervisor confirmation of the G1 and G2 thresholds: **pending** (meeting not held yet). If the
supervisor asks for different values, record them as an amendment before the first result.

## 1. Question

Before committing Weeks 3–13, the pilot answers three yes/no questions:

| ID | Question |
|---|---|
| G1 | Do LLM explanations of credit denials contain direction, prominence or omission errors? |
| G2 | Does a FAX-style baseline (B4) leave a substantial share of those errors? |
| G3 | Is the intervention verifier sound? |

The pilot is not meant to produce publishable results. Synthetic data and the `dummy` generator never
count towards any gate.

## 2. Setting

| Item | Value |
|---|---|
| Dataset | Home Credit Default Risk: `application_train.csv` plus bureau and instalment aggregates (`configs/home_credit.yaml`). German Credit is used only to check the code |
| Models | XGBoost (primary; 500 trees, depth 5, learning rate 0.05, subsample 0.8, colsample 0.8, no early stopping), logistic regression (G3 only, C = 0.5) |
| Split | 70/15/15 train/validation/test, stratified on TARGET. Train is the reference population R |
| Seed | 2026 for numpy, XGBoost, scikit-learn and the samplers; LLM decoding uses it where the API supports it |
| τ | set on the validation split so that 15% of it is denied (`decision.denial_rate`); the same rule for LR with its own τ. The denial policy is simulated and stated as such in any report |
| Applicants | 200 denied test applicants (f(x) ≥ τ): stratified by risk decile and top group with at least 15 per common top group, plus 30–40 hard cases (close top-two groups; unfavourable-looking group with negative SHAP). Selected, and the id list saved, before any explanation is generated |
| Excluded from the model | gender, age, family status, number of children, family size. A claim that mentions any of them is E1 (`sensitive_aliases` in the schema) |
| Groups | `configs/schemas/home_credit.yaml`; reviewed by a person with credit expertise (name and date to be recorded, see §9) |
| LLMs | one small open-weight (7–9B), one large open-weight (Qwen3-32B or Llama 3.3 70B), one proprietary. Exact model ids recorded at freeze time (§9) |
| Prompts | P1 `raw`, P2 `shap`, P3 `shap_groups`. Frozen after the 10-applicant dry run; no edits afterwards |
| Decoding | temperature 0 (or the lowest the API allows), one sample per applicant per configuration |
| Volume | 200 × 3 LLMs × 3 prompts = 1,800 explanations (1,600 if P1 is dropped for the proprietary model) |
| Extractor | strongest available LLM, different from the generator, schema-constrained JSON. The `keyword` extractor is for offline tests only |
| Extractor check | one member reads 30 explanations against the extractor output; if more than 15% are wrong on group or direction, fix the prompt before continuing |

## 3. Verifier definition

- Δ_g(x) = f(x) − E[f(h(x̃_g, x_{−g}))], whole group g replaced, derived features recomputed by h, on
  the probability scale. S = 50 replacement rows per group.
- Primary reference: kNN (conditional), K = 50, index over x_{−g} standardised; R is a random subsample
  of 20,000 training applicants (`verifier.reference_max_rows`) to keep the index small. Other
  references reported: marginal, approved (marginal over approved training applicants), population median.
- Direction δ̂_g: `+` if Δ_g > ε_g, `−` if Δ_g < −ε_g, `0` otherwise, with ε_g = max(1.96 · SE_g, m),
  m = 0.005 (proposal §2.1: Monte Carlo standard error plus a minimum effect size).
- G⁺ = groups with δ̂_g = `+`, ranked by signed Δ_g. Principal reasons = top k of G⁺, k = min(4, |G⁺|).
- Sensitivity grid (all reported): m ∈ {0.002, 0.005, 0.01}; k ∈ {2, 4}; references {kNN, approved,
  marginal}; the literal fixed-ε rule of the pilot plan (ε = m, no SE term, i.e. z = 0); SHAP-sign (B2)
  as a comparison column.
- A group is a risk reason only through G⁺: ranking by signed Δ_g means a favourable group is never
  counted as a reason for denial.
- Δ_g is an intervention on the model input, not a causal effect on the borrower. It is a defined and
  auditable reference, not ground truth.

## 4. Verdict rules

Each claim receives one label. Precedence in the implementation: E5, then E1 for factors that are not
model inputs, then E6 for recourse claims (not evaluated in the pilot), then E1, E2, E3.

| Error | Rule |
|---|---|
| E1 | group is not a model input (including age, gender, family status, children), or stated direction ≠ neutral and δ̂_g = 0 |
| E2 | δ̂_g ≠ 0 and stated direction ≠ δ̂_g (a "neutral" statement about a non-zero group counts) |
| E3 | stated principal but g ∉ top-k(G⁺); or g in top-k(G⁺) but stated minor. Checked only when the direction is correct |
| E4 | a group in top-k(G⁺) appears in no claim (explanation level; report top-1 and top-k separately) |
| E5 | group unmappable. Reported separately, **not** counted in G1 |
| E6 | not evaluated in the pilot |

Every verdict is stored with Δ_g, its standard error and interval, rank in G⁺, sampler, reference,
ε, k and seed.

## 5. Gates

| Gate | Metric | Pass condition |
|---|---|---|
| G1 | share of explanations with ≥ 1 E2, E3 or E4 error, pooled over LLM × prompt | ≥ 10% |
| G2 | Gap = (errors in L still present after B4 + new E4 created by B4) / \|L\|, where L is the set of errors the verifier finds in the original explanations | ≥ 30% (the team's operationalisation of "a substantial share"; the slides give no number) |
| G3-a | sign agreement with the analytic LR reference: marginal sampler, logit-space Δ, no cross-group derived features, excluding pairs where the reference is near zero (\|a_g\| < 1e-3 in logit units) | ≥ 99%. Anything lower is a bug, not a finding |
| G3-b | same agreement, conditional kNN sampler, probability-space Δ | ≥ 95% |

Reporting: pooled and per LLM × prompt; 95% bootstrap CI (1,000 resamples over applicants); E2, E3, E4
separately, plus E1 (sensitive attributes) and E5 separately; G2 by error type plus the share of
supported claims B4 removed (preservation cost). A gate counts as reliably passed only if it holds in
most configurations of the sensitivity grid.

## 6. Decision rules

| Outcome | Decision |
|---|---|
| G1, G2, G3 pass | Go: start Weeks 3–5 as planned |
| G1 and G3 pass; G2 fails | Adjust: lead with taxonomy, benchmark and empirical study, or focus on correlated features |
| G1 fails overall but passes for P1 | Adjust: narrow the scope to explanations written without attributions |
| G1 fails for every prompt | Stop this direction; consider recourse (E6) or conversational explanations |
| G3-a fails | Fix the verifier; extend the pilot by up to one week; draw no conclusion from G1 or G2 |
| G3-a passes; G3-b below 95% in correlated groups | Not a failure: record as early evidence for the dependency-aware verifier; decide on G1 and G2 |

## 7. B4 (FAX-style baseline)

Re-implemented for credit from Kim et al. (2026, arXiv:2605.27879): identify claims, analyse the
evidence behind each, plan at most three verification calls per claim, run them with faithful tools
(`edit_state`, `get_counterfactual`; the SHAP table is a noisy tool only), label claims corroborated,
refuted or inconclusive; keep corroborated, remove or rewrite refuted, omit or hedge inconclusive.
Backbone Qwen3-32B if available. B4 runs on 200 applicants for one configuration (large LLM × P2),
extended if time allows. The exact tool set is fixed after reading the full FAX text and recorded in
§9 before B4 is run; a change after that is an amendment.

## 8. Gold set and agreement

50 explanations sampled at random, balanced across LLMs and prompts, labelled independently by both
students (claims, groups, directions, prominence, E1–E5). Report Cohen's κ on the error labels, then
agree one gold version. Used to measure extractor accuracy and automatic-vs-gold agreement; because
gold labels rely on the same Δ_g, this is not ground truth for the verifier.

## 9. Open at freeze

These need a value, not a decision on the method. Each is filled by a dated amendment before the
step that uses it.

| Item | Needed before | Value |
|---|---|---|
| Model ids and versions of the three generators and the extractor | 21 Oct (first generation) | _pending_ |
| Definition of "close top-two groups" for hard cases (proposed: rank-2 Δ_g ≥ 80% of rank-1 Δ_g) | 22 Oct (selection) | _pending_ |
| Reviewer of the group table (name, date) | 22 Oct | _pending_ |
| Final B4 tool set and prompts | 27 Oct | _pending_ |
| Supervisor confirmation of G1 ≥ 10% and G2 ≥ 30% | 2 Nov meeting at the latest | _pending_ |

## 10. Reproducibility

Every reported number is tied to a config path, a commit hash and a seed. Cache all LLM outputs,
SHAP values and intervention samples; never call an LLM twice on the same input. Tag the pilot run
`pilot-<date>`.

## Known deviations from the pilot plan

Recorded now so they are not discovered after the results.

- No early stopping for XGBoost; fixed `n_estimators`.
- kNN index over a 20,000-row subsample of R, not the full training set.
- Second reference is `approved` (marginal over approved applicants), not kNN restricted to approved
  applicants.
- ε follows the proposal (standard error plus floor); the plan's literal fixed ε is run as a sensitivity
  configuration.
- Sensitivity grid, applicant stratification, B4 and the gold-set tooling are not implemented yet; this
  file fixes what they must do, not that they exist.

## Amendments

_None._
