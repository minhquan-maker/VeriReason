# Pilot pre-registration (DRAFT)

Status: **draft, not yet committed as final.** The final version must be agreed by both students
and committed on **Mon 19 Oct 2026**, before any pilot result exists. After that commit, thresholds
change only through a dated amendment at the bottom of this file, never by editing the text above it.

Source: *VeriReason Weeks 1–2 Pilot Plan* (Steps 6–13) and proposal §5.7. Thresholds marked
**[to confirm]** are illustrative and need the supervisor's agreement.

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
| Dataset | Home Credit Default Risk (`application_train.csv` plus bureau and instalment aggregates, if added) |
| Models | XGBoost (primary), logistic regression (G3 only) |
| Split | stratified on TARGET, fixed seed. Plan: 70/15/15; current config: 80/20 **[to confirm]** |
| Seed | one seed for numpy, XGBoost, scikit-learn and LLM decoding where supported. Plan: 2026; current config: 13 **[to confirm]** |
| τ | set from the target denial rate on the validation split, same rule for LR. Plan: about 15%; current config: 20% **[to confirm]** |
| Applicants | about 200 denied test applicants, stratified by risk decile and top group, plus 30–40 hard cases (close top-two groups; unfavourable-looking group with negative SHAP) |
| LLMs | one small open-weight (7–9B), one large open-weight (Qwen3-32B or Llama 3.3 70B), one proprietary. Exact model ids recorded at freeze time |
| Prompts | P1 `raw`, P2 `shap`, P3 `shap_groups`. Frozen before the full run; no edits afterwards |
| Decoding | temperature 0 (or lowest allowed), one sample per applicant per configuration |
| Volume | 200 × 3 LLMs × 3 prompts = 1,800 explanations (1,600 if P1 is dropped for the proprietary model) |
| Extractor | strongest available LLM, different from the generator, schema-constrained JSON |
| Excluded from the model | gender; also age, family status and children per the plan **[to confirm vs. current config, which keeps age]** |

## 3. Verifier definition

- Δ_g(x) = f(x) − E[f(h(x̃_g, x_{−g}))], whole group g replaced, derived features recomputed by h.
- Primary reference: kNN (conditional), K = 50 **[to confirm vs. config K = 100]**. Second reference: marginal.
- Direction δ̂_g: `+` if Δ_g > ε, `−` if Δ_g < −ε, `0` otherwise. Primary ε = 0.005.
- G⁺ = groups with δ̂_g = `+`, ranked by signed Δ_g. Principal reasons = top k of G⁺, k = min(4, |G⁺|).
- Sensitivity grid (all reported): ε ∈ {0.002, 0.005, 0.01}; k ∈ {2, 4}; references {kNN, marginal}; SHAP-sign comparison.

## 4. Verdict rules

Each claim receives one label, precedence E1 → E2 → E3.

| Error | Rule |
|---|---|
| E1 | group is not a model input, or direction ≠ neutral and δ̂_g = 0 |
| E2 | δ̂_g ≠ 0 and stated direction ≠ δ̂_g |
| E3 | stated principal but g ∉ top-k(G⁺); or g is top in G⁺ but stated minor. Checked only when direction is correct |
| E4 | a group in top-k(G⁺) appears in no claim (explanation level; report top-1 and top-k separately) |
| E5 | group unmappable. Reported separately, **not** counted in G1 |
| E6 | not evaluated in the pilot |

## 5. Gates

| Gate | Metric | Pass condition |
|---|---|---|
| G1 | share of explanations with ≥ 1 E2, E3 or E4 error, pooled over LLM × prompt | ≥ 10% **[to confirm]** |
| G2 | Gap = (errors in L still present after B4 + new E4 created by B4) / \|L\|, where L is the set of errors the verifier finds in the original explanations | ≥ 30% **[to confirm; the team's proposal, not from the slides]** |
| G3-a | sign agreement with the analytic LR reference, marginal sampler, logit-space Δ, no cross-group derived features, excluding pairs where the reference is near zero | ≥ 99% **[to confirm]** ("close to 100%"). Anything lower is a bug, not a finding |
| G3-b | same agreement, conditional kNN sampler, probability-space Δ | ≥ 95% **[to confirm]** |

Reporting: pooled and per LLM × prompt; 95% bootstrap CI (1,000 resamples over applicants);
E2, E3, E4 separately; G2 by error type plus the share of supported claims B4 removed
(preservation cost). A pass counts as reliable only if it holds in most sensitivity configurations.

## 6. Decision rules

| Outcome | Decision |
|---|---|
| G1, G2, G3 pass | Go: start Weeks 3–5 as planned |
| G1 and G3 pass; G2 fails | Adjust: lead with taxonomy, benchmark and empirical study, or focus on correlated features |
| G1 fails overall but passes for P1 | Adjust: narrow the scope to explanations written without attributions |
| G1 fails for every prompt | Stop this direction; consider recourse (E6) or conversational explanations |
| G3-a fails | Fix the verifier; extend the pilot by up to one week; draw no conclusion from G1 or G2 |
| G3-a passes; G3-b below 95% in correlated groups | Not a failure: record as early evidence for the dependency-aware verifier; decide on G1 and G2 |

## 7. Gold set and agreement

50 explanations sampled at random, balanced across LLMs and prompts, labelled independently by both
students (claims, groups, directions, prominence, E1–E5). Report Cohen's κ on the error labels, then
agree one gold version. Used to measure extractor accuracy and automatic-vs-gold agreement; because
gold labels rely on the same Δ_g, this is not ground truth for the verifier.

## 8. Reproducibility

Every reported number is tied to a config path, a commit hash and a seed. Cache all LLM outputs,
SHAP values and intervention samples; never call an LLM twice on the same input. Tag the pilot run
`pilot-<date>`.

## Amendments

_None._
