# Slide corrections (VeriReason Proposed Slide, 29 slides)

The slide deck is a LaTeX Beamer PDF and its `.tex` source is not in this repository, so the slides
are not edited here. Below is the exact replacement text per slide. Sources: *Project Timeline*
section 1.1 (five corrections) and sections 2, 5 and 6 (dates, datasets, risks), and the Weeks 1–2
pilot plan (gates). Paste into the `.tex` and recompile.

Nothing below was checked against the full papers. Items marked **(verify)** must be confirmed after
the Week 0 full-text reading of FAX and the LLM-in-credit papers (see `docs/READING_LIST.md`).

## Slide 4: Why Claim-Level Verification?

Requirement bullet:

> **Requirement.** A denial must state the *specific, principal reasons* (Regulation B, 12 CFR 1002.9
> and its Official Interpretation).

Optional footnote: *CFPB Circular 2022-03 was withdrawn on 12 May 2025; circulars are background
only.* The slide does not mention a circular now, so add the footnote only if one is cited.

## Slide 12: LLM Explanations in Credit and Finance

Replace the AlMarri bullet (cite the paper actually used; two exist, arXiv 2510.25701 and
2512.00163):

> **AlMarri et al. (2025):** when an LLM explains its own default predictions, its feature-impact
> statements often disagree with SHAP on financial tabular tasks. The LLM is the classifier here,
> unlike our setting, where it explains another model.

## Slide 13: Faithfulness and Claim Verification

Replace the Counterfactual-tests and FAX bullets:

> **Counterfactual tests:** Atanasova et al. (2023), Siegel et al. (75 models, 13 families), Walk the
> Talk (2025); Faithful Serum (2026) adds attribution-guided steering.
>
> **FAX (2026):** decomposes drafts into claims and tests them with tools that query the target
> model; refuted claims are removed or rewritten, inconclusive ones are not presented as verified.
> Evaluated mainly outside credit (main benchmark: an RL environment); German Credit is one of three
> TalkToModel tabular sets, and the authors caution that such benchmarks can conflate task accuracy
> with model-specific faithfulness. *Closest prior work.*

Takeaway (replace):

> FAX is a main baseline. It has no notion of principal reasons, omission or prominence, and no
> constraint to preserve supported claims. Missing: a credit-specific taxonomy, interventions for
> correlated features, and repair that preserves supported claims; the pilot tests whether FAX suffices.

## Slide 14: Positioning

Two FAX cells change:

| Row | FAX (old) | FAX (new) |
|---|---|---|
| Credit-specific setting | Partial (German Credit) | Partial (German Credit only; mainly non-credit benchmarks) |
| Selective repair | Partial (filtering) | Partial (remove or rewrite; no preservation constraint) |

LaTeX:

```latex
% Credit-specific setting row, FAX column:
Partial (German Credit; mainly non-credit)
% Selective repair row, FAX column:
Partial (remove or rewrite; no preservation)
```

Footnote: keep "n/r = not reported in an abstract-level reading, not proven absent" until every n/r
cell is replaced with a page or section reference **(verify)**. State the difference from FAX as:
verbatim preservation of supported claims, and insertion for omitted principal reasons (E4).

## Slide 21: Datasets, Models and Generators

Generators line: the PDF text shows "34 LLMs" because the en dash is lost; write it as `3--4 LLMs`.

Datasets table (replace):

| Dataset | Role | Note |
|---|---|---|
| Home Credit Default Risk | Primary (pilot) | Linked tables; Kaggle account and acceptance of the competition rules required; do not publish raw data |
| German Credit (UCI, id 144) | Comparability | CC BY 4.0; used by prior work and as one of FAX's three tabular sets; added in Weeks 3–5 |
| FICO HELOC | Comparability (optional) | Request access via the FICO form; negative special codes need handling |
| Lending Club / Freddie Mac | Secondary replication | Lending Club copy on Kaggle is user-uploaded, check its licence and remove leaking columns; Freddie Mac needs registration and cannot be redistributed |

## Slide 24: Pilot and Go/No-Go Gate

Replace the three bullets with the pre-registered gates (`pilot/PREREGISTRATION.md`):

> **G1 Problem exists:** at least 10% of explanations contain an E2, E3 or E4 error.
> **G2 Gap exists:** FAX-style baseline B4 leaves at least 30% of the verifier's errors undetected or
> unrepaired (threshold proposed by the team).
> **G3 Verifier is sound:** direction verdicts match the analytic LR reference: close to 100% with the
> marginal sampler (implementation check), at least 95% with the conditional kNN sampler.

Keep "Thresholds are illustrative" and the "If the gate fails" box.

## Slide 26: Current Status and Next Steps

> Initial idea from a targeted literature search; scaffold code and a pre-registration are in place.
> Next: full-text review of FAX and the LLM-credit studies (Week 0), then the two-week pilot
> (19 Oct to 1 Nov) and the go/no-go meeting on 2 Nov, then refine or reconsider the direction.

## Slide 28: Timeline

Use real dates and add Week 0:

| Weeks | Dates | Work | Output / decision |
|---|---|---|---|
| 0 | 10 to 18 Oct 2026 | Supervisor meeting, full-text reading, data and licences, environment | Related-work table, working environment |
| 1 to 2 | 19 Oct to 1 Nov | Data, reason groups, generators, pilot with FAX-style baseline | Go/no-go meeting, Mon 2 Nov |
| 3 to 5 | 2 to 22 Nov | Extractor, taxonomy, verifier, LR and semi-synthetic validation | Validated verifier |
| 6 to 8 | 23 Nov to 13 Dec | Selective repair, remaining baselines | End-to-end system |
| 9 to 11 | 14 Dec to 3 Jan 2027 | Full experiments, ablations, expert annotation | Main results |
| 12 to 13 | 4 to 17 Jan | Writing, figures, full-text related-work check | Submission draft |

Keep: "Target: a finance-NLP workshop (e.g. FinNLP); ACL 2027 via ARR as a stretch goal (dates to be
confirmed)". Add the fallback from the timeline: if no end-to-end run exists by Week 8, drop Lending
Club, Freddie Mac, random forest and E6.

## Slide 27: Risks and Mitigations (optional additions)

- *Delay in Weeks 3 to 5:* simplify the schema, keep only the kNN sampler, postpone the generative sampler.
- *No expert annotators:* ask the supervisor in Week 0; fallback is finance students with a written guide.
- *New competing paper:* weekly arXiv check (cs.CL, cs.LG, q-fin.RM); update the positioning table.
