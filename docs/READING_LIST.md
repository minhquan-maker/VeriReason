# Reading list and download links

Source: *VeriReason Project Timeline* (10 Oct 2026), section 4. Priority: **P1** read in full in
Week 0; **P2** during Weeks 0–2; **P3** as needed. Download everything with
`bash scripts/fetch_papers.sh` (writes to `papers/`, which is git-ignored).

All arXiv links are preprints unless a venue is shown. Entries marked † are standard references
taken from background knowledge in the timeline: open the link and confirm the version before
citing. The links were **not** opened when this file was written (arXiv was unreachable from the
authoring environment), so check each one resolves to the title shown.

## A. Closest work and LLMs in credit (P1)

| Paper | Venue / date | Abstract | PDF |
|---|---|---|---|
| FAX: Towards Faithful Agentic XAI (Kim et al.) | arXiv, May 2026 | [2605.27879](https://arxiv.org/abs/2605.27879) | [pdf](https://arxiv.org/pdf/2605.27879) |
| Communicating Credit Risk with LLMs (Zandi et al.) | arXiv, Aug 2026 | [2608.17715](https://arxiv.org/abs/2608.17715) | [pdf](https://arxiv.org/pdf/2608.17715) |
| Could LLMs work as Post-hoc Explainability Tools in Credit Risk Models? (Geng et al.) | arXiv, Feb 2026 | [2602.18895](https://arxiv.org/abs/2602.18895) | [pdf](https://arxiv.org/pdf/2602.18895) |
| Interpreting LLMs as Credit Risk Classifiers (AlMarri et al.) | CIKM 2025 FinAI workshop | [2510.25701](https://arxiv.org/abs/2510.25701) | [pdf](https://arxiv.org/pdf/2510.25701) |
| Measuring What LLMs Think They Do: SHAP Faithfulness (AlMarri et al.) | arXiv, Nov 2025 | [2512.00163](https://arxiv.org/abs/2512.00163) | [pdf](https://arxiv.org/pdf/2512.00163) |
| LEXMA (Cheng, Wang, Ghose) | arXiv, Jan 2026 | [2601.04208](https://arxiv.org/abs/2601.04208) | [pdf](https://arxiv.org/pdf/2601.04208) |

## B. Faithfulness and claim verification (P2)

| Paper | Venue / date | Abstract | PDF |
|---|---|---|---|
| FActScore † (Min et al.) | EMNLP 2023 | [2305.14251](https://arxiv.org/abs/2305.14251) | [pdf](https://arxiv.org/pdf/2305.14251) |
| RARR † (Gao et al.) | ACL 2023 | [2210.08726](https://arxiv.org/abs/2210.08726) | [pdf](https://arxiv.org/pdf/2210.08726) |
| Faithfulness Tests for Natural Language Explanations † (Atanasova et al.) | ACL 2023 | [2305.18029](https://arxiv.org/abs/2305.18029) | [pdf](https://arxiv.org/pdf/2305.18029) |
| Verbosity Tradeoffs and the Impact of Scale on the Faithfulness of LLM Self-Explanations (Siegel et al.; current version covers 75 models) | arXiv | [2503.13445](https://arxiv.org/abs/2503.13445) | [pdf](https://arxiv.org/pdf/2503.13445) |
| Walk the Talk? (Matton et al.) | ICLR 2025 | [2504.14150](https://arxiv.org/abs/2504.14150) | [pdf](https://arxiv.org/pdf/2504.14150) |
| Faithful Serum (Alon, Zimerman, Wolf) | ACL 2026 | [2604.14325](https://arxiv.org/abs/2604.14325) | [pdf](https://arxiv.org/pdf/2604.14325) |
| EDCT (Ding et al.) | NeurIPS 2025 | [2510.00047](https://arxiv.org/abs/2510.00047) | [pdf](https://arxiv.org/pdf/2510.00047) |
| TalkToModel (Slack et al.) | Nature Machine Intelligence 2023 | [2207.04154](https://arxiv.org/abs/2207.04154) | [pdf](https://arxiv.org/pdf/2207.04154) |
| A Positive Case for Faithfulness (Mayne et al.), P3 | arXiv, 2026 | [2602.02639](https://arxiv.org/abs/2602.02639) | [pdf](https://arxiv.org/pdf/2602.02639) |

## C. SHAP, references and on-manifold interventions (P2, all †)

| Paper | Venue / date | Abstract | PDF |
|---|---|---|---|
| A Unified Approach to Interpreting Model Predictions (Lundberg, Lee) | NeurIPS 2017 | [1705.07874](https://arxiv.org/abs/1705.07874) | [pdf](https://arxiv.org/pdf/1705.07874) |
| From Local Explanations to Global Understanding with Explainable AI for Trees (Lundberg et al.) | Nature MI 2020 | [1905.04610](https://arxiv.org/abs/1905.04610) | [pdf](https://arxiv.org/pdf/1905.04610) |
| Explaining Individual Predictions When Features Are Dependent (Aas et al.) | Artificial Intelligence 2021 | [1903.10464](https://arxiv.org/abs/1903.10464) | [pdf](https://arxiv.org/pdf/1903.10464) |
| Shapley Explainability on the Data Manifold (Frye et al.) | ICLR 2021 | [2006.01272](https://arxiv.org/abs/2006.01272) | [pdf](https://arxiv.org/pdf/2006.01272) |
| Feature Relevance Quantification in Explainable AI: A Causal Problem (Janzing et al.) | AISTATS 2020 | [1910.13413](https://arxiv.org/abs/1910.13413) | [pdf](https://arxiv.org/pdf/1910.13413) |
| True to the Model or True to the Data? (Chen et al.) | 2020 | [2006.16234](https://arxiv.org/abs/2006.16234) | [pdf](https://arxiv.org/pdf/2006.16234) |

## D. Regulatory basis (P2)

Not on arXiv, so not in the download script:
- Regulation B, 12 CFR 1002.9 and Official Interpretation (comment 9(b)(2)): https://www.consumerfinance.gov/rules-policy/regulations/1002/9/
- CFPB Circulars 2022-03 (withdrawn 12 May 2025; background only) and 2023-03 (check its current status before citing): https://www.consumerfinance.gov/compliance/circulars/

## Reading notes

Use the notes template in the timeline (section 4.5): task and setup; unit of evaluation;
verification method; correlated features (page or section); repair (does it preserve correct
content?); key numbers; difference from VeriReason, with a page reference.
