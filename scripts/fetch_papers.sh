#!/usr/bin/env bash
# Download the arXiv papers of the reading list (docs/READING_LIST.md) into papers/ (git-ignored).
# Run on a machine with internet access: bash scripts/fetch_papers.sh
set -euo pipefail
OUT="${1:-papers}"
mkdir -p "$OUT"
while read -r id name; do
  [ -z "$id" ] && continue
  f="$OUT/${name}_${id}.pdf"
  [ -s "$f" ] && { echo "have $f"; continue; }
  curl -fsSL "https://arxiv.org/pdf/${id}" -o "$f" && echo "ok   $f" || echo "FAIL $id ($name)"
  sleep 3   # be polite to arXiv
done <<'LIST'
2605.27879 P1_FAX
2608.17715 P1_Zandi_credit_risk_LLM
2602.18895 P1_Geng_LLM_posthoc_credit
2510.25701 P1_AlMarri_LLM_credit_classifiers
2512.00163 P1_AlMarri_SHAP_faithfulness
2601.04208 P1_LEXMA
2305.14251 P2_FActScore
2210.08726 P2_RARR
2305.18029 P2_Atanasova_faithfulness_tests
2503.13445 P2_Siegel_verbosity_faithfulness
2504.14150 P2_Matton_Walk_the_Talk
2604.14325 P2_Faithful_Serum
2510.00047 P2_EDCT
2207.04154 P2_TalkToModel
2602.02639 P3_Mayne_positive_case
1705.07874 P2_Lundberg_SHAP
1905.04610 P2_Lundberg_TreeSHAP
1903.10464 P2_Aas_dependent_features
2006.01272 P2_Frye_data_manifold
1910.13413 P2_Janzing_causal_relevance
2006.16234 P2_Chen_true_to_model_or_data
LIST
