#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${ROOT}/code"
PYTHON="${PYTHON:-python3}"
GOLD_DIR="${WORKSPACE}/annotation_qc/outputs/formal_qc_20260718"
V3_OUT_DIR="${V3_OUT_DIR:-${WORKSPACE}/grammar_kg/kg_v3_compiled}"

"${PYTHON}" -c 'import sys; expected=(3,12,13); actual=sys.version_info[:3]; assert actual == expected, f"ExecKG v3 requires CPython {expected}, got {actual}"'
"${PYTHON}" "${ROOT}/scripts/verify_release.py"

# Deterministically rebuild the complete 572-point compiler layer. This step
# needs no handbook text and is separate from the 37-label empirical benchmark.
"${PYTHON}" "${WORKSPACE}/grammar_kg/tools/compile_standard_inventory_v3.py" \
  --catalog "${WORKSPACE}/grammar_kg/kg/standard_grammar_points_gf0025_v2.csv" \
  --constraints "${WORKSPACE}/grammar_kg/kg/constraints_v0_1.json" \
  --corrections "${WORKSPACE}/grammar_kg/kg/standard_catalog_corrections_v3.csv" \
  --out-dir "${V3_OUT_DIR}"

required=(
  "gold_main_by_filename_majority_annotator2_20260718.csv"
  "gold_evidence_by_filename_majority_annotator2_20260718.csv"
  "gold_slot_span_by_filename_majority_annotator2_20260718.csv"
)

for filename in "${required[@]}"; do
  if [[ ! -f "${GOLD_DIR}/${filename}" ]]; then
    printf 'Missing authorized file: %s\n' "${GOLD_DIR}/${filename}" >&2
    printf 'See DATA_CARD.md for the copyright-aware access procedure.\n' >&2
    exit 2
  fi
done

export ASGD_ROOT="${WORKSPACE}"

"${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/run_grouped_kg_experiments.py"
"${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/analyze_diagnostic_regimes.py"
"${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/run_grouped_auxiliary_experiments.py"
"${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/analyze_grouped_significance.py"
"${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/analyze_trace_state_validity.py"
"${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/analyze_selective_review.py"
"${PYTHON}" "${WORKSPACE}/grammar_kg/tools/build_candidate_evidence_kg_v1.py"
"${PYTHON}" "${WORKSPACE}/annotation_qc/tools/audit_annotation_agreement_20260718.py"

if [[ "${RUN_MACBERT:-0}" == "1" ]]; then
  "${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/run_macbert_grouped_baseline.py" \
    --seeds 13 42 2027 --epochs 5 --batch-size 16 --learning-rate 2e-5 \
    --weight-decay 0.01 --warmup-ratio 0.10 --max-length 128
fi

if [[ "${RUN_EFFICIENCY:-0}" == "1" ]]; then
  "${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/benchmark_model_efficiency.py"
fi

if [[ "${RUN_LLM_ANALYSIS:-0}" == "1" ]]; then
  "${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/apply_llm_kg_governor.py"
  "${PYTHON}" "${WORKSPACE}/asgd_fast_experiments/scripts/analyze_openai_llm_comparison.py"
fi

printf 'Reproduction run completed under %s\n' "${WORKSPACE}/asgd_fast_experiments/outputs"
