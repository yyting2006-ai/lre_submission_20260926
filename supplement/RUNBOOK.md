# Reproduction Runbook

## 1. Exact Environment

The v3 compiler must run under exactly CPython 3.12.13. Its deterministic regex
witness generator uses private `re._parser` and `re._constants` APIs, whose
behavior is not a cross-version interface. The verifier fails fast on any other
Python version.

Reported empirical runs used NumPy 2.3.5, SciPy 1.18.0, scikit-learn 1.9.0,
PyTorch 2.13.0, and Transformers 4.57.6. The recorded hardware was an Apple M5
Pro with 15 CPU cores, 24 GB memory, and MPS on macOS 26.4.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 --version
python3 scripts/verify_release.py
```

The expected version line is `Python 3.12.13`.

## 2. Rebuild the 572-Point Compiler Layer

The released catalog, correction table, and seed constraint library are
sufficient to regenerate the crosswalk, templates, runtime bindings, contract
tests, and coverage report without handbook text:

```bash
rm -rf /tmp/execKG-v3
python3 code/grammar_kg/tools/compile_standard_inventory_v3.py \
  --catalog code/grammar_kg/kg/standard_grammar_points_gf0025_v2.csv \
  --constraints code/grammar_kg/kg/constraints_v0_1.json \
  --corrections code/grammar_kg/kg/standard_catalog_corrections_v3.csv \
  --out-dir /tmp/execKG-v3

for name in \
  standard_diagnostic_crosswalk_v3.csv \
  diagnostic_templates_v3.csv \
  executable_bindings_full_inventory_v3.json \
  diagnostic_unit_tests_v3.csv \
  full_inventory_executable_coverage_v3.json; do
  cmp "/tmp/execKG-v3/${name}" "graph/${name}"
done
```

Expected outputs are 572 official points, 646 units/profiles/bindings, 233
templates, and 4,522 passing contract tests. The binding source counts are 503
normalized-regex bindings, including 39 unit-specialized split detectors, and
143 source-literal scaffold bindings. All 103 taxonomy paths are covered.

These are static binding contracts. The synthetic witnesses exercise detector
and decision branches; they do not estimate semantic accuracy on prose.

## 3. Rebuild Handbook Alignment (Authorized Layer)

The public package releases only source IDs, page numbers, short matched
keywords, scores, confidence, and page hashes. It does not include handbook
pages or extracted handbook text. With lawful local copies already converted
to the aligner's page-JSONL format, run:

```bash
python3 code/grammar_kg/tools/align_standard_points_to_handbooks_v3.py \
  --points code/grammar_kg/kg/standard_grammar_points_gf0025_v2.csv \
  --pages /authorized/path/source_pages_raw.jsonl \
  --catalog-corrections code/grammar_kg/kg/standard_catalog_corrections_v3.csv \
  --alignment-out /tmp/standard_handbook_alignment_v3.csv \
  --coverage-out /tmp/full_inventory_handbook_coverage_v3.json
```

Expected alignment is 572/572: 569 high confidence, three medium confidence,
and no low or unresolved rows. The released coverage JSON replaces local input
paths with public or restricted-layer labels; numeric results are unchanged.

## 4. Rebuild the v3 Graph (Authorized Layer)

The graph builder is included, but a full public rebuild is intentionally not
possible because the preserved v2 `nodes.csv` contains licensed sentence text.
With an authorized v2 graph and the released v3 artifact set arranged in an
authorized `kg` directory, run:

```bash
python3 code/grammar_kg/tools/build_full_compilation_kg_v3.py \
  --base /authorized/path/kg_v2_full_inventory \
  --kg /authorized/path/v3_artifacts \
  --out /tmp/kg_v3_full_compilation
```

Expected totals are 35,379 nodes and 72,898 edges. Integrity checks must report
zero duplicate node IDs, zero duplicate edges, zero dangling edges, and exact
preservation of 28,158 v2 nodes and 64,046 v2 edges. The public package verifies
the summary, schema, and hashes of the v3 input artifacts without distributing
the text-bearing full graph.

## 5. Candidate Construction and Source Audit

With lawful local handbook PDFs, reconstruct the seven regex-derived family
pools and label-free sample:

```bash
python3 code/data_preparation/extract_handbook_candidates.py \
  --elementary /authorized/path/elementary.pdf \
  --intermediate /authorized/path/intermediate.pdf \
  --advanced /authorized/path/advanced.pdf \
  --output /tmp/structured_candidates.json \
  --seed 20260629

python3 code/data_preparation/select_annotation_sample.py \
  --structured-candidates /tmp/structured_candidates.json \
  --temporal-locative-candidates /authorized/path/temporal_locative_pool.json \
  --output /tmp/asgd3000_preannotation.json \
  --seed 20260629
```

Candidate construction uses no gold label. It intentionally retains
surface-triggered negatives so boundary and non-target diagnoses remain
measurable.

## 6. Authorized Gold Placement and Empirical Runs

Place the three authorized, unredacted files under:

```text
code/annotation_qc/outputs/formal_qc_20260718/
  gold_main_by_filename_majority_annotator2_20260718.csv
  gold_evidence_by_filename_majority_annotator2_20260718.csv
  gold_slot_span_by_filename_majority_annotator2_20260718.csv
```

Then run:

```bash
PYTHON=python3 bash run_all.sh
RUN_MACBERT=1 PYTHON=python3 bash run_all.sh
RUN_EFFICIENCY=1 PYTHON=python3 bash run_all.sh
```

The empirical benchmark remains 37 operational labels and 3,000 candidates.
Seeds are 13, 42, and 2027. Hyperparameter selection uses development macro-F1
over `C in {0.25, 1, 4}`. MacBERT needs network access for model download.

## 7. External LLM Evaluation

External processing must not run without explicit authorization and lawful
access to source text. The public package contains text-free predictions and
protocol hashes, not text-bearing prompts or raw responses. Authorized users
can run the restricted API scripts separately, then recompute analyses with:

```bash
RUN_LLM_ANALYSIS=1 PYTHON=python3 bash run_all.sh
```

The verifier reproduces the released LLM scores, provider selection checks,
and explanation-stability statistics offline and without an API key.

## 8. Expected Output and Interpretation

Inventory-wide compiler outputs should byte-match the five released v3
artifacts under `graph/` when run with CPython 3.12.13. Empirical outputs appear
under `code/asgd_fast_experiments/outputs/` and can be compared with `results/`.
Timing may vary by hardware; neural metrics can vary slightly on nondeterministic
accelerators.

Do not interpret the compiler contracts as natural-language accuracy tests.
Annotated positive support currently covers 219 of 572 official points through
3,645 unique sentence--point pairs. That evidence audit is reported separately
from complete compilation coverage.
