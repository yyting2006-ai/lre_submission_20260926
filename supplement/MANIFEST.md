# Package Manifest

## Inventory-Wide v3 Release Artifacts

- `graph/standard_catalog_corrections_v3.csv`: four verified field corrections
  affecting two official points.
- `graph/standard_diagnostic_crosswalk_v3.csv`: 646 diagnostic units/profiles
  mapped to all 572 points.
- `graph/diagnostic_templates_v3.csv`: 233 operational templates.
- `graph/executable_bindings_full_inventory_v3.json`: 646 machine-compiled
  runtime bindings.
- `graph/diagnostic_unit_tests_v3.csv`: 4,522 deterministic binding contracts.
- `graph/full_inventory_executable_coverage_v3.json`: counts, branch coverage,
  integrity checks, and claim boundary.
- `graph/standard_handbook_alignment_v3.csv`: 572 copyright-safe page
  alignments, with no page text.
- `graph/full_inventory_handbook_coverage_v3.json`: 569 high- and three
  medium-confidence alignment summary.
- `graph/standard_positive_evidence_summary_v3.csv`: per-point counts and
  sentence hashes for authorized positive evidence.
- `graph/full_inventory_example_coverage_v3.json`: aggregate evidence coverage
  for 3,645 unique pairs and 219 supported points.
- `graph/graph_summary_v3_full_compilation.json`: v3 totals, preservation counts,
  integrity checks, and input hashes.
- `graph/schema_v3_full_compilation.json`: v3 node and edge schema.

## Released Compiler Inputs and Code

- `code/grammar_kg/kg/standard_grammar_points_gf0025_v2.csv`
- `code/grammar_kg/kg/constraints_v0_1.json`
- `code/grammar_kg/kg/standard_catalog_corrections_v3.csv`
- `code/grammar_kg/tools/compile_standard_inventory_v3.py`
- `code/grammar_kg/tools/align_standard_points_to_handbooks_v3.py`
- `code/grammar_kg/tools/build_full_compilation_kg_v3.py`

The compiler can be rerun from public inputs under CPython 3.12.13. Handbook
alignment and full graph construction additionally require authorized inputs.

## Earlier Graph and Experimental Layers

- `data/main_labels_release.csv`, `data/evidence_labels_release.csv`, and
  `data/slot_labels_release.csv`: consolidated labels without source text.
- `graph/rule_catalog_release.csv`: 144 source-linked experimental rules.
- `graph/grammar_points_release.csv`: 37 empirical labels.
- `graph/standard_inventory_release_v2.csv`: 572-point copyright-safe v2
  authority catalog.
- `graph/graph_summary.json` and `graph/schema.json`: evaluated v1 subgraph.
- `graph/graph_summary_v2_full_inventory.json`,
  `graph/full_inventory_coverage_v2.json`, and
  `graph/schema_v2_full_inventory.json`: v2 authority-layer summaries.
- `results/`: released per-instance predictions, aggregate metrics, agreement,
  bootstrap analyses, neural results, and text-free LLM outputs.

## Other Executable Files

- `code/data_preparation/extract_handbook_candidates.py`
- `code/data_preparation/select_annotation_sample.py`
- `code/asgd_fast_experiments/scripts/`: grouped lightweight, auxiliary,
  neural, significance, efficiency, trace-state, and LLM analysis scripts.
- `code/annotation_qc/tools/audit_annotation_agreement_20260718.py`
- `scripts/export_release_safe_data.py`: copyright/path audit and release
  exporter.
- `scripts/verify_release.py`: deterministic package and headline verifier.
- `run_all.sh`: authorized empirical workflow.
- `.python-version`: exact interpreter pin (`3.12.13`).
- `requirements.txt`: exact library pins used by reported runs.

## Intentionally Excluded

- full v3 `nodes.csv` and `edges.csv`, because preserved v2 sentence nodes
  contain licensed text;
- handbook PDFs and extracted handbook page text;
- verbatim source sentences, candidate strings, corrections, and copied spans;
- annotator-level text and free-form notes;
- text-bearing API prompts, demonstrations, and raw responses;
- API credentials, account metadata, model weights, and local caches;
- `__pycache__`, `.pyc`, author identities, grant identifiers, and local
  filesystem paths.

## License Scope

- `LICENSE`: BSD 3-Clause License for source code under `code/`, `scripts/`,
  and `run_all.sh`.
- `DATA_LICENSE.md`: CC BY 4.0 notice for author-created release-safe metadata
  and annotations under `data/`, `graph/`, and `results/`.
