# ExecKG Structured Grammar Diagnosis: Reproducibility Package

This supplementary package accompanies *ExecKG: compiling the Chinese grammar
standard GF 0025--2021 into an executable knowledge graph*.

## What This Package Verifies

The package separates inventory-wide compilation from candidate-level empirical
evaluation.

The v3 compiler covers all 572 numbered points in GF0025--2021. It expands them
into 646 diagnostic units, 646 diagnostic profiles, and 646 runtime bindings.
The units use 233 operational templates and cover all 103 official taxonomy
paths. The compiler emits 4,522 passing contract tests, with seven deterministic
branches per unit. Of the bindings, 503 use normalized legacy regexes; 39 of
those add unit-specialized detectors for split points. The remaining 143 use
source-literal detector scaffolds.

The source layer records four verified catalog-field corrections affecting two
official points. It also records 572/572 handbook page alignments: 569 high
confidence and three medium confidence, with no low or unresolved alignment.
The standard inventory itself has 362 OCR-detected page anchors and 210
appendix-span anchors.

The v3 graph has 35,379 nodes and 72,898 edges. It preserves all 28,158 nodes
and 64,046 edges from the v2 full-inventory graph. The public package includes
the v3 graph summary and schema, not the full node and edge tables, because the
preserved evaluated subgraph contains licensed sentence text.

Candidate-level experiments remain the frozen 37-label, eight-family benchmark:
3,000 candidate-conditioned decisions in 2,557 sentence groups, 1,000 evidence
decisions, and 1,320 slot rows. The released result files retain all reported
per-seed metrics, bootstrap analyses, neural and lightweight baselines, and
text-free LLM evaluation outputs.

## Claim Boundary

The 4,522 generated tests validate deterministic binding contracts and branch
coverage. They are not semantic or behavioral validation on natural-language
examples. Source-literal detectors are explicit compiler scaffolds. Full
572-point compilation does not turn the 37-label/3,000-candidate experiment
into a 572-point behavioral benchmark, and it does not imply linguist review of
every generated profile.

## Quick Verification

The compiler witness generator uses private `re` parser APIs. Use exactly
CPython 3.12.13, as pinned in `.python-version` and enforced by the verifier.

```bash
python3 --version  # Python 3.12.13
python3 scripts/verify_release.py
```

The verifier checks the v3 compiler artifacts, all 572 authority records,
handbook and positive-evidence summaries, graph preservation, privacy rules,
and the paper's frozen empirical headline values. It performs no network calls
and does not require restricted handbook text.

To regenerate the inventory-wide compiler outputs from the released inputs:

```bash
python3 code/grammar_kg/tools/compile_standard_inventory_v3.py \
  --catalog code/grammar_kg/kg/standard_grammar_points_gf0025_v2.csv \
  --constraints code/grammar_kg/kg/constraints_v0_1.json \
  --corrections code/grammar_kg/kg/standard_catalog_corrections_v3.csv \
  --out-dir /tmp/execKG-v3
```

See `RUNBOOK.md` for artifact comparison and for authorized-only handbook and
graph regeneration.

## Copyright-Aware Data Layer

The public release omits verbatim source sentences, candidate strings, copied
slot spans, correction text, annotator-level columns, handbook page text,
free-text notes, and text-bearing API prompts or responses. Stable IDs,
sentence-group hashes, consolidated labels, source/page metadata, rule traces,
page hashes, split manifests, compiler artifacts, and predictions are retained.
See `DATA_CARD.md`.

## Directory Map

```text
data/       copyright-safe consolidated labels and group hashes
graph/      v1/v2/v3 release artifacts, schemas, summaries, and audits
results/    local and LLM predictions, metrics, agreement, and bootstrap outputs
code/       data preparation, experiments, graph builders, and v3 compiler code
scripts/    release exporter and deterministic integrity verifier
```

Code is released under the BSD 3-Clause License. Copyrightable metadata and
annotations created by the authors are released under CC BY 4.0, subject to the
file-level scope in `DATA_LICENSE.md`. Neither license grants rights to
reconstruct or redistribute third-party source sentences or handbook text.
