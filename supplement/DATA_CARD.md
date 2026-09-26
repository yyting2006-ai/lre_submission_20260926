# Data Card

## Task and Scope

Each main experimental row contains a sentence, a supplied candidate span, a
proposed grammar point, and a diagnostic family. The primary task judges
whether the candidate instantiates the proposed point. Companion fields cover
candidate-boundary quality, diagnostic function, structural-evidence support,
slot status, and copied slot spans.

The authority catalog contains all 572 numbered points in GF0025--2021. The v3
compiler expands them to 646 diagnostic units and profiles because 50 points
contain multiple operational usages. Candidate-level experiments remain a
separate frozen space of 37 labels in eight families and 3,000 candidates.

## Inventory-Wide Compilation

The compiler materializes 646 source-anchored runtime bindings through 233
templates and all 103 official taxonomy paths. It applies four manually verified
catalog-field corrections to two points. It emits seven deterministic contract
cases for every unit, for 4,522 passing rows total.

Of the 646 bindings, 503 use normalized legacy regexes. This count includes 39
split units whose active detector adds a unit-specialization pattern. The other
143 bindings use literal cues derived from the official point entry. These
source-literal patterns are scaffolds for candidate detection, not validated
natural-language recognizers.

Contract tests verify schema, detector invocation, positive detector matching,
negative gating, and the seven diagnosis branches. They are synthetic static
contracts. They do not establish semantic correctness, deployment recall, or
linguist approval of every generated profile.

## Source Alignment and Positive Evidence

Every official point has a copyright-safe handbook alignment. Confidence is
high for 569 points and medium for three; none is low or unresolved. The public
alignment stores source ID, page number, source-derived keywords, score,
confidence, and a page SHA-256 only. It contains no handbook page text or prose
snippet.

The official standard anchors comprise 362 OCR-detected point pages and 210
verified appendix-span anchors. These are authority-location records, distinct
from the three-handbook alignment.

A separate audit of authorized annotated material finds 3,645 unique positive
sentence--point pairs supporting 219 official points. The remaining 353 points
do not have released annotated positive evidence. Sentence text is withheld;
the public summary retains counts, a small sample of unsalted SHA-256 hashes,
and the source-workbook hash. This evidence count must not be confused with the
complete 572-point static compilation claim.

## Candidate Construction

Candidate construction preceded annotation. Fixed regex inventories were run
over searchable local copies of three Grammar Learning Handbooks for seven
families. Exact sentence-family-span duplicates were removed. Within fixed
family quotas, deterministic label-free ranking favored concise, sentence-like
candidates and pages with matching grammar-topic cues. Ties used seed 20260629.
The 650 temporal/locative candidates came from an earlier authoritative pool.

The quotas define a diagnostic evaluation sample and do not estimate natural
deployment prevalence. Source PDFs and extracted pages are excluded for rights
reasons.

## Annotation and Composition

Three annotators completed qualification and calibration before formal
annotation. Consolidation was field-specific: unanimous decisions were kept,
two-of-three decisions used the majority, and no-majority cases used a
predesignated resolver. Agreement files report pre-consolidation statistics.

| Product | Rows | Notes |
|---|---:|---|
| Main candidate decisions | 3,000 | 2,557 normalized sentence groups |
| Evidence-support decisions | 1,000 | Stratified subset |
| Slot rows | 1,320 | Slot assertions, not extra sentences |
| Official authority points | 572 | Complete GF0025--2021 numbered inventory |
| Diagnostic units/profiles/bindings | 646 each | Machine-compiled and statically audited |
| Contract tests | 4,522 | Seven synthetic branches per unit |

The main target distribution is 2,539 target, 457 non-target, and four review
cases. Review cases are excluded from supervised target scoring.

## Sources, Rights, and Public Fields

Examples come from authoritative standards, grammar references, and teaching
materials with page-level provenance. Some works are not licensed for public
redistribution. The public package therefore excludes verbatim sentence text,
candidate text, correction text, copied slot spans, annotator free text,
handbook page text, exact text-bearing prompts, and raw API responses.

The package retains stable IDs, normalized sentence hashes, grammar labels,
consolidated decisions, official-standard labels, source IDs, page anchors,
page hashes, split IDs, compiler structures, rule traces, and predictions. API
keys and account metadata are never stored.

The complete v3 `nodes.csv` and `edges.csv` are also withheld. They preserve the
evaluated v2 subgraph, whose sentence nodes contain licensed text. The public
`graph_summary_v3_full_compilation.json` and schema report totals and integrity
without leaking those nodes.

## Known Limitations

- The study is candidate-conditioned and does not measure end-to-end candidate
  generation recall or deployment prevalence.
- Full inventory coverage means source-anchored machine compilation and static
  contract validation. It is not a 572-point behavioral accuracy benchmark.
- Source-literal detector patterns for 143 bindings are compiler scaffolds.
- Candidate-level empirical results cover 37 labels and 3,000 candidates.
- Annotated positive evidence covers 219 of 572 official points.
- Fine-grained boundary and slot labels have modest chance-corrected agreement.
- External model checks cover a fixed split and prompt-selection space; they are
  not a general ranking of LLM capability.
- Exact external-call reproduction needs authorized source text and the
  restricted prompt bundle.
- No student learning-outcome or high-stakes assessment claim is supported.
