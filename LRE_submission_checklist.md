# LRE Submission Checklist

## Manuscript

- [x] Full-length paper framing is resource-centered.
- [x] Title, author affiliations, corresponding author, and ORCID identifiers are present.
- [x] Abstract is within the usual 150--250 word range.
- [x] Six keywords are provided.
- [x] Author--year citations are enabled with the Springer `NameDate` option.
- [x] The bibliography is alphabetized through the Springer `alpha` setting.
- [x] Figures and tables have captions and labels.
- [x] Limitations are explicit and placed before the statements section.
- [x] Statements and Declarations include funding, competing interests, ethics, consent, data availability, code availability, and author contributions.

## Evidence

- [x] Inventory-wide counts are tied to the frozen release.
- [x] Benchmark counts and headline metrics match the release records.
- [x] The 244 author-written candidates are treated as a transfer diagnostic.
- [x] The MacBERT comparison and the complex-functional internal steps retain intervals that include zero.
- [x] Contract tests are described as deterministic binding checks, not semantic validation.

## Supplementary package

- [x] `supplement/README.md`
- [x] `supplement/DATA_CARD.md`
- [x] `supplement/DATA_LICENSE.md`
- [x] `supplement/scripts/verify_release.py`
- [x] `supplement/lre_addendum/verify_lre_addendum.py`
- [x] Compiler, graph summaries, release-safe labels, predictions, and audit records.
- [x] `ExecKG_LRE_supplement_20260926.zip` created from the verified `supplement/` tree.

## Local validation performed

- [x] LRE manuscript compiled with Tectonic 0.15.0.
- [x] PDF rendered to 22 pages.
- [x] Visual inspection completed for the title/abstract page, introduction, main results, declarations, and appendix.
- [x] LRE addendum verifier passed under pinned CPython 3.12.13.
- [x] Full release verifier passed under pinned CPython 3.12.13 with bytecode generation disabled.
- [x] Supplement ZIP was inspected: 148 entries, both verifiers present, no `__pycache__` entries.
