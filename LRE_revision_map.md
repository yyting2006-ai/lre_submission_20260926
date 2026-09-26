# LRE Revision Map

Target venue: *Language Resources and Evaluation*, full-length paper.

## Manuscript contract

- Problem: teaching-grammar standards name constructions but do not provide executable, source-auditable verification constraints.
- Contribution type: language resource and evaluation protocol, with a bounded classifier study.
- Central claim: ExecKG supplies a versioned interface that connects official grammar points to diagnostic profiles, executable bindings, contract tests, and source anchors; the interface improves candidate verification under the released benchmark conditions.
- Scope: GF 0025--2021, 572 official points, 646 diagnostic units, 37 benchmark labels, 3,000 supplied candidates, and 244 author-written candidates.
- Boundary: static compilation and contract tests do not constitute natural-language behavioral validation for all 572 points.

## Changes made in this submission version

| Area | Action | Evidence |
| --- | --- | --- |
| Title and abstract | Reframed the paper as a source-anchored grammar resource; bounded behavioral claims in the abstract. | `execkg_lre.tex` |
| Introduction | Added an explicit resource contract and three contribution statements. | `execkg_lre.tex` |
| Resource accounting | Clarified that 524 unique page-anchor records support 572 alignment relations. | `execkg_lre.tex` |
| Specification audit | Added the distinction between source consistency review and linguistic adjudication. | `execkg_lre.tex` |
| Independent transfer | Kept the 244 author-written set in the main evidence chain and reported the negative transfer result. | `execkg_lre.tex`, `supplement/lre_addendum/` |
| Declarations | Changed the heading to `Statements and Declarations`; stated the exact review-package name and license boundary. | `execkg_lre.tex` |
| References | Switched the Springer class to author--year citations with `NameDate` and enabled the style's alphabetized `alpha` option. | `execkg_lre.tex`, `execkg_lre.bib` |
| Cover letter | Rewrote the title, contribution description, package name, and resource boundary. | `cover_letter.tex` |

## Material unresolved constraints

- Standard text, handbook prose, and benchmark sentences remain restricted by source licenses and are not redistributed.
- The public-facing repository is `https://github.com/yyting2006-ai/lre_submission_20260926`; the same release is also prepared as a supplementary upload.
- The behavioral evaluation remains an offline benchmark, and the pooled test splits share some sentence groups across seeds.
