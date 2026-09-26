# ExecKG LRE Submission Package

This directory is the prepared submission state for *Language Resources and Evaluation*.

## Files

- `execkg_lre.pdf`: compiled manuscript.
- `execkg_lre.tex`: manuscript source.
- `execkg_lre.bib`: bibliography.
- `cover_letter.pdf`: compiled cover letter.
- `cover_letter.tex`: cover-letter source.
- `fig_*.pdf`: manuscript figures.
- `supplement/`: reproducibility and release-safe supplementary package.
- `ExecKG_LRE_supplement_20260926.zip`: upload-ready supplementary archive built from `supplement/`.
- `LRE_revision_map.md`: claim contract and revision record.
- `LRE_submission_checklist.md`: completed pre-upload checks.
- `SHA256SUMS.txt`: hashes for the manuscript, cover letter, and supplementary archive.

## Compile

From this directory:

```powershell
F:\游庭睿学习\李吉梅\现有资料\分词研究\第九篇_把字句\acl2027_work\tools\tectonic.exe execkg_lre.tex
F:\游庭睿学习\李吉梅\现有资料\分词研究\第九篇_把字句\acl2027_work\tools\tectonic.exe cover_letter.tex
```

The manuscript uses the Springer Nature `sn-jnl` class with author--year citations.

## Supplement verification

The addendum verifier can run with a standard Python 3.12 installation:

```powershell
python supplement\lre_addendum\verify_lre_addendum.py
```

The full release verifier intentionally requires the pinned CPython 3.12.13 environment because its compiler witness generator uses private regular-expression parser APIs.

The verified supplementary archive is `ExecKG_LRE_supplement_20260926.zip`. The archive excludes the licensed standard text, handbook prose, and benchmark sentences; those materials remain with their rights holders.
