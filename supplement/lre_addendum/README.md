# ExecKG LRE addendum

This directory sits inside the supplementary package. It holds the records added for the Language Resources and Evaluation manuscript.

`author_written_244.csv` contains the 244 author-written candidates: sentence, span, label, construction plan, and adjudicated gold. `executor_predictions.csv` and `svm_predictions.csv` are the frozen executor and the refit SVM outputs on those candidates. `specification_review_record.csv` is the page review of all 646 diagnostic units. `unit_rewrites_checked.csv` is the checked unit-level rewrite for the eight condition mismatches. `joint_group_bootstrap.csv` keeps the pooled shared-weight intervals. The complex-functional marker step is 3.83 in the manuscript, the mean of the three released seed estimates. This file records 3.84 for that row.

From the package root:

```bash
python lre_addendum/verify_lre_addendum.py
```

`scripts/verify_release.py` remains the check for the frozen compiler and the 3,000-candidate scores. Benchmark sentences, handbook prose, and standard text are not in this directory.
