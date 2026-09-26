# -*- coding: utf-8 -*-
"""Recompute the LRE addendum counts and headline scores."""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(name):
    with (HERE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def macro_f1(items):
    labels = ["不是目标语法点", "是目标语法点"]
    scores = []
    for label in labels:
        tp = sum(r["gold"] == label and r["prediction"] == label for r in items)
        fp = sum(r["gold"] != label and r["prediction"] == label for r in items)
        fn = sum(r["gold"] == label and r["prediction"] != label for r in items)
        scores.append(0.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / 2


def near(actual, expected):
    if abs(actual - expected) > 0.00015:
        raise SystemExit(f"expected {expected}, got {actual}")


def main():
    gold = rows("author_written_244.csv")
    assert len(gold) == 244
    assert sum(r["gold"] == "是目标语法点" for r in gold) == 122
    assert sum(r["benchmark_label"] == "yes" for r in gold) == 148
    executor = rows("executor_predictions.csv")
    assert len(executor) == 244
    by_block = {
        "all": executor,
        "yes": [r for r in executor if r["in_benchmark_labels"] == "1"],
        "no": [r for r in executor if r["in_benchmark_labels"] == "0"],
    }
    near(macro_f1(by_block["all"]), 0.5989)
    near(macro_f1(by_block["yes"]), 0.6592)
    near(macro_f1(by_block["no"]), 0.3333)
    svm = rows("svm_predictions.csv")
    assert len(svm) == 1464
    review = rows("specification_review_record.csv")
    assert len(review) == 646
    counts = {}
    for row in review:
        counts[row["specification_judgment"]] = counts.get(row["specification_judgment"], 0) + 1
    assert counts == {
        "consistent_with_cited_page": 368,
        "wording_repair_condition_unchanged": 55,
        "condition_mismatch": 8,
        "anchor_too_coarse": 215,
    }
    assert all(row["runtime_binding_changed"] == "false" for row in review)
    rewrites = rows("unit_rewrites_checked.csv")
    assert len(rewrites) == 8
    assert all(row["check"] == "可用" for row in rewrites)
    boot = rows("joint_group_bootstrap.csv")
    assert all(row["seed"] == "pooled" for row in boot)
    wanted = {
        ("all", "char_tfidf_svm", "char_svm_plus_kg_constraints", "macro_f1"): (2.09, 0.84, 3.4),
        ("all", "macbert_base_finetuned", "char_svm_plus_kg_constraints", "macro_f1"): (0.98, -1.53, 3.6),
        ("structural_pattern", "char_lr_plus_kg_markers", "char_lr_plus_kg_constraints", "macro_f1"): (2.57, 1.08, 4.25),
    }
    found = 0
    for row in boot:
        key = (row["regime"], row["baseline"], row["model"], row["metric"])
        if key not in wanted:
            continue
        point, low, high = wanted[key]
        near(float(row["delta_pct"]), point)
        near(float(row["ci_low_pct"]), low)
        near(float(row["ci_high_pct"]), high)
        found += 1
    if found != len(wanted):
        raise SystemExit(f"bootstrap rows matched {found}")
    print("PASS", {"author_written": 244, "units": 646, "rewrites": 8})


if __name__ == "__main__":
    main()
