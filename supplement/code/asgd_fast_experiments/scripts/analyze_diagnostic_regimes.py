#!/usr/bin/env python3
"""Aggregate released predictions under the two diagnostic regimes."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "results/main_predictions_release.csv"
DEFAULT_BY_SEED = ROOT / "results/diagnostic_regime_results_by_seed.csv"
DEFAULT_AGGREGATE = ROOT / "results/diagnostic_regime_results_aggregate.csv"

NEGATIVE = "不是目标语法点"
POSITIVE = "是目标语法点"
FUNCTIONAL_FAMILY = "时间/处所功能结构"
EXPECTED_FAMILIES = {
    "介词/框架结构",
    "处置/受事重组构式",
    "复句关联结构",
    "多谓词/论元结构",
    FUNCTIONAL_FAMILY,
    "比较构式",
    "补语结构",
    "被动/受事凸显构式",
}
MODELS = (
    "char_tfidf_lr",
    "char_lr_plus_kg_markers",
    "char_lr_plus_kg_constraints",
    "char_tfidf_svm",
    "char_svm_plus_kg_constraints",
)
REGIMES = ("structural_pattern", "complex_functional")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--by-seed-output", type=Path, default=DEFAULT_BY_SEED)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def regime_for(family: str) -> str:
    if family not in EXPECTED_FAMILIES:
        raise ValueError(f"unexpected diagnostic family: {family}")
    return "complex_functional" if family == FUNCTIONAL_FAMILY else "structural_pattern"


def scores(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot score an empty group")
    correct = sum(row["gold"] == row["prediction"] for row in rows)
    f1_values = []
    for label in (NEGATIVE, POSITIVE):
        true_positive = sum(
            row["gold"] == label and row["prediction"] == label for row in rows
        )
        false_positive = sum(
            row["gold"] != label and row["prediction"] == label for row in rows
        )
        false_negative = sum(
            row["gold"] == label and row["prediction"] != label for row in rows
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    negatives = [row for row in rows if row["gold"] == NEGATIVE]
    if not negatives:
        raise ValueError("diagnostic regime has no negative examples")
    return {
        "accuracy": correct / len(rows),
        "macro_f1": sum(f1_values) / 2,
        "hard_negative_accuracy": sum(
            row["prediction"] == NEGATIVE for row in negatives
        )
        / len(negatives),
    }


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in read_rows(args.predictions)
        if row["task"] == "target_label" and row["model"] in MODELS
    ]
    observed_families = {row["语法家族"] for row in rows}
    if observed_families != EXPECTED_FAMILIES:
        raise ValueError(
            f"family mismatch: expected {sorted(EXPECTED_FAMILIES)}, "
            f"observed {sorted(observed_families)}"
        )

    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), regime_for(row["语法家族"]), row["model"])].append(row)

    by_seed: list[dict[str, object]] = []
    for seed in sorted({key[0] for key in grouped}):
        for regime in REGIMES:
            for model in MODELS:
                group = grouped[(seed, regime, model)]
                metric = scores(group)
                by_seed.append({
                    "seed": seed,
                    "regime": regime,
                    "model": model,
                    "n": len(group),
                    "n_negative": sum(row["gold"] == NEGATIVE for row in group),
                    **metric,
                })

    aggregate: list[dict[str, object]] = []
    for regime in REGIMES:
        for model in MODELS:
            selected = [
                row for row in by_seed
                if row["regime"] == regime and row["model"] == model
            ]
            if len(selected) != 3:
                raise ValueError(f"expected three seeds for {regime}/{model}")
            aggregate.append({
                "regime": regime,
                "model": model,
                "n_mean": statistics.mean(float(row["n"]) for row in selected),
                "n_negative_mean": statistics.mean(
                    float(row["n_negative"]) for row in selected
                ),
                "accuracy_mean": statistics.mean(
                    float(row["accuracy"]) for row in selected
                ),
                "accuracy_std": statistics.stdev(
                    float(row["accuracy"]) for row in selected
                ),
                "macro_f1_mean": statistics.mean(
                    float(row["macro_f1"]) for row in selected
                ),
                "macro_f1_std": statistics.stdev(
                    float(row["macro_f1"]) for row in selected
                ),
                "hard_negative_accuracy_mean": statistics.mean(
                    float(row["hard_negative_accuracy"]) for row in selected
                ),
                "hard_negative_accuracy_std": statistics.stdev(
                    float(row["hard_negative_accuracy"]) for row in selected
                ),
            })

    write_rows(args.by_seed_output, by_seed)
    write_rows(args.aggregate_output, aggregate)
    print(f"wrote {args.by_seed_output}")
    print(f"wrote {args.aggregate_output}")


if __name__ == "__main__":
    main()
