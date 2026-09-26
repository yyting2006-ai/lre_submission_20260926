#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
MAIN_PREDICTIONS = (
    ROOT
    / "asgd_fast_experiments/outputs/exp_20260718_grouped_kg_v2/predictions_with_rule_traces.csv"
)
MACBERT_PREDICTIONS = (
    ROOT
    / "asgd_fast_experiments/outputs/exp_20260718_macbert_grouped_v1/predictions.csv"
)
OUT_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_selective_review_v1"
BUDGETS = (0.05, 0.10, 0.20)
NEGATIVE_LABEL = "不是目标语法点"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(row: dict, policy: str) -> tuple[float, list[str]]:
    confidence_risk = 1.0 - float(row["confidence"])
    reasons = ["low_confidence"]
    if policy == "confidence_only":
        return confidence_risk, reasons

    conflict = int(row["kg_strong_reject"]) == 1 and row["prediction"] != NEGATIVE_LABEL
    weak_reject = int(row["kg_weak_reject"]) == 1
    risk = confidence_risk
    if conflict:
        risk += 1.0
        reasons.append("strong_rule_prediction_conflict")
    if weak_reject:
        risk += 0.25
        reasons.append("weak_rule_warning")
    return risk, reasons


def evaluate(rows: list[dict], budget: float, policy: str) -> tuple[dict, list[dict]]:
    ranked = []
    for row in rows:
        risk, reasons = score(row, policy)
        ranked.append((risk, row["编号"], reasons, row))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    review_n = max(1, round(len(ranked) * budget))
    reviewed = ranked[:review_n]
    retained = ranked[review_n:]
    all_errors = sum(1 - int(item[3]["correct"]) for item in ranked)
    captured_errors = sum(1 - int(item[3]["correct"]) for item in reviewed)
    retained_errors = sum(1 - int(item[3]["correct"]) for item in retained)
    reviewed_conflicts = sum(
        "strong_rule_prediction_conflict" in item[2] for item in reviewed
    )

    metrics = {
        "review_budget": budget,
        "review_n": review_n,
        "review_rate": review_n / len(ranked),
        "retained_coverage": len(retained) / len(ranked),
        "retained_accuracy": (
            sum(int(item[3]["correct"]) for item in retained) / len(retained)
            if retained
            else 0.0
        ),
        "error_capture_rate": captured_errors / all_errors if all_errors else 0.0,
        "review_precision": captured_errors / review_n,
        "retained_errors": retained_errors,
        "total_errors": all_errors,
        "reviewed_kg_conflicts": reviewed_conflicts,
    }
    audit = []
    for rank, (risk, _, reasons, row) in enumerate(reviewed, start=1):
        audit.append(
            {
                "rank": rank,
                "review_budget": budget,
                "编号": row["编号"],
                "sentence_group": row["sentence_group"],
                "gold": row["gold"],
                "prediction": row["prediction"],
                "correct": int(row["correct"]),
                "confidence": float(row["confidence"]),
                "risk": risk,
                "review_reasons": ";".join(reasons),
                "kg_strong_reject": int(row["kg_strong_reject"]),
                "kg_weak_reject": int(row["kg_weak_reject"]),
                "kg_rule_trace": row["kg_rule_trace"],
            }
        )
    return metrics, audit


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["system"], row["review_policy"], row["review_budget"])].append(row)
    metrics = (
        "review_rate",
        "retained_coverage",
        "retained_accuracy",
        "error_capture_rate",
        "review_precision",
        "retained_errors",
        "total_errors",
        "reviewed_kg_conflicts",
    )
    output = []
    for key, items in sorted(groups.items()):
        base = dict(zip(("system", "review_policy", "review_budget"), key))
        record = {**base, "seeds": len(items)}
        for metric in metrics:
            values = [float(item[metric]) for item in items]
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        output.append(record)
    return output


def run() -> None:
    main_rows = read_csv(MAIN_PREDICTIONS)
    macbert_rows = read_csv(MACBERT_PREDICTIONS)
    kg_signal = {
        (row["seed"], row["编号"]): row
        for row in main_rows
        if row["task"] == "target_label" and row["model"] == "char_svm_plus_kg_full"
    }

    systems: list[tuple[str, str, list[dict]]] = []
    for model, system, policy in (
        ("char_tfidf_svm", "Char-SVM", "confidence_only"),
        ("char_svm_plus_kg_full", "Char-SVM+ExecKG", "confidence_only"),
        (
            "char_svm_plus_kg_full",
            "Char-SVM+ExecKG-review",
            "kg_conflict_aware",
        ),
    ):
        selected = [
            row
            for row in main_rows
            if row["task"] == "target_label" and row["model"] == model
        ]
        systems.append((system, policy, selected))

    mac_rows = []
    for row in macbert_rows:
        signal = kg_signal[(row["seed"], row["编号"])]
        mac_rows.append(
            {
                **row,
                "kg_weak_reject": signal["kg_weak_reject"],
                "kg_rule_trace": signal["kg_rule_trace"],
            }
        )
    systems.extend(
        (
            ("MacBERT", "confidence_only", mac_rows),
            ("MacBERT+ExecKG-review", "kg_conflict_aware", mac_rows),
        )
    )

    by_seed_rows: list[dict] = []
    audit_rows: list[dict] = []
    for system, policy, rows in systems:
        rows_by_seed: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            rows_by_seed[int(row["seed"])].append(row)
        for seed, seed_rows in sorted(rows_by_seed.items()):
            assert len(seed_rows) in {599, 600}, (system, seed, len(seed_rows))
            for budget in BUDGETS:
                metrics, audit = evaluate(seed_rows, budget, policy)
                by_seed_rows.append(
                    {
                        "system": system,
                        "review_policy": policy,
                        "seed": seed,
                        "n": len(seed_rows),
                        **metrics,
                    }
                )
                for row in audit:
                    audit_rows.append(
                        {
                            "system": system,
                            "review_policy": policy,
                            "seed": seed,
                            **row,
                        }
                    )

    aggregate_rows = aggregate(by_seed_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "selective_review_by_seed.csv", by_seed_rows)
    write_csv(OUT_DIR / "selective_review_aggregate.csv", aggregate_rows)
    write_csv(OUT_DIR / "review_queue_audit.csv", audit_rows)
    summary = {
        "budgets": list(BUDGETS),
        "ranking_is_label_free": True,
        "risk_definition": {
            "confidence_only": "1 - predicted-class confidence",
            "kg_conflict_aware": (
                "1 - confidence + 1.0 * strong-rule/prediction conflict "
                "+ 0.25 * weak-rule warning"
            ),
        },
        "test_label_use": "Metrics only; no threshold or weight was tuned on test labels.",
        "rows_by_seed": {str(seed): len(rows) for seed, rows in sorted(rows_by_seed.items())},
    }
    (OUT_DIR / "protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
