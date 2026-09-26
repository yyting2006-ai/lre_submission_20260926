#!/usr/bin/env python3
"""Measure held-out validity and coverage of deterministic ExecKG states."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
PREDICTIONS = ROOT / "asgd_fast_experiments/outputs/exp_20260718_grouped_kg_v2/predictions_with_rule_traces.csv"
OUT_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_trace_state_audit_v1"
MODEL = "char_svm_plus_kg_constraints"
POSITIVE = "是目标语法点"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state(row: dict[str, str]) -> str:
    if row["kg_strong_reject"] == "1": return "strong_reject"
    if row["kg_weak_reject"] == "1": return "weak_reject"
    if row["kg_accepted"] == "1": return "accept"
    return "no_decision"


def main() -> None:
    rows = [row for row in read_csv(PREDICTIONS) if row["task"] == "target_label" and row["model"] == MODEL]
    by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows: by_seed[int(row["seed"])].append(row)
    detail: list[dict] = []; states = ("accept", "weak_reject", "strong_reject", "no_decision")
    for seed, seed_rows in sorted(by_seed.items()):
        for state_name in states:
            selected = [row for row in seed_rows if state(row) == state_name]; positives = sum(row["gold"] == POSITIVE for row in selected); n = len(selected)
            detail.append({"seed": seed, "state": state_name, "n": n, "coverage": n / len(seed_rows), "target_rate": positives / n if n else 0.0, "non_target_rate": (n - positives) / n if n else 0.0})
    aggregate: list[dict] = []
    for state_name in states:
        selected = [row for row in detail if row["state"] == state_name]; record = {"state": state_name, "seeds": len(selected)}
        for metric in ("n", "coverage", "target_rate", "non_target_rate"):
            values = np.array([float(row[metric]) for row in selected]); record[f"{metric}_mean"] = float(values.mean()); record[f"{metric}_std"] = float(values.std(ddof=1)); record[f"{metric}_min"] = float(values.min()); record[f"{metric}_max"] = float(values.max())
        aggregate.append(record)
    OUT_DIR.mkdir(parents=True, exist_ok=True); write_csv(OUT_DIR / "trace_state_by_seed.csv", detail); write_csv(OUT_DIR / "trace_state_aggregate.csv", aggregate)
    summary = {"task": "target_label", "model_test_membership": MODEL, "state_is_model_independent": True, "seeds": sorted(by_seed), "descriptive_only": True, "no_test_tuning": True, "states": aggregate}
    (OUT_DIR / "trace_state_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
