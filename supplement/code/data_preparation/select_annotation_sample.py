#!/usr/bin/env python3
"""Select the fixed, label-free 3,000-candidate annotation sample."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


SEED = 20260629
TEMPORAL_FAMILY = "C7_TEMPORAL_LOCATIVE_FUNCTION"
TARGET_COUNTS = {
    "C1_DISPOSAL_PATIENT_REORDERING": 400,
    "C2_PASSIVE_PATIENT_PROMINENCE": 143,
    "C3_COMPARATIVE_CONSTRUCTION": 320,
    "C4_COMPLEMENT_STRUCTURE": 450,
    "C5_MULTI_VERB_ARGUMENT_STRUCTURE": 260,
    "C6_PREPOSITIONAL_FRAME": 527,
    TEMPORAL_FAMILY: 650,
    "C8_COMPLEX_SENTENCE_CONNECTIVE": 250,
}

BAD_EXPLANATION_TERMS = set(
    "表示 可以 不能 一般 通常 用于 用在 结构 特点 小提示 语义 用法 前边 后边 "
    "成分 主语 谓语 宾语 定语 状语 补语 动词 形容词 名词 词语 句中 句末 "
    "疑问句 肯定形式 否定形式".split()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-candidates", type=Path, required=True)
    parser.add_argument("--temporal-locative-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("formal_rows", "candidates", "rows"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Cannot find candidate rows in {path}")


def family_of(row: dict) -> str:
    return str(row.get("construction_family") or row.get("family_id") or "")


def sentence_of(row: dict) -> str:
    return str(row.get("sentence") or "")


def span_of(row: dict) -> str:
    return str(row.get("candidate_span") or "")


def candidate_quality(row: dict) -> int:
    sentence = sentence_of(row)
    score = 0
    if 8 <= len(sentence) <= 42:
        score += 5
    elif len(sentence) <= 65:
        score += 2
    if any(term in sentence for term in BAD_EXPLANATION_TERMS):
        score -= 5
    if any(char in sentence for char in ["“", "”", "(", ")", "（", "）", "+", "=", ":", "："]):
        score -= 2
    if row.get("grammar_page_clue"):
        score += 1
    if span_of(row) and len(span_of(row)) <= 16:
        score += 1
    if any(token in sentence for token in ["我", "你", "他", "她", "我们", "他们", "老师", "学生", "妈妈", "爸爸", "小王"]):
        score += 2
    return score


def deduplicate(rows: list[dict]) -> list[dict]:
    unique: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (family_of(row), sentence_of(row), span_of(row))
        if key not in unique:
            unique[key] = dict(row)
        elif row.get("grammar_page_clue") and not unique[key].get("grammar_page_clue"):
            unique[key] = dict(row)
    return list(unique.values())


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    structured = deduplicate(load_rows(args.structured_candidates))
    temporal = [
        row for row in deduplicate(load_rows(args.temporal_locative_candidates))
        if family_of(row) == TEMPORAL_FAMILY and row.get("source_status", "AUTH_READY") == "AUTH_READY"
    ]

    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in structured:
        by_family[family_of(row)].append(row)
    for rows in by_family.values():
        rng.shuffle(rows)
        rows.sort(key=candidate_quality, reverse=True)

    if len(temporal) < TARGET_COUNTS[TEMPORAL_FAMILY]:
        raise RuntimeError(f"Need 650 temporal/locative candidates, found {len(temporal)}")

    selected = [dict(row) for row in temporal[: TARGET_COUNTS[TEMPORAL_FAMILY]]]
    for family, target in TARGET_COUNTS.items():
        if family == TEMPORAL_FAMILY:
            continue
        available = by_family[family]
        if len(available) < target:
            raise RuntimeError(f"Not enough candidates for {family}: {len(available)} < {target}")
        selected.extend(dict(row) for row in available[:target])

    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["formal_id"] = f"ASGD-2027-{index:04d}"
        row["suggested_batch"] = f"第{((index - 1) % 6) + 1}批"

    payload = {
        "metadata": {
            "dataset_name": "ASGD-3000",
            "version": "preannotation_selection_v1",
            "seed": args.seed,
            "selection_uses_gold_labels": False,
            "formal_row_count": len(selected),
            "family_quotas": TARGET_COUNTS,
        },
        "formal_rows": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
