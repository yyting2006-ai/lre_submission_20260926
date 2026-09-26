#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from itertools import combinations
from pathlib import Path


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
BASE = ROOT / "annotation_qc/outputs/formal_qc_20260718"
OUT_DIR = BASE / "agreement_audit"

FILES = {
    "main": BASE / "gold_main_by_filename_majority_annotator2_20260718.csv",
    "evidence": BASE / "gold_evidence_by_filename_majority_annotator2_20260718.csv",
    "slot": BASE / "gold_slot_span_by_filename_majority_annotator2_20260718.csv",
}

FIELDS = {
    "main": [
        "候选提取是否正确",
        "主标签：是否为目标语法点",
        "功能/问题类型",
    ],
    "evidence": [
        "结构槽位是否支持",
        "缺失/异常槽位",
        "易混对象",
    ],
    "slot": ["槽位状态"],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    assert len(a) == len(b)
    if not a:
        return 0.0, 0.0
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    labels = set(a) | set(b)
    counts_a = Counter(a)
    counts_b = Counter(b)
    expected = sum((counts_a[label] / len(a)) * (counts_b[label] / len(b)) for label in labels)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return observed, kappa


def fleiss_kappa(ratings: list[list[str]]) -> tuple[float, float, float]:
    if not ratings:
        return 0.0, 0.0, 0.0
    n_raters = len(ratings[0])
    labels = sorted({label for item in ratings for label in item})
    per_item_agreement = []
    total = Counter()
    for item in ratings:
        counts = Counter(item)
        total.update(item)
        agreement = (sum(count * count for count in counts.values()) - n_raters) / (
            n_raters * (n_raters - 1)
        )
        per_item_agreement.append(agreement)
    observed = sum(per_item_agreement) / len(per_item_agreement)
    denom = len(ratings) * n_raters
    expected = sum((total[label] / denom) ** 2 for label in labels)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    # Gwet's AC1 is reported alongside kappa because the labels are strongly
    # prevalence-skewed; it is not used as a replacement for raw agreement.
    ac1_expected = sum(
        (total[label] / denom) * (1 - total[label] / denom) for label in labels
    ) / max(1, len(labels) - 1)
    ac1 = (observed - ac1_expected) / (1 - ac1_expected) if ac1_expected < 1 else 1.0
    return observed, kappa, ac1


def overlap_match(a: str, b: str) -> bool:
    a = a.strip()
    b = b.strip()
    return bool(a and b and (a in b or b in a))


def audit() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    pairwise: list[dict] = []
    distributions: list[dict] = []

    datasets = {name: read_csv(path) for name, path in FILES.items()}
    for section, rows in datasets.items():
        for field in FIELDS[section]:
            columns = [f"标注者{i}_{field}" for i in (1, 2, 3)]
            ratings = [[row[column].strip() for column in columns] for row in rows]
            observed, kappa, ac1 = fleiss_kappa(ratings)
            three_way = sum(len(set(item)) == 1 for item in ratings) / len(ratings)
            majority = sum(Counter(item).most_common(1)[0][1] >= 2 for item in ratings) / len(ratings)
            summary.append({
                "section": section,
                "field": field,
                "n": len(rows),
                "three_way_exact_agreement": three_way,
                "at_least_two_agree": majority,
                "fleiss_observed_pair_agreement": observed,
                "fleiss_kappa": kappa,
                "gwet_ac1": ac1,
            })
            for left, right in combinations(range(3), 2):
                a = [item[left] for item in ratings]
                b = [item[right] for item in ratings]
                pair_observed, pair_kappa = cohen_kappa(a, b)
                pairwise.append({
                    "section": section,
                    "field": field,
                    "annotator_pair": f"{left + 1}-{right + 1}",
                    "n": len(rows),
                    "exact_agreement": pair_observed,
                    "cohen_kappa": pair_kappa,
                })
            for annotator, column in enumerate(columns, start=1):
                for label, count in Counter(row[column].strip() for row in rows).most_common():
                    distributions.append({
                        "section": section,
                        "field": field,
                        "annotator": annotator,
                        "label": label,
                        "count": count,
                        "proportion": count / len(rows),
                    })

    slot_rows = datasets["slot"]
    span_pairwise = []
    span_field = "槽位片段（从句子中复制；条件必填）"
    status_field = "槽位状态"
    for left, right in combinations((1, 2, 3), 2):
        eligible = [
            row for row in slot_rows
            if row[f"标注者{left}_{status_field}"] == "已标出"
            and row[f"标注者{right}_{status_field}"] == "已标出"
        ]
        exact = sum(
            row[f"标注者{left}_{span_field}"].strip() == row[f"标注者{right}_{span_field}"].strip()
            for row in eligible
        )
        relaxed = sum(
            overlap_match(
                row[f"标注者{left}_{span_field}"],
                row[f"标注者{right}_{span_field}"],
            ) for row in eligible
        )
        span_pairwise.append({
            "annotator_pair": f"{left}-{right}",
            "both_marked_n": len(eligible),
            "exact_span_agreement": exact / len(eligible) if eligible else 0.0,
            "relaxed_span_agreement": relaxed / len(eligible) if eligible else 0.0,
        })

    adjudication = []
    final_fields = {
        "main": [
            ("候选提取是否正确", "候选提取是否正确裁决来源"),
            ("主标签：是否为目标语法点", "主标签：是否为目标语法点裁决来源"),
            ("功能/问题类型", "功能/问题类型裁决来源"),
        ],
        "evidence": [
            ("结构槽位是否支持", "结构槽位是否支持裁决来源"),
            ("缺失/异常槽位", "缺失/异常槽位裁决来源"),
            ("易混对象", "易混对象裁决来源"),
        ],
        "slot": [
            ("槽位状态", "槽位状态裁决来源"),
            (span_field, f"{span_field}裁决来源"),
        ],
    }
    for section, fields in final_fields.items():
        rows = datasets[section]
        for field, source_column in fields:
            for source, count in Counter(row[source_column] for row in rows).most_common():
                adjudication.append({
                    "section": section,
                    "field": field,
                    "decision_source": source,
                    "count": count,
                    "proportion": count / len(rows),
                })

    write_csv(OUT_DIR / "agreement_summary.csv", summary)
    write_csv(OUT_DIR / "pairwise_kappa.csv", pairwise)
    write_csv(OUT_DIR / "label_distributions.csv", distributions)
    write_csv(OUT_DIR / "slot_span_pairwise_agreement.csv", span_pairwise)
    write_csv(OUT_DIR / "adjudication_source_summary.csv", adjudication)
    payload = {
        "agreement_summary": summary,
        "pairwise": pairwise,
        "slot_span_pairwise": span_pairwise,
        "adjudication": adjudication,
    }
    (OUT_DIR / "agreement_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Annotation Agreement Audit",
        "",
        "Agreement is computed on the three independent annotation columns before adjudication.",
        "",
        "| Section | Field | N | Three-way exact | Two-or-more agree | Fleiss kappa | Gwet AC1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['section']} | {row['field']} | {row['n']} | "
            f"{row['three_way_exact_agreement']:.4f} | {row['at_least_two_agree']:.4f} | "
            f"{row['fleiss_kappa']:.4f} | {row['gwet_ac1']:.4f} |"
        )
    lines += [
        "",
        "## Slot spans",
        "",
        "Span agreement is conditioned on both annotators marking the slot as present.",
        "",
        "| Pair | Both marked | Exact | Relaxed containment |",
        "|---|---:|---:|---:|",
    ]
    for row in span_pairwise:
        lines.append(
            f"| {row['annotator_pair']} | {row['both_marked_n']} | "
            f"{row['exact_span_agreement']:.4f} | {row['relaxed_span_agreement']:.4f} |"
        )
    (OUT_DIR / "agreement_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(OUT_DIR),
        "fields_audited": len(summary),
        "slot_pairs": len(span_pairwise),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    audit()
