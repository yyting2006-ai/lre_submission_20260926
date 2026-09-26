#!/usr/bin/env python3
"""Audit handbook page anchors without releasing copyrighted page text."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import fitz


BOOK_LABELS = {"初等": "elementary", "中等": "intermediate", "高等": "advanced"}
CORE_MARKERS = {
    "把字句候选": ["把"], "将字处置候选": ["将"], "被字句候选": ["被"],
    "时间状语候选": ["时间", "时候", "时"], "处所状语候选": ["处所", "地点", "在"],
    "得字补语候选": ["得"], "比字比较候选": ["比"], "没有比较候选": ["没有"],
    "越...越候选": ["越"], "不如比较候选": ["不如"], "一样比较候选": ["一样"],
    "在字介词框架候选": ["在"], "从字介词框架候选": ["从"], "对字介词框架候选": ["对"],
    "向/往/朝介词框架候选": ["向", "往", "朝"],
    "给/为/关于等介词框架候选": ["给", "为", "关于"],
}
SECTION_CUES = ["基本语义及用法", "结构特点", "小提示"]
STATUS_BY_POLARITY = {
    "正向接受规则": ("source-grounded schema transcription", "页面已复核；本条为手册结构/用法说明的结构化转写，并非逐字引文。"),
    "弱拒绝规则": ("computational boundary derivation", "基础结构页面已复核；本条为依据必要结构派生的缺槽或边界规则。"),
    "强拒绝规则": ("operational rejection rule", "基础结构页面已复核；本条为候选治理的操作性排除规则，不主张文献逐字给出。"),
    "偏误推演规则": ("derived diagnostic rule", "基础结构页面已复核；本条为从结构条件派生的诊断规则，不主张文献逐字给出。"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--elementary", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, required=True)
    parser.add_argument("--advanced", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--update-rules", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def page_anchors(raw: str) -> list[tuple[str, int]]:
    return [(BOOK_LABELS[level], int(page)) for level, page in re.findall(r"(初等|中等|高等)\s*p\.(\d+)", raw or "")]


def main() -> None:
    args = parse_args()
    rows = read_csv(args.rules)
    documents = {"elementary": fitz.open(args.elementary), "intermediate": fitz.open(args.intermediate), "advanced": fitz.open(args.advanced)}
    cache: dict[tuple[str, int], str] = {}
    by_point: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        by_point[row["候选语法点"]].update(page_anchors(row["规则说明页码"]))
    audit_rows: list[dict] = []
    verified_points: set[str] = set()
    for point in sorted(by_point):
        anchors = sorted(by_point[point]); existing = marker_pages = definition_like_pages = 0; section_cues: set[str] = set()
        for book, page_number in anchors:
            document = documents[book]
            if not 1 <= page_number <= len(document): continue
            existing += 1; key = (book, page_number)
            if key not in cache: cache[key] = document[page_number - 1].get_text("text")
            text = cache[key]; has_marker = any(marker in text for marker in CORE_MARKERS.get(point, [])); cues = {cue for cue in SECTION_CUES if cue in text}
            marker_pages += int(has_marker); definition_like_pages += int(has_marker and bool(cues)); section_cues.update(cues)
        verified = existing == len(anchors) and definition_like_pages >= 1
        if verified: verified_points.add(point)
        audit_rows.append({"grammar_point": point, "rule_count": sum(row["候选语法点"] == point for row in rows), "anchor_count": len(anchors), "existing_page_count": existing, "marker_page_count": marker_pages, "definition_like_page_count": definition_like_pages, "section_cues_found": ";".join(sorted(section_cues)), "page_anchor_status": "verified" if verified else "needs_manual_review"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "rule_page_anchor_audit.csv", audit_rows, list(audit_rows[0]))
    summary = {"audit_version": "v1.0", "rules": len(rows), "grammar_points": len(by_point), "verified_grammar_points": len(verified_points), "all_points_verified": len(verified_points) == len(by_point), "definition_like_page_criterion": "At least one anchored page contains a point marker and one of: basic meaning/use, structural characteristics, or tip.", "copyright_note": "No handbook page text is stored in the audit output."}
    (args.output_dir / "rule_page_anchor_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.update_rules:
        for row in rows:
            evidence_class, status = STATUS_BY_POLARITY.get(row["规则层级"], ("unclassified", "页面审计已完成；规则来源类型待人工分类。"))
            if row["候选语法点"] not in verified_points: status = "页面自动核验未通过，需人工复核后方可作为来源证据。"
            row["证据状态"] = status; row["待补动作"] = f"证据类型={evidence_class}；保留规则转写判断与版本记录。"
        write_csv(args.update_rules, rows, list(rows[0]))
    for document in documents.values(): document.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
