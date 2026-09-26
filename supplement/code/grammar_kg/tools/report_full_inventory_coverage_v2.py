#!/usr/bin/env python3
"""Report authority, draft, runtime-binding, and evaluation scopes separately."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
CATALOG = PROJECT_DIR / "kg" / "standard_grammar_points_gf0025_v2.csv"
DRAFTS = PROJECT_DIR / "kg" / "diagnostic_rule_drafts_gf0025_v2.jsonl"
EXPERIMENTAL_LABELS = PROJECT_DIR / "kg" / "grammar_points_v0_1.csv"
RULES = PROJECT_DIR / "kg" / "high_frequency_rule_evidence_v0_3.csv"
BINDINGS = PROJECT_DIR / "kg" / "executable_rule_bindings_v1.json"
GRAPH_SUMMARY = PROJECT_DIR / "kg_v2_full_inventory" / "graph_summary.json"
OUTPUT = PROJECT_DIR / "kg" / "full_inventory_coverage_v2.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    catalog = read_csv(CATALOG)
    experimental = read_csv(EXPERIMENTAL_LABELS)
    rules = read_csv(RULES)
    bindings = json.loads(BINDINGS.read_text(encoding="utf-8"))
    graph_summary = json.loads(GRAPH_SUMMARY.read_text(encoding="utf-8"))
    drafts = [json.loads(line) for line in DRAFTS.read_text(encoding="utf-8").splitlines() if line]

    report = {
        "version": "v2.0",
        "scope_definition": {
            "authority_inventory": "official GF0025-2021 StandardGrammarPoint records",
            "legacy_drafts": "unreviewed surface patterns; never counted as executable",
            "experimental_labels": "candidate-level operational labels used in the frozen evaluation",
            "runtime_bindings": "point-specific bindings within the 37-label experimental space",
        },
        "authority_inventory": {
            "count": len(catalog),
            "denominator": 572,
            "coverage": len(catalog) / 572,
            "unique_ids": len({row["standard_point_id"] for row in catalog}),
            "level_distribution": dict(Counter(row["level"] for row in catalog)),
            "page_anchor_status": dict(Counter(row["page_anchor_status"] for row in catalog)),
        },
        "legacy_drafts": {
            "records": len(drafts),
            "draft_unreviewed": sum(row["draft_status"] == "draft_unreviewed" for row in drafts),
            "not_available": sum(row["draft_status"] == "not_available" for row in drafts),
            "runtime_eligible": sum(bool(row["runtime_eligible"]) for row in drafts),
        },
        "experimental_scope": {
            "diagnostic_labels": len(experimental),
            "typed_rule_records": len(rules),
            "point_specific_runtime_bindings": len(bindings["rules"]),
            "binding_denominator": len(experimental),
            "binding_coverage_within_experimental_labels": len(bindings["rules"]) / len(experimental),
            "official_inventory_crosswalk_denominator_not_assumed": True,
        },
        "materialized_graph": graph_summary["total"],
        "integrity": {
            "authority_inventory_complete": len(catalog) == 572,
            "drafts_do_not_claim_runtime": all(not row["runtime_eligible"] for row in drafts),
            "experimental_and_official_spaces_separated": True,
            "graph_integrity_pass": bool(graph_summary["integrity"]["integrity_pass"]),
        },
    }
    if not all(report["integrity"].values()):
        raise ValueError(f"Coverage integrity failed: {report['integrity']}")
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
