#!/usr/bin/env python3
"""Build the ExecKG v3 full-compilation graph without mutating v2.

The builder treats the 572 StandardGrammarPoint records as authority entries and
adds the complete compiler layer (units, profiles, templates, bindings, contract
tests, handbook page anchors, and catalog corrections).  It validates every
referential edge before publishing the graph.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "kg_v2_full_inventory"
DEFAULT_KG = ROOT / "kg"
DEFAULT_OUT = ROOT / "kg_v3_full_compilation"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_node(node_id: str, node_type: str, label: str, props: dict) -> dict[str, str]:
    return {"id": node_id, "type": node_type, "label": label, "properties_json": stable_json(props)}


def make_edge(source: str, relation: str, target: str, props: dict | None = None) -> dict[str, str]:
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "properties_json": stable_json(props or {}),
    }


def parse_json_cell(value: str):
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def handbook_ref(source_id: str) -> str:
    mapping = {
        "SRC-GRAMMAR-HANDBOOK-ELEM-2022": "REF-GRAMMAR-MANUAL-ELEM",
        "SRC-GRAMMAR-HANDBOOK-INTER-2022": "REF-GRAMMAR-MANUAL-INTER",
        "SRC-GRAMMAR-HANDBOOK-ADV-2022": "REF-GRAMMAR-MANUAL-ADV",
    }
    if source_id not in mapping:
        raise ValueError(f"Unknown handbook source id: {source_id}")
    return mapping[source_id]


def build(base_dir: Path, kg_dir: Path, out_dir: Path) -> dict:
    base_nodes_path = base_dir / "nodes.csv"
    base_edges_path = base_dir / "edges.csv"
    crosswalk_path = kg_dir / "standard_diagnostic_crosswalk_v3.csv"
    templates_path = kg_dir / "diagnostic_templates_v3.csv"
    tests_path = kg_dir / "diagnostic_unit_tests_v3.csv"
    alignment_path = kg_dir / "standard_handbook_alignment_v3.csv"
    corrections_path = kg_dir / "standard_catalog_corrections_v3.csv"

    required = [
        base_nodes_path,
        base_edges_path,
        crosswalk_path,
        templates_path,
        tests_path,
        alignment_path,
        corrections_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input(s): " + ", ".join(missing))

    base_nodes = read_csv(base_nodes_path)
    base_edges = read_csv(base_edges_path)
    crosswalk = read_csv(crosswalk_path)
    templates = read_csv(templates_path)
    tests = read_csv(tests_path)
    alignment = read_csv(alignment_path)
    corrections = read_csv(corrections_path)

    if len({row["standard_point_id"] for row in crosswalk}) != 572:
        raise ValueError("Crosswalk must cover exactly 572 distinct standard points")
    if len(alignment) != 572 or len({row["standard_point_id"] for row in alignment}) != 572:
        raise ValueError("Handbook alignment must contain exactly one row for each of 572 points")

    nodes = list(base_nodes)
    edges = list(base_edges)
    added_nodes: list[dict[str, str]] = []
    added_edges: list[dict[str, str]] = []

    # One reusable node per diagnostic template.
    for row in templates:
        props = {key: parse_json_cell(value) for key, value in row.items() if key != "template_id"}
        added_nodes.append(make_node(row["template_id"], "DiagnosticTemplate", row["template_name"], props))

    # Compiler nodes are deliberately separated to keep authority, diagnostic
    # interpretation, executable binding, and validation state auditable.
    for row in crosswalk:
        std_node = f"standard-point:{row['standard_point_id']}"
        unit_props = {
            "standard_point_id": row["standard_point_id"],
            "unit_index": row["unit_index"],
            "unit_path": row["unit_path"],
            "unit_text": row["unit_text"],
            "split_kind": row["split_kind"],
            "split_marker": row["split_marker"],
            "regime": row["regime"],
            "mapping_disposition": row["mapping_disposition"],
            "source_pdf_page": row["source_pdf_page"],
        }
        profile_props = {
            "slot_schema": parse_json_cell(row["slot_schema"]),
            "positive_constraints": parse_json_cell(row["positive_constraints"]),
            "weak_rejections": parse_json_cell(row["weak_rejections"]),
            "strong_rejections": parse_json_cell(row["strong_rejections"]),
            "confusable_uses": parse_json_cell(row["confusable_uses"]),
            "diagnosis_type_labels": parse_json_cell(row["diagnosis_type_labels"]),
            "structured_output_fields": parse_json_cell(row["structured_output_fields"]),
            "review_status": row["review_status"],
        }
        binding_props = {
            "candidate_pattern": row["candidate_pattern"],
            "candidate_match_mode": row["candidate_match_mode"],
            "candidate_pattern_source": row["candidate_pattern_source"],
            "regex_normalization_status": row["regex_normalization_status"],
            "compilation_status": row["compilation_status"],
            "source_grounding_status": row["source_grounding_status"],
            "static_audit_status": row["static_audit_status"],
            "profile_status": row["profile_status"],
        }
        added_nodes.extend(
            [
                make_node(row["diagnostic_unit_id"], "DiagnosticUnit", row["unit_text"], unit_props),
                make_node(row["diagnostic_profile_id"], "DiagnosticProfile", f"Profile: {row['unit_text']}", profile_props),
                make_node(row["runtime_binding_id"], "RuntimeBinding", f"Binding: {row['unit_text']}", binding_props),
            ]
        )
        added_edges.extend(
            [
                make_edge(std_node, "HAS_DIAGNOSTIC_UNIT", row["diagnostic_unit_id"], {"unit_index": row["unit_index"]}),
                make_edge(row["diagnostic_unit_id"], "HAS_DIAGNOSTIC_PROFILE", row["diagnostic_profile_id"]),
                make_edge(row["diagnostic_profile_id"], "USES_DIAGNOSTIC_TEMPLATE", row["template_id"]),
                make_edge(row["diagnostic_profile_id"], "COMPILES_TO", row["runtime_binding_id"]),
                make_edge(row["runtime_binding_id"], "TRACES_TO", row["source_id"], {"pdf_page": row["source_pdf_page"]}),
            ]
        )

    # Contract tests are represented explicitly so the graph never conflates a
    # compiled binding with candidate-level empirical validation.
    for row in tests:
        props = {key: value for key, value in row.items() if key not in {"test_id", "binding_id"}}
        added_nodes.append(make_node(row["test_id"], "DiagnosticContractTest", row["case_type"], props))
        added_edges.append(make_edge(row["binding_id"], "HAS_CONTRACT_TEST", row["test_id"]))

    # Deduplicate page nodes but retain the point-specific ranking and confidence
    # on each alignment edge.
    page_nodes: dict[str, dict[str, str]] = {}
    page_relations: dict[str, dict[str, str]] = {}
    for row in alignment:
        page_id = f"handbook-page:{row['handbook_source_id']}:{row['primary_page']}"
        page_nodes.setdefault(
            page_id,
            make_node(
                page_id,
                "HandbookPageAnchor",
                f"{row['handbook_source_id']} p.{row['primary_page']}",
                {
                    "handbook_source_id": row["handbook_source_id"],
                    "page": row["primary_page"],
                    "page_text_sha256": row["page_text_sha256"],
                    "text_released": False,
                },
            ),
        )
        page_relations.setdefault(page_id, make_edge(page_id, "PAGE_OF", handbook_ref(row["handbook_source_id"])))
        added_edges.append(
            make_edge(
                f"standard-point:{row['standard_point_id']}",
                "ALIGNED_TO_HANDBOOK_PAGE",
                page_id,
                {
                    "candidate_pages": row["candidate_pages"],
                    "matched_source_derived_keywords": row["matched_source_derived_keywords"],
                    "alignment_score": row["alignment_score"],
                    "score_margin": row["score_margin"],
                    "alignment_method": row["alignment_method"],
                    "alignment_status": row["alignment_status"],
                    "confidence": row["confidence"],
                },
            )
        )
    added_nodes.extend(page_nodes.values())
    added_edges.extend(page_relations.values())

    for index, row in enumerate(corrections, 1):
        correction_id = f"catalog-correction:v3:{index:03d}"
        added_nodes.append(make_node(correction_id, "CatalogCorrection", f"{row['standard_point_id']} {row['field_name']}", row))
        added_edges.append(make_edge(correction_id, "CORRECTS_FIELD_OF", f"standard-point:{row['standard_point_id']}"))

    base_node_ids = {row["id"] for row in base_nodes}
    added_node_ids = [row["id"] for row in added_nodes]
    if len(added_node_ids) != len(set(added_node_ids)):
        counts = Counter(added_node_ids)
        raise ValueError(f"Duplicate added node ids: {[key for key, value in counts.items() if value > 1][:10]}")
    if base_node_ids.intersection(added_node_ids):
        raise ValueError("Added nodes collide with v2 node ids")

    nodes.extend(added_nodes)
    edges.extend(added_edges)
    all_node_ids = {row["id"] for row in nodes}
    dangling = [edge for edge in edges if edge["source"] not in all_node_ids or edge["target"] not in all_node_ids]
    if dangling:
        raise ValueError(f"Dangling edges detected: {dangling[:5]}")

    edge_keys = [(row["source"], row["relation"], row["target"], row["properties_json"]) for row in edges]
    if len(edge_keys) != len(set(edge_keys)):
        raise ValueError("Duplicate graph edges detected")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "nodes.csv", ["id", "type", "label", "properties_json"], nodes)
    write_csv(out_dir / "edges.csv", ["source", "relation", "target", "properties_json"], edges)

    node_types = Counter(row["type"] for row in nodes)
    relation_types = Counter(row["relation"] for row in edges)
    added_node_types = Counter(row["type"] for row in added_nodes)
    added_relation_types = Counter(row["relation"] for row in added_edges)
    test_statuses = Counter(row["execution_status"] for row in tests)
    test_labels = Counter(row["expected_diagnosis_type"] for row in tests)
    alignment_confidence = Counter(row["confidence"] for row in alignment)
    summary = {
        "version": "v3.0-full-compilation",
        "standard": "GF0025-2021",
        "scope": {
            "standard_grammar_points": len({row["standard_point_id"] for row in crosswalk}),
            "diagnostic_units": len(crosswalk),
            "diagnostic_profiles": len({row["diagnostic_profile_id"] for row in crosswalk}),
            "diagnostic_templates": len(templates),
            "runtime_bindings": len({row["runtime_binding_id"] for row in crosswalk}),
            "contract_tests": len(tests),
            "handbook_alignments": len(alignment),
            "catalog_corrections": len(corrections),
        },
        "base_graph": {"directory": str(base_dir), "nodes": len(base_nodes), "edges": len(base_edges)},
        "added": {
            "nodes": len(added_nodes),
            "edges": len(added_edges),
            "node_type_counts": dict(sorted(added_node_types.items())),
            "relation_counts": dict(sorted(added_relation_types.items())),
        },
        "total": {
            "nodes": len(nodes),
            "edges": len(edges),
            "node_type_counts": dict(sorted(node_types.items())),
            "relation_counts": dict(sorted(relation_types.items())),
        },
        "validation": {
            "contract_test_execution_statuses": dict(sorted(test_statuses.items())),
            "contract_test_expected_labels": dict(sorted(test_labels.items())),
            "handbook_alignment_confidence": dict(sorted(alignment_confidence.items())),
            "duplicate_node_ids": 0,
            "duplicate_edges": 0,
            "dangling_edges": 0,
            "v2_node_ids_preserved": len(base_node_ids),
            "v2_edges_preserved": len(base_edges),
            "integrity_pass": True,
        },
        "claim_boundary": (
            "Full inventory coverage denotes source-anchored machine compilation and static/contract validation. "
            "It does not convert the frozen 37-label candidate evaluation into a 572-point behavioral benchmark, "
            "nor does it imply linguist adjudication of every generated profile."
        ),
        "input_sha256": {path.name: digest_file(path) for path in required},
    }
    (out_dir / "graph_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    schema = {
        "version": "v3.0-full-compilation",
        "node_types": sorted(node_types),
        "edge_types": sorted(relation_types),
        "preserved_candidate_path": "CandidateInstance -> GrammarPoint -> GrammarRule -> AuthoritySource",
        "full_compilation_path": (
            "StandardGrammarPoint -> DiagnosticUnit -> DiagnosticProfile -> RuntimeBinding -> AuthoritySource"
        ),
        "template_path": "DiagnosticProfile -> DiagnosticTemplate",
        "validation_path": "RuntimeBinding -> DiagnosticContractTest",
        "handbook_path": "StandardGrammarPoint -> HandbookPageAnchor -> AuthoritySource",
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build(args.base, args.kg, args.out)
    print(json.dumps(summary["scope"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
