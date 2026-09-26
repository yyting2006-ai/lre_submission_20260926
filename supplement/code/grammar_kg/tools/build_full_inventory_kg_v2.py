#!/usr/bin/env python3
"""Build the versioned ExecKG v2 authority inventory graph.

The v2 graph is an additive extension of ``kg_v1_candidate_evidence``.  It
copies every v1 node and edge unchanged, then adds the complete 572-point
GF0025-2021 inventory as a separate ``StandardGrammarPoint`` namespace.
The 37 experimental ``GrammarPoint`` nodes therefore retain their original
meaning and IDs.

The input catalog is expected at::

    kg/standard_grammar_points_gf0025_v2.csv

Preferred columns are ``standard_point_id``, ``grammar_point``, ``level``,
``category``, ``subcategory``, ``source_id``, ``source_label``,
``page_anchor``, and ``source_status``.  A small set of documented English
and Chinese aliases is accepted by :func:`catalog_value` below.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_V1_DIR = PROJECT_DIR / "kg_v1_candidate_evidence"
DEFAULT_CATALOG = PROJECT_DIR / "kg" / "standard_grammar_points_gf0025_v2.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "kg_v2_full_inventory"

NODE_FIELDS = ["id", "type", "label", "properties_json"]
EDGE_FIELDS = ["source", "relation", "target", "properties_json"]
EXPECTED_LEVEL_COUNTS = {
    "1": 48,
    "2": 81,
    "3": 81,
    "4": 76,
    "5": 71,
    "6": 67,
    "7-9": 148,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = []
        for raw in reader:
            rows.append({str(key).strip(): (value or "").strip() for key, value in raw.items()})
        return rows


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def canonical_properties(value: str) -> str:
    """Validate and canonically serialize an existing properties object."""
    payload = json.loads(value or "{}")
    if not isinstance(payload, dict):
        raise ValueError(f"properties_json must contain an object, got {type(payload).__name__}")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def properties(**kwargs: str) -> str:
    return json.dumps(
        {key: value for key, value in kwargs.items() if value not in {None, ""}},
        ensure_ascii=False,
        sort_keys=True,
    )


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def catalog_value(row: dict[str, str], *aliases: str, required: bool = False) -> str:
    for alias in aliases:
        value = row.get(alias, "").strip()
        if value:
            return value
    if required:
        raise ValueError(f"Catalog row lacks required field {aliases}: {row}")
    return ""


def normalize_level(raw: str) -> str:
    compact = raw.strip().lower().replace("level", "").replace("级", "").replace(" ", "")
    compact = compact.replace("—", "-").replace("–", "-").replace("~", "-").replace("至", "-")
    chinese = {
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七-九": "7-9",
        "七九": "7-9",
        "高等": "7-9",
    }
    compact = chinese.get(compact, compact)
    if compact in {"7", "8", "9", "7-8-9", "7/8/9", "7、8、9"}:
        compact = "7-9"
    if compact not in EXPECTED_LEVEL_COUNTS:
        raise ValueError(f"Unrecognized standard level: {raw!r}")
    return compact


def assert_unique_rows(rows: list[dict[str, str]], fields: list[str], name: str) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        raise ValueError(f"{name} contains {duplicates} duplicate row(s) by {fields}")


def load_v1(v1_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    nodes = read_csv(v1_dir / "nodes.csv")
    edges = read_csv(v1_dir / "edges.csv")
    for row in nodes:
        if set(row) != set(NODE_FIELDS):
            raise ValueError(f"Unexpected v1 node columns: {list(row)}")
        row["properties_json"] = canonical_properties(row["properties_json"])
    for row in edges:
        if set(row) != set(EDGE_FIELDS):
            raise ValueError(f"Unexpected v1 edge columns: {list(row)}")
        row["properties_json"] = canonical_properties(row["properties_json"])
    assert_unique_rows(nodes, ["id"], "v1 nodes")
    assert_unique_rows(edges, EDGE_FIELDS, "v1 edges")
    return nodes, edges


def add_node(nodes: dict[str, dict[str, str]], row: dict[str, str]) -> bool:
    previous = nodes.get(row["id"])
    if previous is not None:
        if previous != row:
            raise ValueError(f"Conflicting node definition for {row['id']}")
        return False
    nodes[row["id"]] = row
    return True


def add_edge(edges: dict[tuple[str, str, str, str], dict[str, str]], row: dict[str, str]) -> bool:
    key = tuple(row[field] for field in EDGE_FIELDS)
    if key in edges:
        return False
    edges[key] = row
    return True


def build(v1_dir: Path, catalog_path: Path, out_dir: Path) -> dict:
    v1_nodes, v1_edges = load_v1(v1_dir)
    catalog = read_csv(catalog_path)
    if len(catalog) != 572:
        raise ValueError(f"Expected exactly 572 catalog rows, found {len(catalog)}")

    nodes = {row["id"]: row for row in v1_nodes}
    edges = {tuple(row[field] for field in EDGE_FIELDS): row for row in v1_edges}
    original_node_ids = set(nodes)
    original_edge_keys = set(edges)

    point_records: list[dict[str, str]] = []
    point_catalog_ids: set[str] = set()
    category_values: set[str] = set()
    subcategory_values: set[str] = set()
    category_subcategory_pairs: set[tuple[str, str]] = set()
    level_counts: Counter[str] = Counter()

    for index, row in enumerate(catalog, start=2):
        try:
            catalog_id = catalog_value(
                row,
                "standard_point_id", "standard_grammar_point_id", "point_id",
                "标准语法点ID", "语法点ID", "编号", required=True,
            )
            label = catalog_value(
                row,
                "grammar_point", "grammar_point_name", "canonical_label", "official_label", "label",
                "语法点", "语法点名称", "名称", required=True,
            )
            level = normalize_level(catalog_value(row, "level", "standard_level", "等级", required=True))
            category = catalog_value(
                row, "category", "primary_category", "top_category", "一级分类", required=True,
            )
            subcategory = catalog_value(
                row, "subcategory", "secondary_category", "二级分类", required=True,
            )
            source_id = catalog_value(
                row, "source_id", "authority_source_id", "来源ID", "文献ID", required=True,
            )
        except ValueError as exc:
            raise ValueError(f"Catalog line {index}: {exc}") from exc

        if catalog_id in point_catalog_ids:
            raise ValueError(f"Duplicate standard point ID at catalog line {index}: {catalog_id}")
        point_catalog_ids.add(catalog_id)
        level_counts[level] += 1
        category_values.add(category)
        subcategory_values.add(subcategory)
        category_subcategory_pairs.add((category, subcategory))
        point_records.append(
            {
                "catalog_id": catalog_id,
                "label": label,
                "level": level,
                "category": category,
                "subcategory": subcategory,
                "source_id": source_id,
                "source_label": catalog_value(
                    row, "source_label", "authority_source", "来源", "文献名称"
                ) or source_id,
                "page_anchor": catalog_value(row, "page_anchor", "page", "页码", "来源页码"),
                "source_status": catalog_value(row, "source_status", "证据状态", "状态"),
                "standard_code": catalog_value(row, "standard_code", "code", "标准编号"),
            }
        )

    if dict(level_counts) != EXPECTED_LEVEL_COUNTS:
        raise ValueError(
            "GF0025 level distribution mismatch: "
            f"expected {EXPECTED_LEVEL_COUNTS}, got {dict(sorted(level_counts.items()))}"
        )
    if len(category_values) != 12:
        raise ValueError(f"Expected 12 primary categories, found {len(category_values)}")
    if len(subcategory_values) != 33:
        raise ValueError(f"Expected 33 category-qualified subcategories, found {len(subcategory_values)}")

    added_by_type: Counter[str] = Counter()
    added_by_relation: Counter[str] = Counter()

    for level in EXPECTED_LEVEL_COUNTS:
        node = {
            "id": f"standard-level:{level}",
            "type": "StandardLevel",
            "label": f"Level {level}",
            "properties_json": properties(
                level=level,
                expected_point_count=str(EXPECTED_LEVEL_COUNTS[level]),
                standard="GF0025-2021",
            ),
        }
        if add_node(nodes, node):
            added_by_type[node["type"]] += 1

    category_ids: dict[str, str] = {}
    for category in sorted(category_values):
        node_id = stable_id("standard-category", category)
        category_ids[category] = node_id
        node = {
            "id": node_id,
            "type": "StandardCategory",
            "label": category,
            "properties_json": properties(taxonomy_level="primary", standard="GF0025-2021"),
        }
        if add_node(nodes, node):
            added_by_type[node["type"]] += 1

    subcategory_ids: dict[str, str] = {}
    subcategory_parents: dict[str, list[str]] = {
        subcategory: sorted(
            category
            for category, candidate_subcategory in category_subcategory_pairs
            if candidate_subcategory == subcategory
        )
        for subcategory in subcategory_values
    }
    for subcategory in sorted(subcategory_values):
        node_id = stable_id("standard-subcategory", subcategory)
        subcategory_ids[subcategory] = node_id
        node = {
            "id": node_id,
            "type": "StandardSubcategory",
            "label": subcategory,
            "properties_json": properties(
                taxonomy_level="secondary",
                primary_categories=" | ".join(subcategory_parents[subcategory]),
                standard="GF0025-2021",
            ),
        }
        if add_node(nodes, node):
            added_by_type[node["type"]] += 1

    for category, subcategory in sorted(category_subcategory_pairs):
        edge = {
            "source": subcategory_ids[subcategory],
            "relation": "SUBCATEGORY_OF",
            "target": category_ids[category],
            "properties_json": "{}",
        }
        if add_edge(edges, edge):
            added_by_relation[edge["relation"]] += 1

    for record in point_records:
        source_id = record["source_id"]
        existing_source = nodes.get(source_id)
        if existing_source is None:
            source_node = {
                "id": source_id,
                "type": "AuthoritySource",
                "label": record["source_label"],
                "properties_json": properties(
                    standard="GF0025-2021", source_status=record["source_status"]
                ),
            }
            if add_node(nodes, source_node):
                added_by_type[source_node["type"]] += 1
        elif existing_source["type"] != "AuthoritySource":
            raise ValueError(
                f"Catalog source ID {source_id!r} collides with v1 {existing_source['type']} node"
            )

        point_id = f"standard-point:{record['catalog_id']}"
        point_node = {
            "id": point_id,
            "type": "StandardGrammarPoint",
            "label": record["label"],
            "properties_json": properties(
                catalog_id=record["catalog_id"],
                standard_code=record["standard_code"],
                level=record["level"],
                category=record["category"],
                subcategory=record["subcategory"],
                source_status=record["source_status"],
                inventory_scope="authority layer",
            ),
        }
        if add_node(nodes, point_node):
            added_by_type[point_node["type"]] += 1

        point_edges = [
            {
                "source": point_id,
                "relation": "IN_STANDARD_LEVEL",
                "target": f"standard-level:{record['level']}",
                "properties_json": "{}",
            },
            {
                "source": point_id,
                "relation": "BELONGS_TO_STANDARD_CATEGORY",
                "target": category_ids[record["category"]],
                "properties_json": "{}",
            },
            {
                "source": point_id,
                "relation": "BELONGS_TO_STANDARD_SUBCATEGORY",
                "target": subcategory_ids[record["subcategory"]],
                "properties_json": "{}",
            },
            {
                "source": point_id,
                "relation": "DEFINED_BY",
                "target": source_id,
                "properties_json": properties(
                    page_anchor=record["page_anchor"], source_status=record["source_status"]
                ),
            },
        ]
        for edge in point_edges:
            if add_edge(edges, edge):
                added_by_relation[edge["relation"]] += 1

    # The v1 experimental graph must remain byte-semantically unchanged.
    for node_id in original_node_ids:
        original = next(row for row in v1_nodes if row["id"] == node_id)
        if nodes[node_id] != original:
            raise AssertionError(f"v1 node was modified: {node_id}")
    if not original_edge_keys.issubset(edges):
        raise AssertionError("One or more v1 edges were removed or modified")

    node_rows = sorted(nodes.values(), key=lambda row: row["id"])
    edge_rows = sorted(
        edges.values(), key=lambda row: (row["source"], row["relation"], row["target"], row["properties_json"])
    )
    assert_unique_rows(node_rows, ["id"], "v2 nodes")
    assert_unique_rows(edge_rows, EDGE_FIELDS, "v2 edges")
    node_ids = set(nodes)
    dangling = [row for row in edge_rows if row["source"] not in node_ids or row["target"] not in node_ids]
    if dangling:
        raise ValueError(f"v2 graph contains {len(dangling)} dangling edge(s)")

    v1_type_counts = Counter(row["type"] for row in v1_nodes)
    v2_type_counts = Counter(row["type"] for row in node_rows)
    v1_relation_counts = Counter(row["relation"] for row in v1_edges)
    v2_relation_counts = Counter(row["relation"] for row in edge_rows)

    summary = {
        "version": "v2.0-full-inventory",
        "standard": "GF0025-2021",
        "inventory_scope": {
            "standard_grammar_points": 572,
            "experimental_grammar_points_preserved": v2_type_counts["GrammarPoint"],
            "standard_levels": len(EXPECTED_LEVEL_COUNTS),
            "primary_categories": len(category_values),
            "secondary_categories": len(subcategory_values),
            "category_subcategory_memberships": len(category_subcategory_pairs),
            "level_distribution": dict(EXPECTED_LEVEL_COUNTS),
        },
        "base_graph": {
            "directory": str(v1_dir.resolve()),
            "nodes": len(v1_nodes),
            "edges": len(v1_edges),
            "node_type_counts": dict(sorted(v1_type_counts.items())),
            "relation_counts": dict(sorted(v1_relation_counts.items())),
        },
        "added": {
            "nodes": len(node_rows) - len(v1_nodes),
            "edges": len(edge_rows) - len(v1_edges),
            "node_type_counts": dict(sorted(added_by_type.items())),
            "relation_counts": dict(sorted(added_by_relation.items())),
        },
        "total": {
            "nodes": len(node_rows),
            "edges": len(edge_rows),
            "node_type_counts": dict(sorted(v2_type_counts.items())),
            "relation_counts": dict(sorted(v2_relation_counts.items())),
        },
        "integrity": {
            "catalog_rows": len(catalog),
            "unique_standard_point_ids": len(point_catalog_ids),
            "duplicate_node_ids": len(node_rows) - len({row["id"] for row in node_rows}),
            "duplicate_edges": len(edge_rows) - len({tuple(row[field] for field in EDGE_FIELDS) for row in edge_rows}),
            "dangling_edges": len(dangling),
            "v1_nodes_preserved": len(original_node_ids),
            "v1_edges_preserved": len(original_edge_keys),
            "integrity_pass": True,
        },
    }

    schema = {
        "version": "v2.0-full-inventory",
        "node_types": sorted(v2_type_counts),
        "edge_types": sorted(v2_relation_counts),
        "preserved_v1_primary_path": "CandidateInstance -> GrammarPoint -> GrammarRule -> AuthoritySource",
        "standard_authority_paths": [
            "StandardGrammarPoint -> StandardLevel",
            "StandardGrammarPoint -> StandardCategory",
            "StandardGrammarPoint -> StandardSubcategory -> StandardCategory",
            "StandardGrammarPoint -> AuthoritySource",
        ],
        "scope_note": (
            "StandardGrammarPoint records represent authority-layer inventory coverage. "
            "They do not imply executable runtime bindings or evaluation coverage."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "nodes.csv", node_rows, NODE_FIELDS)
    write_csv(out_dir / "edges.csv", edge_rows, EDGE_FIELDS)
    write_json(out_dir / "graph_summary.json", summary)
    write_json(out_dir / "schema.json", schema)
    return summary


def main() -> None:
    args = parse_args()
    summary = build(args.v1_dir.resolve(), args.catalog.resolve(), args.out_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
