#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
sys.path.insert(0, str(ROOT / "asgd_fast_experiments/scripts"))

from run_grouped_kg_experiments import KG, sentence_group_id  # noqa: E402


GOLD_DIR = ROOT / "annotation_qc/outputs/formal_qc_20260718"
MAIN_CSV = GOLD_DIR / "gold_main_by_filename_majority_annotator2_20260718.csv"
EVIDENCE_CSV = GOLD_DIR / "gold_evidence_by_filename_majority_annotator2_20260718.csv"
SLOT_CSV = GOLD_DIR / "gold_slot_span_by_filename_majority_annotator2_20260718.csv"
KG_DIR = ROOT / "grammar_kg/kg"
POINT_CSV = KG_DIR / "grammar_points_v0_1.csv"
RULE_CSV = KG_DIR / "high_frequency_rule_evidence_v0_3.csv"
RELIABILITY_CSV = ROOT / "asgd_fast_experiments/outputs/exp_20260718_grouped_kg_v2/rule_reliability_by_seed.csv"
OUT_DIR = ROOT / "grammar_kg/kg_v1_candidate_evidence"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def properties(**kwargs) -> str:
    return json.dumps(
        {key: value for key, value in kwargs.items() if value not in {None, ""}},
        ensure_ascii=False,
        sort_keys=True,
    )


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str, str, str], dict] = {}

    def node(self, node_id: str, node_type: str, label: str, **props) -> str:
        row = {
            "id": node_id,
            "type": node_type,
            "label": label,
            "properties_json": properties(**props),
        }
        previous = self.nodes.get(node_id)
        if previous and previous != row:
            raise ValueError(f"Conflicting node definition: {node_id}")
        self.nodes[node_id] = row
        return node_id

    def edge(self, source: str, relation: str, target: str, **props) -> None:
        prop_json = properties(**props)
        key = (source, relation, target, prop_json)
        self.edges[key] = {
            "source": source,
            "relation": relation,
            "target": target,
            "properties_json": prop_json,
        }

    def write(self) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with (OUT_DIR / "nodes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "type", "label", "properties_json"])
            writer.writeheader()
            writer.writerows(sorted(self.nodes.values(), key=lambda row: row["id"]))
        with (OUT_DIR / "edges.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "relation", "target", "properties_json"])
            writer.writeheader()
            writer.writerows(sorted(self.edges.values(), key=lambda row: (row["source"], row["relation"], row["target"])))


def candidate_id(row_id: str) -> str:
    return f"candidate:{row_id}"


def source_node_id(source_name: str) -> str:
    return stable_id("source", source_name.strip() or "UNKNOWN")


def build() -> dict:
    graph = GraphBuilder()
    main_rows = read_csv(MAIN_CSV)
    evidence_rows = read_csv(EVIDENCE_CSV)
    slot_rows = read_csv(SLOT_CSV)
    point_rows = read_csv(POINT_CSV)
    rule_rows = read_csv(RULE_CSV)
    reliability_rows = read_csv(RELIABILITY_CSV) if RELIABILITY_CSV.exists() else []

    point_id_by_candidate: dict[str, str] = {}
    for row in point_rows:
        point_id = row["grammar_point_id"]
        point_id_by_candidate[row["candidate_label"]] = point_id
        graph.node(
            point_id,
            "GrammarPoint",
            row["canonical_label"],
            candidate_label=row["candidate_label"],
            family=row["family"],
            sample_count=row["sample_count"],
            markers=row["markers"],
            source_status=row["source_status"],
        )
        family_id = stable_id("family", row["family"])
        graph.node(family_id, "GrammarFamily", row["family"])
        graph.edge(point_id, "BELONGS_TO_FAMILY", family_id)

    source_descriptions: dict[str, str] = {}
    for row in rule_rows:
        rule_id = f"rule:{row['规则ID']}"
        graph.node(
            rule_id,
            "GrammarRule",
            row["规则文本"],
            rule_id=row["规则ID"],
            rule_group=row["规则组"],
            rule_level=row["规则层级"],
            candidate_grammar_point=row["候选语法点"],
            page_anchor=row["规则说明页码"],
            keywords=row["规则关键词"],
            evidence_status=row["证据状态"],
        )
        point_id = point_id_by_candidate.get(row["候选语法点"])
        if point_id:
            graph.edge(point_id, "GOVERNED_BY", rule_id, rule_level=row["规则层级"])
        for source_id in [item for item in row["支撑文献ID"].split("；") if item]:
            graph.node(source_id, "AuthoritySource", source_id)
            source_descriptions[source_id] = row["支撑文献说明"]
            graph.edge(
                rule_id,
                "CITES",
                source_id,
                page_anchor=row["规则说明页码"],
                evidence_status=row["证据状态"],
            )

    for source_id, description in source_descriptions.items():
        current = graph.nodes[source_id]
        graph.nodes[source_id] = {
            **current,
            "properties_json": properties(description=description),
        }

    for row in reliability_rows:
        rule_id = f"rule:{row['rule_id']}"
        if rule_id not in graph.nodes:
            graph.node(rule_id, "RuntimeRule", row["rule_id"], rule_id=row["rule_id"])
        reliability_id = f"reliability:{row['task']}:{row['seed']}:{row['rule_id']}"
        graph.node(
            reliability_id,
            "RuleReliability",
            f"{row['rule_id']} {row['task']} seed {row['seed']}",
            task=row["task"],
            split_seed=row["seed"],
            signal_type=row["signal_type"],
            train_support=row["support"],
            train_correct=row["correct"],
            posterior_reliability=row["posterior_reliability"],
            reliability_grade=row["grade"],
            effective_weight=row["effective_weight"],
            estimation_scope="training split only",
        )
        graph.edge(rule_id, "HAS_TRAIN_RELIABILITY", reliability_id, task=row["task"], seed=row["seed"])

    main_candidate_ids: set[str] = set()
    for row in main_rows:
        cid = candidate_id(row["编号"])
        main_candidate_ids.add(cid)
        graph.node(
            cid,
            "CandidateInstance",
            row["候选片段"],
            row_id=row["编号"],
            candidate_span=row["候选片段"],
            candidate_grammar_point=row["候选语法点"],
            grammar_family=row["语法家族"],
            gold_confidence_tier=row["金标置信层级"],
            evidence_audit_flag=row["结构证据抽检"],
        )
        sid = f"sentence:{sentence_group_id(row['句子'])}"
        graph.node(sid, "SentenceGroup", row["句子"], sentence=row["句子"])
        graph.edge(cid, "OCCURS_IN", sid)

        point_id = point_id_by_candidate.get(row["候选语法点"])
        if point_id:
            graph.edge(cid, "CANDIDATE_OF", point_id)
        family_id = stable_id("family", row["语法家族"])
        graph.node(family_id, "GrammarFamily", row["语法家族"])
        graph.edge(cid, "IN_FAMILY", family_id)

        source_name = row["书名/文献名"] or "UNKNOWN"
        source_id = source_node_id(source_name)
        graph.node(source_id, "AuthoritySource", source_name, source_name=source_name)
        graph.edge(cid, "DERIVED_FROM", source_id, page=row["页码"])

        consensus_id = f"consensus:main:{row['编号']}"
        graph.node(
            consensus_id,
            "ConsensusDecision",
            row["最终主标签：是否为目标语法点"],
            candidate_boundary=row["最终候选提取是否正确"],
            target_label=row["最终主标签：是否为目标语法点"],
            diagnosis_type=row["最终功能/问题类型"],
            boundary_decision_source=row["候选提取是否正确裁决来源"],
            target_decision_source=row["主标签：是否为目标语法点裁决来源"],
            diagnosis_decision_source=row["功能/问题类型裁决来源"],
        )
        graph.edge(cid, "HAS_CONSENSUS", consensus_id)

        for annotator in (1, 2, 3):
            decision_id = f"decision:main:{row['编号']}:A{annotator}"
            graph.node(
                decision_id,
                "AnnotationDecision",
                f"{row['编号']} annotator {annotator}",
                annotator=annotator,
                candidate_boundary=row[f"标注者{annotator}_候选提取是否正确"],
                target_label=row[f"标注者{annotator}_主标签：是否为目标语法点"],
                diagnosis_type=row[f"标注者{annotator}_功能/问题类型"],
                corrected_span=row[f"标注者{annotator}_若提取不正确请修正片段（条件必填）"],
            )
            graph.edge(cid, "ANNOTATED_AS", decision_id)
            graph.edge(decision_id, "CONTRIBUTES_TO", consensus_id)

        runtime = KG.evaluate(row)
        for rule_id in runtime["positive_rule_ids"] + runtime["weak_rule_ids"] + runtime["strong_rule_ids"]:
            rid = f"rule:{rule_id}"
            if rid not in graph.nodes:
                graph.node(rid, "RuntimeRule", rule_id, rule_id=rule_id)
            signal = (
                "positive" if rule_id in runtime["positive_rule_ids"]
                else "weak_reject" if rule_id in runtime["weak_rule_ids"]
                else "strong_reject"
            )
            graph.edge(cid, "ACTIVATES_RULE", rid, signal=signal)

    for row in evidence_rows:
        cid = candidate_id(row["编号"])
        if cid not in main_candidate_ids:
            raise ValueError(f"Evidence row lacks main candidate: {row['编号']}")
        consensus_id = f"consensus:evidence:{row['编号']}"
        graph.node(
            consensus_id,
            "EvidenceConsensus",
            row["最终结构槽位是否支持"],
            structure_support=row["最终结构槽位是否支持"],
            abnormal_slot=row["最终缺失/异常槽位"],
            confusion_target=row["最终易混对象"],
            evidence_span=row["最终证据片段（选填）"],
        )
        graph.edge(cid, "HAS_EVIDENCE_CONSENSUS", consensus_id)
        for annotator in (1, 2, 3):
            assessment_id = f"assessment:evidence:{row['编号']}:A{annotator}"
            graph.node(
                assessment_id,
                "EvidenceAssessment",
                f"{row['编号']} evidence annotator {annotator}",
                annotator=annotator,
                structure_support=row[f"标注者{annotator}_结构槽位是否支持"],
                abnormal_slot=row[f"标注者{annotator}_缺失/异常槽位"],
                confusion_target=row[f"标注者{annotator}_易混对象"],
                evidence_span=row[f"标注者{annotator}_证据片段（选填）"],
            )
            graph.edge(cid, "ASSESSED_BY", assessment_id)
            graph.edge(assessment_id, "CONTRIBUTES_TO", consensus_id)

    for row in slot_rows:
        cid = candidate_id(row["编号"])
        if cid not in main_candidate_ids:
            raise ValueError(f"Slot row lacks main candidate: {row['编号']}")
        slot_type_id = stable_id("slot_type", f"{row['槽位ID']}|{row['槽位名称']}")
        graph.node(
            slot_type_id,
            "SlotType",
            row["槽位名称"],
            slot_id=row["槽位ID"],
            annotation_requirement=row["标注要求"],
        )
        consensus_id = f"consensus:slot:{row['编号']}:{row['槽位ID']}"
        graph.node(
            consensus_id,
            "SlotConsensus",
            row["最终槽位状态"],
            slot_status=row["最终槽位状态"],
            slot_span=row["最终槽位片段（从句子中复制；条件必填）"],
            unavailable_reason=row["最终无法标出原因"],
        )
        graph.edge(cid, "HAS_SLOT_CONSENSUS", consensus_id)
        graph.edge(consensus_id, "INSTANCE_OF_SLOT", slot_type_id)
        for annotator in (1, 2, 3):
            assertion_id = f"assertion:slot:{row['编号']}:{row['槽位ID']}:A{annotator}"
            graph.node(
                assertion_id,
                "SlotAssertion",
                f"{row['编号']} {row['槽位ID']} annotator {annotator}",
                annotator=annotator,
                slot_status=row[f"标注者{annotator}_槽位状态"],
                slot_span=row[f"标注者{annotator}_槽位片段（从句子中复制；条件必填）"],
                unavailable_reason=row[f"标注者{annotator}_无法标出原因"],
            )
            graph.edge(cid, "HAS_SLOT_ASSERTION", assertion_id)
            graph.edge(assertion_id, "INSTANCE_OF_SLOT", slot_type_id)
            graph.edge(assertion_id, "CONTRIBUTES_TO", consensus_id)

    graph.write()

    node_ids = set(graph.nodes)
    dangling = [
        edge for edge in graph.edges.values()
        if edge["source"] not in node_ids or edge["target"] not in node_ids
    ]
    type_counts = Counter(row["type"] for row in graph.nodes.values())
    relation_counts = Counter(row["relation"] for row in graph.edges.values())
    candidate_edges: dict[str, Counter] = defaultdict(Counter)
    for edge in graph.edges.values():
        if edge["source"].startswith("candidate:"):
            candidate_edges[edge["source"]][edge["relation"]] += 1
    candidate_integrity = {
        "has_sentence": sum(candidate_edges[cid]["OCCURS_IN"] == 1 for cid in main_candidate_ids),
        "has_grammar_point": sum(candidate_edges[cid]["CANDIDATE_OF"] == 1 for cid in main_candidate_ids),
        "has_family": sum(candidate_edges[cid]["IN_FAMILY"] == 1 for cid in main_candidate_ids),
        "has_source": sum(candidate_edges[cid]["DERIVED_FROM"] == 1 for cid in main_candidate_ids),
        "has_consensus": sum(candidate_edges[cid]["HAS_CONSENSUS"] == 1 for cid in main_candidate_ids),
        "has_three_main_decisions": sum(candidate_edges[cid]["ANNOTATED_AS"] == 3 for cid in main_candidate_ids),
        "activates_runtime_rule": sum(candidate_edges[cid]["ACTIVATES_RULE"] >= 1 for cid in main_candidate_ids),
    }
    report = {
        "version": "v1.0",
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "node_type_counts": dict(sorted(type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "candidate_instances": len(main_candidate_ids),
        "sentence_groups": len({sentence_group_id(row["句子"]) for row in main_rows}),
        "evidence_candidates": len(evidence_rows),
        "slot_annotation_rows": len(slot_rows),
        "rule_reliability_rows": len(reliability_rows),
        "candidate_integrity_counts": candidate_integrity,
        "dangling_edges": len(dangling),
        "duplicate_node_ids": 0,
        "duplicate_edges": 0,
        "integrity_pass": not dangling and all(
            candidate_integrity[key] == len(main_candidate_ids)
            for key in (
                "has_sentence", "has_grammar_point", "has_family", "has_source",
                "has_consensus", "has_three_main_decisions",
            )
        ),
    }
    (OUT_DIR / "graph_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    schema = {
        "node_types": sorted(type_counts),
        "edge_types": sorted(relation_counts),
        "primary_path": "CandidateInstance -> GrammarPoint -> GrammarRule -> AuthoritySource",
        "audit_paths": [
            "CandidateInstance -> SentenceGroup",
            "CandidateInstance -> AnnotationDecision -> ConsensusDecision",
            "CandidateInstance -> EvidenceAssessment -> EvidenceConsensus",
            "CandidateInstance -> SlotAssertion -> SlotConsensus -> SlotType",
            "CandidateInstance -> GrammarRule -> AuthoritySource",
            "GrammarRule -> RuleReliability (training split only)",
        ],
    }
    (OUT_DIR / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
