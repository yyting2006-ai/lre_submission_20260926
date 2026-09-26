#!/usr/bin/env python3
"""Build the copyright-safe release layer from an authorized local workspace.

The exporter deliberately omits sentence text, candidate text, copied spans,
annotator-level decisions, free-text notes, and correction text. It retains
stable IDs, sentence-group hashes, consolidated labels, graph rules, splits,
predictions, and aggregate results needed to audit the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="rerun the copyright/path audit without rebuilding release artifacts",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_sentence(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "").strip()).lower()
    text = text.translate(str.maketrans({
        "，": ",", "。": ".", "！": "!", "？": "?", "；": ";",
        "：": ":", "（": "(", "）": ")", "“": '"', "”": '"',
        "‘": "'", "’": "'",
    }))
    return re.sub(r"\s+", "", text)


def sentence_group_id(text: str) -> str:
    value = normalize_sentence(text)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def split_sources(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[；;]", value or "") if part.strip()]


def scrub_paths(value: Any, workspace_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: scrub_paths(item, workspace_root) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_paths(item, workspace_root) for item in value]
    if isinstance(value, str):
        return value.replace(str(workspace_root), "${ASGD_ROOT}")
    return value


def copy_csv(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_json(source: Path, target: Path, workspace_root: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(scrub_paths(payload, workspace_root), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def export_labels(workspace_root: Path, output_root: Path) -> list[str]:
    gold = workspace_root / "annotation_qc/outputs/formal_qc_20260718"
    main = read_csv(gold / "gold_main_by_filename_majority_annotator2_20260718.csv")
    evidence = read_csv(gold / "gold_evidence_by_filename_majority_annotator2_20260718.csv")
    slots = read_csv(gold / "gold_slot_span_by_filename_majority_annotator2_20260718.csv")

    main_fields = [
        "instance_id", "sentence_group", "grammar_point", "grammar_family",
        "source_title", "page_anchor", "target_label", "extraction_label",
        "function_label", "target_consolidation", "extraction_consolidation",
        "function_consolidation", "gold_confidence_tier",
    ]
    main_rows = [{
        "instance_id": row["编号"],
        "sentence_group": sentence_group_id(row["句子"]),
        "grammar_point": row["候选语法点"],
        "grammar_family": row["语法家族"],
        "source_title": row["书名/文献名"],
        "page_anchor": row["页码"],
        "target_label": row["最终主标签：是否为目标语法点"],
        "extraction_label": row["最终候选提取是否正确"],
        "function_label": row["最终功能/问题类型"],
        "target_consolidation": row["主标签：是否为目标语法点裁决来源"],
        "extraction_consolidation": row["候选提取是否正确裁决来源"],
        "function_consolidation": row["功能/问题类型裁决来源"],
        "gold_confidence_tier": row["金标置信层级"],
    } for row in main]
    write_csv(output_root / "data/main_labels_release.csv", main_rows, main_fields)

    evidence_fields = [
        "instance_id", "sentence_group", "grammar_point", "grammar_family",
        "source_title", "page_anchor", "evidence_label",
        "evidence_consolidation", "gold_confidence_tier",
    ]
    evidence_rows = [{
        "instance_id": row["编号"],
        "sentence_group": sentence_group_id(row["句子"]),
        "grammar_point": row["候选语法点"],
        "grammar_family": row["语法家族"],
        "source_title": row["书名/文献名"],
        "page_anchor": row["页码"],
        "evidence_label": row["最终结构槽位是否支持"],
        "evidence_consolidation": row["结构槽位是否支持裁决来源"],
        "gold_confidence_tier": row["金标置信层级"],
    } for row in evidence]
    write_csv(output_root / "data/evidence_labels_release.csv", evidence_rows, evidence_fields)

    slot_fields = [
        "instance_id", "sentence_group", "grammar_point", "grammar_family",
        "slot_id", "slot_name", "slot_status", "span_length",
        "slot_consolidation", "gold_confidence_tier",
    ]
    slot_rows = [{
        "instance_id": row["编号"],
        "sentence_group": sentence_group_id(row["句子"]),
        "grammar_point": row["候选语法点"],
        "grammar_family": row["语法家族"],
        "slot_id": row["槽位ID"],
        "slot_name": row["槽位名称"],
        "slot_status": row["最终槽位状态"],
        "span_length": len(row["最终槽位片段（从句子中复制；条件必填）"] or ""),
        "slot_consolidation": row["槽位状态裁决来源"],
        "gold_confidence_tier": row["金标置信层级"],
    } for row in slots]
    write_csv(output_root / "data/slot_labels_release.csv", slot_rows, slot_fields)

    return [row["句子"] for row in main if len(row["句子"].strip()) >= 6]


def export_graph(workspace_root: Path, output_root: Path) -> None:
    kg = workspace_root / "grammar_kg/kg"
    rules = read_csv(kg / "high_frequency_rule_evidence_v0_3.csv")
    points = read_csv(kg / "grammar_points_v0_1.csv")
    bindings = json.loads((kg / "executable_rule_bindings_v1.json").read_text(encoding="utf-8"))
    bound_points = set(bindings.get("rules", {}))

    rule_fields = [
        "rule_id", "rule_group", "grammar_point", "canonical_name", "grammar_family",
        "polarity", "rule_text", "source_ids", "rule_page_anchor", "rule_keywords",
        "evidence_status", "runtime_bound",
    ]
    rule_rows = [{
        "rule_id": row["规则ID"],
        "rule_group": row["规则组"],
        "grammar_point": row["候选语法点"],
        "canonical_name": row["规范名称"],
        "grammar_family": row["语法家族"],
        "polarity": row["规则层级"],
        "rule_text": row["规则文本"],
        "source_ids": row["支撑文献ID"],
        "rule_page_anchor": row["规则说明页码"],
        "rule_keywords": row["规则关键词"],
        "evidence_status": row["证据状态"],
        "runtime_bound": int(row["候选语法点"] in bound_points),
    } for row in rules]
    write_csv(output_root / "graph/rule_catalog_release.csv", rule_rows, rule_fields)

    # Keep an original-header compatibility copy for the experiment scripts,
    # but remove the column that embeds verbatim example spans.
    compatible_rules = []
    compatible_rule_fields = list(rules[0]) if rules else []
    for row in rules:
        safe_row = dict(row)
        safe_row["样例锚点"] = "WITHHELD_COPYRIGHTED_EXAMPLES"
        compatible_rules.append(safe_row)
    write_csv(
        output_root / "code/grammar_kg/kg/high_frequency_rule_evidence_v0_3.csv",
        compatible_rules,
        compatible_rule_fields,
    )

    point_fields = [
        "grammar_point_id", "candidate_label", "canonical_label", "grammar_family",
        "sample_count", "markers", "source_status", "runtime_bound",
    ]
    point_rows = [{
        "grammar_point_id": row["grammar_point_id"],
        "candidate_label": row["candidate_label"],
        "canonical_label": row["canonical_label"],
        "grammar_family": row["family"],
        "sample_count": row["sample_count"],
        "markers": row["markers"],
        "source_status": row["source_status"],
        "runtime_bound": int(row["candidate_label"] in bound_points),
    } for row in points]
    write_csv(output_root / "graph/grammar_points_release.csv", point_rows, point_fields)
    write_csv(
        output_root / "code/grammar_kg/kg/grammar_points_v0_1.csv",
        points,
        list(points[0]) if points else [],
    )

    source_map: dict[str, str] = {}
    for row in rules:
        for entry in split_sources(row.get("支撑文献说明", "")):
            source_id, separator, title = entry.partition("=")
            if separator:
                source_map.setdefault(source_id.strip(), title.strip())
        for source_id in split_sources(row.get("支撑文献ID", "")):
            source_map.setdefault(source_id, "")
    source_rows = [{"source_id": key, "source_title": source_map[key]} for key in sorted(source_map)]
    write_csv(output_root / "graph/authority_sources_release.csv", source_rows, ["source_id", "source_title"])

    copy_json(kg / "executable_rule_bindings_v1.json", output_root / "graph/executable_rule_bindings_v1.json", workspace_root)
    copy_json(
        kg / "executable_rule_bindings_v1.json",
        output_root / "code/grammar_kg/kg/executable_rule_bindings_v1.json",
        workspace_root,
    )
    graph = workspace_root / "grammar_kg/kg_v1_candidate_evidence"
    copy_json(graph / "graph_summary.json", output_root / "graph/graph_summary.json", workspace_root)
    copy_json(graph / "schema.json", output_root / "graph/schema.json", workspace_root)
    page_audit = kg / "page_anchor_audit_v1"
    copy_csv(
        page_audit / "rule_page_anchor_audit.csv",
        output_root / "graph/rule_page_anchor_audit.csv",
    )
    copy_json(
        page_audit / "rule_page_anchor_audit.json",
        output_root / "graph/rule_page_anchor_audit.json",
        workspace_root,
    )


def export_results(workspace_root: Path, output_root: Path) -> None:
    outputs = workspace_root / "asgd_fast_experiments/outputs"
    agreement = workspace_root / "annotation_qc/outputs/formal_qc_20260718/agreement_audit"

    csv_files = {
        outputs / "exp_20260718_grouped_kg_v2/results_by_seed.csv": "main_results_by_seed.csv",
        outputs / "exp_20260718_grouped_kg_v2/results_aggregate.csv": "main_results_aggregate.csv",
        outputs / "exp_20260718_grouped_kg_v2/family_results_by_seed.csv": "family_results_by_seed.csv",
        outputs / "exp_20260718_grouped_kg_v2/selected_hyperparameters.csv": "selected_hyperparameters.csv",
        outputs / "exp_20260718_grouped_kg_v2/review_exclusion_by_seed.csv": "review_exclusion_by_seed.csv",
        outputs / "exp_20260718_grouped_kg_v2/statistical_analysis/group_bootstrap_confidence_intervals.csv": "bootstrap_confidence_intervals.csv",
        outputs / "exp_20260718_grouped_kg_v2/statistical_analysis/paired_group_bootstrap_differences.csv": "paired_bootstrap_differences.csv",
        outputs / "exp_20260718_grouped_kg_v2/statistical_analysis/mcnemar_descriptive.csv": "mcnemar_descriptive.csv",
        outputs / "exp_20260718_grouped_auxiliary_v1/results_by_seed.csv": "aux_results_by_seed.csv",
        outputs / "exp_20260718_grouped_auxiliary_v1/results_aggregate.csv": "aux_results_aggregate.csv",
        outputs / "exp_20260718_macbert_grouped_v1/results_by_seed.csv": "macbert_results_by_seed.csv",
        outputs / "exp_20260718_selective_review_v1/selective_review_by_seed.csv": "selective_review_by_seed.csv",
        outputs / "exp_20260718_selective_review_v1/selective_review_aggregate.csv": "selective_review_aggregate.csv",
        outputs / "exp_20260718_trace_state_audit_v1/trace_state_by_seed.csv": "trace_state_by_seed.csv",
        outputs / "exp_20260718_trace_state_audit_v1/trace_state_aggregate.csv": "trace_state_aggregate.csv",
        agreement / "agreement_summary.csv": "annotation_agreement_summary.csv",
        agreement / "pairwise_kappa.csv": "annotation_pairwise_kappa.csv",
        agreement / "slot_span_pairwise_agreement.csv": "slot_span_pairwise_agreement.csv",
        agreement / "adjudication_source_summary.csv": "adjudication_source_summary.csv",
        outputs / "exp_20260718_openai_gpt56terra_seed42_v2/release_safe/main_predictions_release.csv": "llm_global/main_predictions_release.csv",
        outputs / "exp_20260718_openai_gpt56terra_seed42_v2/release_safe/evidence_predictions_release.csv": "llm_global/evidence_predictions_release.csv",
        outputs / "exp_20260718_openai_gpt56terra_seed42_v2/release_safe/slot_predictions_release.csv": "llm_global/slot_predictions_release.csv",
        outputs / "exp_20260718_openai_gpt56terra_seed42_v2/release_safe/direct_audit_release.csv": "llm_global/direct_audit_release.csv",
        outputs / "exp_20260718_openai_gpt56terra_seed42_v2/release_safe/rule_audit_release.csv": "llm_global/rule_audit_release.csv",
        outputs / "exp_20260718_openai_gpt56terra_strong_seed42_v1/release_safe/dev_point_retrieved_predictions_release.csv": "llm_strong/dev_point_retrieved_predictions_release.csv",
        outputs / "exp_20260718_openai_gpt56terra_strong_seed42_v1/release_safe/dev_point_retrieved_textkg_predictions_release.csv": "llm_strong/dev_point_retrieved_textkg_predictions_release.csv",
        outputs / "exp_20260718_openai_gpt56terra_strong_seed42_v1/release_safe/selected_test_predictions_release.csv": "llm_strong/selected_test_predictions_release.csv",
        outputs / "exp_20260718_llm_kg_governor_v1/dev_predictions_release.csv": "llm_governor/dev_predictions_release.csv",
        outputs / "exp_20260718_llm_kg_governor_v1/test_predictions_release.csv": "llm_governor/test_predictions_release.csv",
        outputs / "exp_20260718_llm_comparison_v1/main_metrics_seed42.csv": "llm_comparison/main_metrics_seed42.csv",
        outputs / "exp_20260718_llm_comparison_v1/group_bootstrap_confidence_intervals.csv": "llm_comparison/group_bootstrap_confidence_intervals.csv",
        outputs / "exp_20260718_llm_comparison_v1/paired_group_bootstrap_differences.csv": "llm_comparison/paired_group_bootstrap_differences.csv",
        outputs / "exp_20260718_llm_comparison_v1/mcnemar_descriptive.csv": "llm_comparison/mcnemar_descriptive.csv",
        outputs / "exp_20260718_llm_comparison_v1/family_metrics.csv": "llm_comparison/family_metrics.csv",
        outputs / "exp_20260718_llm_comparison_v1/gold_confidence_tier_metrics.csv": "llm_comparison/gold_confidence_tier_metrics.csv",
        outputs / "exp_20260718_llm_comparison_v1/binding_coverage_metrics.csv": "llm_comparison/binding_coverage_metrics.csv",
        outputs / "exp_20260718_llm_comparison_v1/trace_state_metrics.csv": "llm_comparison/trace_state_metrics.csv",
        outputs / "exp_20260718_llm_comparison_v1/auxiliary_comparison_seed42.csv": "llm_comparison/auxiliary_comparison_seed42.csv",
        outputs / "exp_20260718_llm_comparison_v1/paired_audit_bootstrap.csv": "llm_comparison/paired_audit_bootstrap.csv",
        outputs / "exp_20260718_llm_comparison_v1/paired_audit_mcnemar.csv": "llm_comparison/paired_audit_mcnemar.csv",
    }
    for source, name in csv_files.items():
        copy_csv(source, output_root / "results" / name)

    json_files = {
        outputs / "exp_20260718_grouped_kg_v2/experiment_config.json": "main_experiment_config.json",
        outputs / "exp_20260718_grouped_kg_v2/split_manifest.json": "main_split_manifest.json",
        outputs / "exp_20260718_grouped_kg_v2/statistical_analysis/statistical_summary.json": "statistical_summary.json",
        outputs / "exp_20260718_grouped_auxiliary_v1/split_manifest.json": "aux_split_manifest.json",
        outputs / "exp_20260718_macbert_grouped_v1/experiment_config_and_summary.json": "macbert_config_and_summary.json",
        outputs / "exp_20260718_efficiency_v1/efficiency_summary.json": "efficiency_summary.json",
        outputs / "exp_20260718_trace_state_audit_v1/trace_state_summary.json": "trace_state_summary.json",
        agreement / "agreement_audit.json": "annotation_agreement_audit.json",
        outputs / "exp_20260718_openai_gpt56terra_seed42_v2/release_safe/protocol_release.json": "llm_global/protocol_release.json",
        outputs / "exp_20260718_openai_gpt56terra_seed42_v2/release_safe/summary_release.json": "llm_global/summary_release.json",
        outputs / "exp_20260718_openai_gpt56terra_strong_seed42_v1/release_safe/summary_release.json": "llm_strong/summary_release.json",
        outputs / "exp_20260718_llm_kg_governor_v1/summary.json": "llm_governor/summary.json",
        outputs / "exp_20260718_llm_comparison_v1/summary.json": "llm_comparison/summary.json",
    }
    for source, name in json_files.items():
        copy_json(source, output_root / "results" / name, workspace_root)

    main_predictions = read_csv(outputs / "exp_20260718_grouped_kg_v2/predictions_with_rule_traces.csv")
    if main_predictions:
        write_csv(
            output_root / "results/main_predictions_release.csv",
            main_predictions,
            list(main_predictions[0]),
        )

    aux_predictions = read_csv(outputs / "exp_20260718_grouped_auxiliary_v1/predictions.csv")
    aux_fields = [
        "seed", "task", "model", "编号", "gold", "prediction", "correct",
        "kg_rule_trace", "槽位ID", "gold_status", "grounded",
    ]
    write_csv(output_root / "results/aux_predictions_release.csv", aux_predictions, aux_fields)

    mac_predictions = read_csv(outputs / "exp_20260718_macbert_grouped_v1/predictions.csv")
    if mac_predictions:
        write_csv(
            output_root / "results/macbert_predictions_release.csv",
            mac_predictions,
            list(mac_predictions[0]),
        )


def audit_release(output_root: Path, source_sentences: list[str]) -> dict[str, Any]:
    forbidden_tokens = [
        "/Volumes/", "/Users/", "标注者1_", "标注者2_", "标注者3_",
        "最终备注/疑问", "最终不确定原因", "最终若提取不正确请修正片段",
    ]
    findings: list[dict[str, str]] = []
    audited_files = 0
    cross_provider_files_audited = 0
    v3_files_audited = 0
    # A handbook extraction row can exactly equal an official GF0025 inventory
    # label (for example, a list of classifiers). Such authority metadata is
    # intentionally released and is not a source sentence leak.
    authority_metadata: set[str] = set()
    compiler_catalog = output_root / "code/grammar_kg/kg/standard_grammar_points_gf0025_v2.csv"
    if compiler_catalog.exists():
        for row in read_csv(compiler_catalog):
            for field in ("official_label", "grammar_point", "canonical_label", "standard_text"):
                value = row.get(field, "").strip()
                if value:
                    authority_metadata.add(value)
    authority_metadata_exemptions = 0
    for cache_dir in output_root.rglob("__pycache__"):
        findings.append({"file": str(cache_dir.relative_to(output_root)), "finding": "python cache directory"})
    for pyc in output_root.rglob("*.pyc"):
        findings.append({"file": str(pyc.relative_to(output_root)), "finding": "compiled Python cache"})
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or (
            path.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".sh", ".py", ".tex", ".bib"}
            and path.name != ".python-version"
        ):
            continue
        if path.name in {"release_audit.json", "verify_release.py"} or path.resolve() == Path(__file__).resolve():
            continue
        audited_files += 1
        relative_path = path.relative_to(output_root)
        if relative_path.parts[:2] == ("results", "llm_cross_provider"):
            cross_provider_files_audited += 1
        if "v3" in path.name or path.name in {
            "executable_bindings_full_inventory_v3.json",
            "diagnostic_templates_v3.csv",
            "diagnostic_unit_tests_v3.csv",
        }:
            v3_files_audited += 1
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for token in forbidden_tokens:
            if token in text:
                findings.append({"file": str(relative_path), "finding": f"forbidden token: {token}"})
        for sentence in source_sentences:
            if sentence in text:
                normalized_sentence = sentence.strip()
                if normalized_sentence in authority_metadata or (
                    "、" in normalized_sentence
                    and any(normalized_sentence in value for value in authority_metadata)
                ):
                    authority_metadata_exemptions += 1
                    continue
                findings.append({"file": str(relative_path), "finding": "verbatim source sentence"})
                break
        if re.search(r"(?:[A-Za-z]:\\|/(?:Users|Volumes)/)", text):
            findings.append({"file": str(relative_path), "finding": "local filesystem path"})

    alignment_path = output_root / "graph/standard_handbook_alignment_v3.csv"
    alignment_rows = read_csv(alignment_path) if alignment_path.exists() else []
    if alignment_rows:
        restricted_alignment_fields = {"page_text", "snippet", "raw_text", "handbook_text"}
        leaked_fields = restricted_alignment_fields & set(alignment_rows[0])
        if leaked_fields:
            findings.append({
                "file": "graph/standard_handbook_alignment_v3.csv",
                "finding": f"restricted handbook fields: {sorted(leaked_fields)}",
            })
    for forbidden_graph in (
        output_root / "graph/nodes_v3_full_compilation.csv",
        output_root / "graph/edges_v3_full_compilation.csv",
    ):
        if forbidden_graph.exists():
            findings.append({
                "file": str(forbidden_graph.relative_to(output_root)),
                "finding": "text-bearing full v3 graph must remain restricted",
            })
    report = {
        "audit_version": "1.2",
        "audited_files": audited_files,
        "cross_provider_files_audited": cross_provider_files_audited,
        "v3_files_audited": v3_files_audited,
        "handbook_alignment_rows_checked": len(alignment_rows),
        "verbatim_sentences_checked": len(source_sentences),
        "authority_metadata_exemptions": authority_metadata_exemptions,
        "findings": findings,
        "pass": not findings,
    }
    (output_root / "release_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.audit_only:
        gold_path = (
            workspace_root
            / "annotation_qc/outputs/formal_qc_20260718/"
            "gold_main_by_filename_majority_annotator2_20260718.csv"
        )
        main_rows = read_csv(gold_path)
        source_sentences = [
            row["句子"] for row in main_rows if len(row["句子"].strip()) >= 6
        ]
    else:
        source_sentences = export_labels(workspace_root, output_root)
        export_graph(workspace_root, output_root)
        export_results(workspace_root, output_root)
    report = audit_release(output_root, source_sentences)
    if not report["pass"]:
        raise SystemExit(f"Release audit failed: {report['findings'][:3]}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
