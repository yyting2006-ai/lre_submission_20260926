#!/usr/bin/env python3
"""Verify release integrity and the headline values reported in the paper."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: float, expected: float, tolerance: float = 5e-7) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0):
        raise AssertionError(f"expected {expected}, got {actual}")


def sha256_json(value: object) -> str:
    blob = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lookup(rows: list[dict[str, str]], task: str, model: str, metric: str) -> float:
    matches = [
        row for row in rows
        if row["task"] == task and row["model"] == model and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise AssertionError(f"lookup failed: {task}/{model}/{metric}: {len(matches)} rows")
    return float(matches[0]["mean"])


def lookup_regime(
    rows: list[dict[str, str]], regime: str, model: str, metric: str
) -> float:
    matches = [
        row for row in rows
        if row["regime"] == regime and row["model"] == model
    ]
    if len(matches) != 1:
        raise AssertionError(f"regime lookup failed: {regime}/{model}: {len(matches)} rows")
    return float(matches[0][metric])


def binary_scores(
    rows: list[dict[str, str]], prediction_key: str = "prediction"
) -> dict[str, float]:
    labels = ["不是目标语法点", "是目标语法点"]
    correct = sum(row["gold"] == row[prediction_key] for row in rows)
    per_label_f1: list[float] = []
    for label in labels:
        true_positive = sum(
            row["gold"] == label and row[prediction_key] == label for row in rows
        )
        false_positive = sum(
            row["gold"] != label and row[prediction_key] == label for row in rows
        )
        false_negative = sum(
            row["gold"] == label and row[prediction_key] != label for row in rows
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_label_f1.append(2 * true_positive / denominator if denominator else 0.0)
    negatives = [row for row in rows if row["gold"] == labels[0]]
    return {
        "accuracy": correct / len(rows),
        "macro_f1": sum(per_label_f1) / len(per_label_f1),
        "negative_accuracy": sum(
            row[prediction_key] == labels[0] for row in negatives
        )
        / len(negatives),
    }


def exact_repeat_stability(rows: list[dict[str, str]], key: str) -> float:
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(row["编号"], set()).add(row[key])
    return sum(len(values) == 1 for values in grouped.values()) / len(grouped)


def verify_cross_provider_release(
    directory: str,
    *,
    provider: str,
    requested_model: str,
    response_model: str,
    selected_variant: str,
    dev_point_macro_f1: float,
    dev_textkg_macro_f1: float,
    test_accuracy: float,
    test_macro_f1: float,
    test_negative_accuracy: float,
) -> dict:
    """Recompute a release-safe external model's selection and test metrics."""
    base = ROOT / directory
    protocol = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
    release = base / "release_safe"
    summary = json.loads((release / "summary_release.json").read_text(encoding="utf-8"))

    assert summary["protocol"] == protocol
    protocol_core = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    assert protocol["protocol_sha256"] == sha256_json(protocol_core)
    assert protocol["provider"] == provider
    assert protocol["model"] == requested_model
    assert protocol["store_control"] == "not_sent"
    assert protocol["reasoning_effort"] == "not_applicable"
    assert protocol["pricing_status"] == "unknown"
    assert protocol["pricing_usd_per_million_tokens"] is None
    assert protocol["seed"] == 42
    assert protocol["selection_split"] == "development"
    assert protocol["selection_metric"] == "macro_f1"
    assert protocol["variants"] == ["point_retrieved", "point_retrieved_textkg"]
    assert protocol["train_n"] == 2097
    assert protocol["dev_n"] == 299
    assert protocol["test_n"] == 600

    filenames = {
        "point_retrieved": "dev_point_retrieved_predictions_release.csv",
        "point_retrieved_textkg": "dev_point_retrieved_textkg_predictions_release.csv",
    }
    dev_rows: dict[str, list[dict[str, str]]] = {}
    for variant, filename in filenames.items():
        relative = str((release / filename).relative_to(ROOT))
        rows = read_csv(relative)
        dev_rows[variant] = rows
        assert len(rows) == 299
        assert len({row["job_id"] for row in rows}) == 299
        assert {row["task"] for row in rows} == {f"dev_{variant}"}

    assert {row["编号"] for row in dev_rows["point_retrieved"]} == {
        row["编号"] for row in dev_rows["point_retrieved_textkg"]
    }
    for rows in dev_rows.values():
        assert sha256_json([row["编号"] for row in rows]) == protocol["dev_ids_sha256"]

    test_relative = str(
        (release / "selected_test_predictions_release.csv").relative_to(ROOT)
    )
    test_rows = read_csv(test_relative)
    assert len(test_rows) == 600
    assert len({row["job_id"] for row in test_rows}) == 600
    assert {row["task"] for row in test_rows} == {f"test_{selected_variant}"}
    assert sha256_json([row["编号"] for row in test_rows]) == protocol["test_ids_sha256"]

    for rows in (*dev_rows.values(), test_rows):
        assert {row["response_model"] for row in rows} == {response_model}
        assert {row["response_status"] for row in rows} == {"completed"}
        assert {row["valid_output"] for row in rows} == {"1"}

    expected_dev_macro_f1 = {
        "point_retrieved": dev_point_macro_f1,
        "point_retrieved_textkg": dev_textkg_macro_f1,
    }
    for variant, rows in dev_rows.items():
        scores = binary_scores(rows)
        reported = summary["development"][variant]
        assert reported["n"] == 299
        assert reported["estimated_standard_api_cost_usd"] is None
        for metric in ("accuracy", "macro_f1", "negative_accuracy"):
            assert_close(scores[metric], reported[metric])
        assert_close(reported["macro_f1"], expected_dev_macro_f1[variant])

    computed_selection = max(
        filenames,
        key=lambda variant: (
            summary["development"][variant]["macro_f1"],
            summary["development"][variant]["accuracy"],
            variant,
        ),
    )
    assert summary["selected_variant"] == selected_variant == computed_selection

    test_scores = binary_scores(test_rows)
    assert summary["test"]["n"] == 600
    assert summary["test"]["estimated_standard_api_cost_usd"] is None
    for metric in ("accuracy", "macro_f1", "negative_accuracy"):
        assert_close(test_scores[metric], summary["test"][metric])
    assert_close(summary["test"]["accuracy"], test_accuracy)
    assert_close(summary["test"]["macro_f1"], test_macro_f1)
    assert_close(summary["test"]["negative_accuracy"], test_negative_accuracy)
    return summary


def main() -> None:
    expected_python = (3, 12, 13)
    if sys.version_info[:3] != expected_python:
        raise AssertionError(
            "ExecKG v3 requires exactly CPython 3.12.13 because the compiler "
            f"uses private re parser APIs; got {sys.version_info[:3]}"
        )
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12.13"

    main_labels = read_csv("data/main_labels_release.csv")
    evidence_labels = read_csv("data/evidence_labels_release.csv")
    slot_labels = read_csv("data/slot_labels_release.csv")
    rules = read_csv("graph/rule_catalog_release.csv")
    points = read_csv("graph/grammar_points_release.csv")
    standard_inventory = read_csv("graph/standard_inventory_release_v2.csv")
    graph = json.loads((ROOT / "graph/graph_summary.json").read_text(encoding="utf-8"))
    graph_v2 = json.loads(
        (ROOT / "graph/graph_summary_v2_full_inventory.json").read_text(encoding="utf-8")
    )
    full_coverage = json.loads(
        (ROOT / "graph/full_inventory_coverage_v2.json").read_text(encoding="utf-8")
    )
    page_audit = json.loads(
        (ROOT / "graph/rule_page_anchor_audit.json").read_text(encoding="utf-8")
    )
    audit = json.loads((ROOT / "release_audit.json").read_text(encoding="utf-8"))

    # Complete GF0025--2021 compiler layer. These are machine/static contracts,
    # not natural-language behavioral or semantic accuracy tests.
    crosswalk = read_csv("graph/standard_diagnostic_crosswalk_v3.csv")
    templates = read_csv("graph/diagnostic_templates_v3.csv")
    contract_tests = read_csv("graph/diagnostic_unit_tests_v3.csv")
    corrections = read_csv("graph/standard_catalog_corrections_v3.csv")
    handbook_alignment = read_csv("graph/standard_handbook_alignment_v3.csv")
    positive_evidence = read_csv("graph/standard_positive_evidence_summary_v3.csv")
    bindings_v3 = json.loads(
        (ROOT / "graph/executable_bindings_full_inventory_v3.json").read_text(
            encoding="utf-8"
        )
    )
    compiler_coverage = json.loads(
        (ROOT / "graph/full_inventory_executable_coverage_v3.json").read_text(
            encoding="utf-8"
        )
    )
    handbook_coverage = json.loads(
        (ROOT / "graph/full_inventory_handbook_coverage_v3.json").read_text(
            encoding="utf-8"
        )
    )
    example_coverage = json.loads(
        (ROOT / "graph/full_inventory_example_coverage_v3.json").read_text(
            encoding="utf-8"
        )
    )
    graph_v3 = json.loads(
        (ROOT / "graph/graph_summary_v3_full_compilation.json").read_text(
            encoding="utf-8"
        )
    )
    schema_v3 = json.loads(
        (ROOT / "graph/schema_v3_full_compilation.json").read_text(encoding="utf-8")
    )

    compiler_catalog = read_csv(
        "code/grammar_kg/kg/standard_grammar_points_gf0025_v2.csv"
    )
    assert len(compiler_catalog) == 572
    assert len({row["standard_point_id"] for row in compiler_catalog}) == 572
    assert Counter(row["page_anchor_status"] for row in compiler_catalog) == Counter(
        {"ocr_detected": 362, "appendix_span_verified": 210}
    )

    assert len(crosswalk) == 646
    assert len({row["diagnostic_unit_id"] for row in crosswalk}) == 646
    assert len({row["diagnostic_profile_id"] for row in crosswalk}) == 646
    assert len({row["runtime_binding_id"] for row in crosswalk}) == 646
    assert len({row["standard_point_id"] for row in crosswalk}) == 572
    assert len({row["template_id"] for row in crosswalk}) == 233
    assert len(templates) == 233
    assert len({row["template_id"] for row in templates}) == 233
    assert Counter(row["candidate_pattern_source"] for row in crosswalk) == Counter(
        {
            "normalized_legacy_regex": 464,
            "normalized_legacy_regex_with_unit_specialization": 39,
            "source_literal_retrieval_trigger": 143,
        }
    )
    assert sum(bool(row["unit_specialization_pattern"]) for row in crosswalk) == 39
    assert Counter(row["mapping_disposition"] for row in crosswalk) == Counter(
        {
            "taxonomy_instantiated": 452,
            "seed_reused": 93,
            "point_specific_instantiated": 101,
        }
    )
    assert Counter(row["regime"] for row in crosswalk) == Counter(
        {"structural_pattern": 511, "complex_functional": 135}
    )
    assert {row["review_status"] for row in crosswalk} == {"not_claimed"}
    assert {row["static_audit_status"] for row in crosswalk} == {
        "static_audit_passed"
    }
    active_crosswalk_fields = [
        field for field in crosswalk[0] if field != "catalog_correction_provenance"
    ]
    assert all(
        "#NAME?" not in "\x1f".join(row[field] for field in active_crosswalk_fields)
        for row in crosswalk
    )
    provenance_with_excel_original = []
    for row in crosswalk:
        provenance = json.loads(row["catalog_correction_provenance"] or "[]")
        if any(item.get("original_value") == "#NAME?" for item in provenance):
            provenance_with_excel_original.append((row["standard_point_id"], provenance))
            assert row["standard_point_id"] == "GF0025-2021-3-002"
            assert {
                item["field_name"]
                for item in provenance
                if item.get("original_value") == "#NAME?"
            } == {"grammar_point", "canonical_label", "standard_text"}
            assert all("#NAME?" not in item["corrected_value"] for item in provenance)
    assert len(provenance_with_excel_original) == 1
    for row in crosswalk:
        re.compile(row["candidate_pattern"])
        re.compile(row["trigger_pattern"])

    assert len(bindings_v3["bindings"]) == 646
    assert len({row["binding_id"] for row in bindings_v3["bindings"]}) == 646
    assert bindings_v3["review_claim"] == "not_claimed"
    assert bindings_v3["status"] == "machine_compiled_source_anchored_static_audited"

    case_types = {
        "positive",
        "boundary",
        "missing_slot",
        "confusable",
        "semantic_incompatible",
        "other_construction",
        "uncertain",
    }
    assert len(contract_tests) == 4522
    assert Counter(row["case_type"] for row in contract_tests) == Counter(
        {case_type: 646 for case_type in case_types}
    )
    assert Counter(row["diagnostic_unit_id"] for row in contract_tests) == Counter(
        {row["diagnostic_unit_id"]: 7 for row in crosswalk}
    )
    assert {row["execution_status"] for row in contract_tests} == {
        "binding_contract_passed"
    }
    assert {row["generation_status"] for row in contract_tests} == {
        "deterministic_machine_generated_contract_test"
    }
    assert {row["candidate_detector_invoked"] for row in contract_tests} == {"true"}
    positive_contracts = [
        row for row in contract_tests if row["case_type"] == "positive"
    ]
    assert len(positive_contracts) == 646
    assert {row["candidate_detector_matched"] for row in positive_contracts} == {
        "true"
    }

    compiler_counts = compiler_coverage["counts"]
    assert compiler_counts["official_standard_points"] == 572
    assert compiler_counts["diagnostic_units"] == 646
    assert compiler_counts["diagnostic_profiles"] == 646
    assert compiler_counts["runtime_bindings"] == 646
    assert compiler_counts["operational_templates"] == 233
    assert compiler_counts["generated_contract_tests"] == 4522
    assert compiler_counts["generated_contract_tests_passed"] == 4522
    assert compiler_counts["runtime_bindings_using_normalized_legacy_candidate_regex"] == 503
    assert compiler_counts["split_runtime_bindings_with_unit_specialized_candidate_detector"] == 39
    assert compiler_counts["runtime_bindings_using_source_literal_candidate_pattern"] == 143
    assert compiler_counts["official_taxonomy_paths"] == 103
    assert compiler_counts["taxonomy_paths_covered"] == 103
    assert compiler_counts["catalog_corrections_applied"] == 4
    assert compiler_counts["catalog_points_corrected"] == 2
    assert all(compiler_coverage["integrity_audit"].values())
    assert "not behavioral or semantic validation" in compiler_coverage[
        "claim_boundary"
    ]["generated_tests"]

    assert len(corrections) == 4
    assert len({row["standard_point_id"] for row in corrections}) == 2
    assert {row["verification_status"] for row in corrections} == {
        "manually_verified_against_official_page"
    }

    assert len(handbook_alignment) == 572
    assert len({row["standard_point_id"] for row in handbook_alignment}) == 572
    assert Counter(row["confidence"] for row in handbook_alignment) == Counter(
        {"high": 569, "medium": 3}
    )
    assert {row["alignment_status"] for row in handbook_alignment} == {"aligned"}
    assert all(row["primary_page"] and row["page_text_sha256"] for row in handbook_alignment)
    assert "page_text" not in handbook_alignment[0]
    assert "snippet" not in handbook_alignment[0]
    assert handbook_coverage["coverage"]["expected_standard_points"] == 572
    assert handbook_coverage["coverage"]["aligned_with_nonempty_primary_page"] == 572
    assert handbook_coverage["coverage"]["confidence_counts"] == {
        "high": 569,
        "medium": 3,
    }
    assert handbook_coverage["low_confidence_count"] == 0
    assert handbook_coverage["medium_confidence_count"] == 3
    assert all(handbook_coverage["integrity"].values())
    assert handbook_coverage["input"]["handbook_page_corpus"].startswith(
        "restricted/not_released/"
    )

    assert len(positive_evidence) == 572
    assert sum(int(row["positive_example_pairs"]) for row in positive_evidence) == 3645
    assert Counter(row["evidence_status"] for row in positive_evidence) == Counter(
        {"annotated_positive_present": 219, "no_annotated_positive": 353}
    )
    assert example_coverage["catalog_points"] == 572
    assert example_coverage["unique_sentence_point_pairs"] == 3645
    assert example_coverage["points_with_annotated_positive"] == 219
    assert example_coverage["points_without_annotated_positive"] == 353
    assert example_coverage["privacy"]["sentence_text_released"] is False
    assert example_coverage["integrity_pass"] is True

    assert graph_v3["scope"] == {
        "standard_grammar_points": 572,
        "diagnostic_units": 646,
        "diagnostic_profiles": 646,
        "diagnostic_templates": 233,
        "runtime_bindings": 646,
        "contract_tests": 4522,
        "handbook_alignments": 572,
        "catalog_corrections": 4,
    }
    assert graph_v3["base_graph"]["nodes"] == 28158
    assert graph_v3["base_graph"]["edges"] == 64046
    assert graph_v3["total"]["nodes"] == 35379
    assert graph_v3["total"]["edges"] == 72898
    assert graph_v3["validation"]["v2_node_ids_preserved"] == 28158
    assert graph_v3["validation"]["v2_edges_preserved"] == 64046
    assert graph_v3["validation"]["duplicate_node_ids"] == 0
    assert graph_v3["validation"]["duplicate_edges"] == 0
    assert graph_v3["validation"]["dangling_edges"] == 0
    assert graph_v3["validation"]["integrity_pass"] is True
    assert graph_v3["validation"]["handbook_alignment_confidence"] == {
        "high": 569,
        "medium": 3,
    }
    assert schema_v3["version"] == "v3.0-full-compilation"
    assert "DiagnosticContractTest" in schema_v3["node_types"]
    assert "HAS_CONTRACT_TEST" in schema_v3["edge_types"]
    assert graph_v3["base_graph"]["directory"].startswith("restricted/not_released/")
    for name in (
        "standard_diagnostic_crosswalk_v3.csv",
        "diagnostic_templates_v3.csv",
        "diagnostic_unit_tests_v3.csv",
        "standard_handbook_alignment_v3.csv",
        "standard_catalog_corrections_v3.csv",
    ):
        assert sha256_file(ROOT / "graph" / name) == graph_v3["input_sha256"][name]

    assert not (ROOT / "graph/nodes_v3_full_compilation.csv").exists()
    assert not (ROOT / "graph/edges_v3_full_compilation.csv").exists()
    assert not list(ROOT.rglob("__pycache__"))
    assert not list(ROOT.rglob("*.pyc"))

    assert len(main_labels) == 3000
    assert len(evidence_labels) == 1000
    assert len(slot_labels) == 1320
    assert len({row["sentence_group"] for row in main_labels}) == 2557
    assert len(rules) == 144
    assert len(points) == 37
    assert sum(int(row["runtime_bound"]) for row in points) == 16
    assert len(standard_inventory) == 572
    assert len({row["standard_point_id"] for row in standard_inventory}) == 572
    assert Counter(row["level"] for row in standard_inventory) == Counter(
        {"1": 48, "2": 81, "3": 81, "4": 76, "5": 71, "6": 67, "7-9": 148}
    )
    assert Counter(row["legacy_regex_status"] for row in standard_inventory) == Counter(
        {"draft_unreviewed": 474, "not_available": 98}
    )
    assert graph["nodes"] == 27534
    assert graph["edges"] == 61722
    assert graph["candidate_integrity_counts"]["activates_runtime_rule"] == 2251
    assert graph["integrity_pass"] is True
    assert graph_v2["inventory_scope"]["standard_grammar_points"] == 572
    assert graph_v2["inventory_scope"]["experimental_grammar_points_preserved"] == 37
    assert graph_v2["total"]["nodes"] == 28158
    assert graph_v2["total"]["edges"] == 64046
    assert graph_v2["integrity"]["integrity_pass"] is True
    assert full_coverage["legacy_drafts"]["draft_unreviewed"] == 474
    assert full_coverage["legacy_drafts"]["not_available"] == 98
    assert full_coverage["legacy_drafts"]["runtime_eligible"] == 0
    assert all(full_coverage["integrity"].values())
    assert page_audit["rules"] == 144
    assert page_audit["grammar_points"] == 16
    assert page_audit["all_points_verified"] is True
    assert audit["pass"] is True
    assert audit["audit_version"] == "1.2"
    assert audit["cross_provider_files_audited"] == 10
    assert audit["v3_files_audited"] >= 15
    assert audit["handbook_alignment_rows_checked"] == 572
    assert audit["authority_metadata_exemptions"] == 2

    forbidden_headers = {
        "句子", "候选片段", "最终槽位片段（从句子中复制；条件必填）",
        "标注者1_主标签：是否为目标语法点", "最终备注/疑问（选填）",
    }
    for filename in (
        "data/main_labels_release.csv",
        "data/evidence_labels_release.csv",
        "data/slot_labels_release.csv",
    ):
        headers = set(read_csv(filename)[0])
        if headers & forbidden_headers:
            raise AssertionError(f"restricted columns in {filename}: {headers & forbidden_headers}")

    main_results = read_csv("results/main_results_aggregate.csv")
    assert_close(lookup(main_results, "target_label", "char_tfidf_svm", "accuracy"), 0.9432999443516973)
    assert_close(lookup(main_results, "target_label", "char_tfidf_svm", "macro_f1"), 0.8899356879739435)
    assert_close(lookup(main_results, "target_label", "point_majority", "accuracy"), 0.8932684103134854)
    assert_close(lookup(main_results, "target_label", "point_majority", "macro_f1"), 0.6988816359378198)
    assert_close(lookup(main_results, "target_label", "point_majority", "hard_negative_accuracy"), 0.30303563551043955)
    assert_close(lookup(main_results, "target_label", "char_svm_plus_kg_constraints", "accuracy"), 0.9549731033203487)
    assert_close(lookup(main_results, "target_label", "char_svm_plus_kg_constraints", "macro_f1"), 0.9107981370538711)
    assert_close(lookup(main_results, "target_label", "char_svm_plus_kg_constraints", "hard_negative_accuracy"), 0.8552158927298904)

    regime_results = read_csv("results/diagnostic_regime_results_aggregate.csv")
    assert len(regime_results) == 10
    assert_close(lookup_regime(regime_results, "structural_pattern", "char_tfidf_lr", "macro_f1_mean"), 0.8908884204492666)
    assert_close(lookup_regime(regime_results, "structural_pattern", "char_lr_plus_kg_markers", "macro_f1_mean"), 0.8962601260142787)
    assert_close(lookup_regime(regime_results, "structural_pattern", "char_lr_plus_kg_constraints", "macro_f1_mean"), 0.9219423262451814)
    assert_close(lookup_regime(regime_results, "structural_pattern", "char_lr_plus_kg_constraints", "hard_negative_accuracy_mean"), 0.8846087822295464)
    assert_close(lookup_regime(regime_results, "complex_functional", "char_tfidf_lr", "macro_f1_mean"), 0.8117984314452602)
    assert_close(lookup_regime(regime_results, "complex_functional", "char_lr_plus_kg_markers", "macro_f1_mean"), 0.8501490727322296)
    assert_close(lookup_regime(regime_results, "complex_functional", "char_lr_plus_kg_constraints", "macro_f1_mean"), 0.8638367548510767)
    assert_close(lookup_regime(regime_results, "complex_functional", "char_lr_plus_kg_constraints", "hard_negative_accuracy_mean"), 0.9249206349206349)

    regime_by_seed = read_csv("results/diagnostic_regime_results_by_seed.csv")
    assert len(regime_by_seed) == 30
    for seed in ("13", "42", "2027"):
        for regime in ("structural_pattern", "complex_functional"):
            marker = lookup_regime(
                [row for row in regime_by_seed if row["seed"] == seed],
                regime,
                "char_lr_plus_kg_markers",
                "macro_f1",
            )
            constrained = lookup_regime(
                [row for row in regime_by_seed if row["seed"] == seed],
                regime,
                "char_lr_plus_kg_constraints",
                "macro_f1",
            )
            assert constrained > marker

    aux_results = read_csv("results/aux_results_aggregate.csv")
    assert_close(lookup(aux_results, "structure_evidence_support", "char_tfidf_lr", "macro_f1"), 0.6894502131433095)
    assert_close(lookup(aux_results, "structure_evidence_support", "char_lr_plus_kg", "macro_f1"), 0.7242404940656847)
    assert_close(lookup(aux_results, "slot_span", "executable_slot_rules", "exact_span_f1"), 0.37667067660368553)
    assert_close(lookup(aux_results, "slot_span", "executable_slot_rules", "relaxed_span_f1"), 0.6456443165440198)
    assert_close(lookup(aux_results, "slot_span", "executable_slot_rules", "cue_grounding"), 1.0)

    trace_results = read_csv("results/trace_state_aggregate.csv")
    trace_by_state = {row["state"]: row for row in trace_results}
    assert_close(float(trace_by_state["accept"]["coverage_mean"]), 0.6787061769616026)
    assert_close(float(trace_by_state["accept"]["target_rate_mean"]), 0.8959558912102022)
    assert_close(float(trace_by_state["strong_reject"]["coverage_mean"]), 0.024457429048414026)
    assert_close(float(trace_by_state["strong_reject"]["non_target_rate_mean"]), 0.7946428571428571)
    assert_close(float(trace_by_state["weak_reject"]["non_target_rate_mean"]), 0.3217446799146146)

    macbert = read_csv("results/macbert_results_by_seed.csv")
    assert len(macbert) == 3
    assert_close(sum(float(row["accuracy"]) for row in macbert) / 3, 0.9532990168799852)
    assert_close(sum(float(row["macro_f1"]) for row in macbert) / 3, 0.901016933973748)

    llm_global = json.loads(
        (ROOT / "results/llm_global/summary_release.json").read_text(encoding="utf-8")
    )
    llm_strong = json.loads(
        (ROOT / "results/llm_strong/summary_release.json").read_text(encoding="utf-8")
    )
    llm_governor = json.loads(
        (ROOT / "results/llm_governor/summary.json").read_text(encoding="utf-8")
    )
    llm_comparison = json.loads(
        (ROOT / "results/llm_comparison/summary.json").read_text(encoding="utf-8")
    )
    qwen_cross_provider = verify_cross_provider_release(
        "results/llm_cross_provider/qwen_plus",
        provider="qwen",
        requested_model="qwen-plus",
        response_model="qwen-plus",
        selected_variant="point_retrieved",
        dev_point_macro_f1=0.655864635805145,
        dev_textkg_macro_f1=0.6529464285714286,
        test_accuracy=0.735,
        test_macro_f1=0.6707790527133121,
        test_negative_accuracy=0.9263157894736842,
    )
    deepseek_cross_provider = verify_cross_provider_release(
        "results/llm_cross_provider/deepseek_v4_flash",
        provider="deepseek",
        requested_model="deepseek-chat",
        response_model="deepseek-v4-flash",
        selected_variant="point_retrieved_textkg",
        dev_point_macro_f1=0.6055355321319279,
        dev_textkg_macro_f1=0.6082541762201114,
        test_accuracy=0.67,
        test_macro_f1=0.6208639705882353,
        test_negative_accuracy=0.9789473684210527,
    )

    cross_provider_rows = read_csv(
        "results/llm_comparison/cross_provider_llm_results_seed42.csv"
    )
    cross_provider_by_system = {row["system"]: row for row in cross_provider_rows}
    assert len(cross_provider_rows) == len(cross_provider_by_system) == 3
    assert set(cross_provider_by_system) == {
        "GPT-5.6 Terra", "Qwen-Plus", "DeepSeek-V4-Flash"
    }
    for system, summary, provider, requested_model, response_model in (
        ("Qwen-Plus", qwen_cross_provider, "Alibaba Cloud", "qwen-plus", "qwen-plus"),
        (
            "DeepSeek-V4-Flash",
            deepseek_cross_provider,
            "DeepSeek",
            "deepseek-chat",
            "deepseek-v4-flash",
        ),
    ):
        row = cross_provider_by_system[system]
        assert row["provider"] == provider
        assert row["requested_model"] == requested_model
        assert row["response_model"] == response_model
        assert row["selected_variant"] == summary["selected_variant"]
        assert int(row["test_n"]) == summary["test"]["n"] == 600
        assert_close(
            float(row["dev_point_macro_f1"]),
            summary["development"]["point_retrieved"]["macro_f1"],
        )
        assert_close(
            float(row["dev_textkg_macro_f1"]),
            summary["development"]["point_retrieved_textkg"]["macro_f1"],
        )
        assert_close(float(row["test_accuracy"]), summary["test"]["accuracy"])
        assert_close(float(row["test_macro_f1"]), summary["test"]["macro_f1"])
        assert_close(float(row["test_negative_accuracy"]), summary["test"]["negative_accuracy"])

    assert llm_global["protocol"]["model"] == "gpt-5.6-terra"
    assert llm_global["protocol"]["store"] is False
    global_main_rows = read_csv("results/llm_global/main_predictions_release.csv")
    assert len(global_main_rows) == 600
    assert len(read_csv("results/llm_global/evidence_predictions_release.csv")) == 200
    assert len(read_csv("results/llm_global/slot_predictions_release.csv")) == 264
    direct_rows = read_csv("results/llm_global/direct_audit_release.csv")
    rule_rows = read_csv("results/llm_global/rule_audit_release.csv")
    assert len(direct_rows) == 360
    assert len(rule_rows) == 360
    global_scores = binary_scores(global_main_rows)
    assert_close(global_scores["accuracy"], llm_global["main"]["accuracy"])
    assert_close(global_scores["macro_f1"], llm_global["main"]["macro_f1"])
    assert_close(global_scores["negative_accuracy"], llm_global["main"]["negative_accuracy"])
    assert_close(exact_repeat_stability(direct_rows, "prediction"), 0.95)
    assert_close(exact_repeat_stability(rule_rows, "prediction"), 0.9)
    assert_close(exact_repeat_stability(rule_rows, "rule_id"), 0.7833333333333333)
    assert llm_strong["selected_variant"] == "point_retrieved"
    assert_close(llm_strong["development"]["point_retrieved"]["macro_f1"], 0.7252960391996733)
    assert_close(llm_strong["development"]["point_retrieved_textkg"]["macro_f1"], 0.7018463715903243)
    assert_close(llm_strong["test"]["accuracy"], 0.815)
    assert_close(llm_strong["test"]["macro_f1"], 0.7395089783354389)
    assert_close(llm_strong["test"]["negative_accuracy"], 0.8736842105263158)
    strong_rows = read_csv("results/llm_strong/selected_test_predictions_release.csv")
    strong_scores = binary_scores(strong_rows)
    assert_close(strong_scores["accuracy"], llm_strong["test"]["accuracy"])
    assert_close(strong_scores["macro_f1"], llm_strong["test"]["macro_f1"])
    assert_close(strong_scores["negative_accuracy"], llm_strong["test"]["negative_accuracy"])

    governor_rows = read_csv("results/llm_governor/test_predictions_release.csv")
    governor_scores = binary_scores(governor_rows, "governed_prediction")
    assert_close(governor_scores["accuracy"], 0.8783333333333333)
    assert_close(governor_scores["macro_f1"], 0.8128005128753072)
    assert_close(governor_scores["negative_accuracy"], 0.9052631578947369)

    main_metrics = llm_comparison["main_metrics"]
    assert_close(main_metrics["char_svm_plus_kg_constraints"]["macro_f1"], 0.9201171614964718)
    assert_close(main_metrics["macbert_base_finetuned"]["macro_f1"], 0.9258507020004191)
    assert_close(main_metrics["gpt_5_6_terra_point_4shot_medium"]["macro_f1"], 0.7395089783354389)
    assert_close(
        main_metrics["gpt_5_6_terra_point_4shot_plus_exec_kg_governor"]["macro_f1"],
        0.8128005128753072,
    )
    complementarity = llm_comparison["complementarity_exec_kg_vs_strong_llm"]
    assert complementarity["llm_high_confidence_wrong"] == 108
    assert complementarity["llm_high_confidence_wrong_corrected_by_kg"] == 101
    assert_close(llm_comparison["direct_explanation_audit"]["cue_exact_stability"], 0.575)
    assert_close(llm_comparison["rule_shaped_explanation_audit"]["cue_exact_stability"], 0.4083333333333333)
    assert_close(llm_comparison["rule_shaped_explanation_audit"]["rule_exact_stability"], 0.7833333333333333)
    assert_close(llm_comparison["rule_shaped_explanation_audit"]["valid_rule_id_rate"], 1.0)
    assert_close(llm_comparison["formal_api_cost_usd"], 10.65250525)
    assert llm_governor["status"] == "post_hoc_exploratory"
    assert_close(llm_governor["test"]["llm_plus_exec_kg_governor"]["macro_f1"], 0.8128005128753072)

    pair_rows = read_csv("results/llm_comparison/paired_group_bootstrap_differences.csv")
    llm_pair = [
        row for row in pair_rows
        if row["model_a"] == "char_svm_plus_kg_constraints"
        and row["model_b"] == "gpt_5_6_terra_point_4shot_medium"
        and row["metric"] == "macro_f1"
    ]
    assert len(llm_pair) == 1
    assert_close(float(llm_pair[0]["point_delta_a_minus_b"]), 0.1806081831610329)
    assert_close(float(llm_pair[0]["delta_ci_low"]), 0.13332729767888005)
    assert_close(float(llm_pair[0]["delta_ci_high"]), 0.22788018300370844)

    audit_rows = read_csv("results/llm_comparison/paired_audit_bootstrap.csv")
    cue_pair = [row for row in audit_rows if row["metric"] == "cue_exact_stability"]
    assert len(cue_pair) == 1
    assert_close(float(cue_pair[0]["delta_rule_minus_direct"]), -0.16666666666666666)
    assert_close(float(cue_pair[0]["bootstrap_p_two_sided"]), 0.0052)

    llm_forbidden_headers = {"sentence", "candidate_span", "evidence_cues", "reason", "span"}
    for filename in (
        "results/llm_global/main_predictions_release.csv",
        "results/llm_global/evidence_predictions_release.csv",
        "results/llm_global/slot_predictions_release.csv",
        "results/llm_global/direct_audit_release.csv",
        "results/llm_global/rule_audit_release.csv",
        "results/llm_strong/selected_test_predictions_release.csv",
        "results/llm_cross_provider/qwen_plus/release_safe/dev_point_retrieved_predictions_release.csv",
        "results/llm_cross_provider/qwen_plus/release_safe/dev_point_retrieved_textkg_predictions_release.csv",
        "results/llm_cross_provider/qwen_plus/release_safe/selected_test_predictions_release.csv",
        "results/llm_cross_provider/deepseek_v4_flash/release_safe/dev_point_retrieved_predictions_release.csv",
        "results/llm_cross_provider/deepseek_v4_flash/release_safe/dev_point_retrieved_textkg_predictions_release.csv",
        "results/llm_cross_provider/deepseek_v4_flash/release_safe/selected_test_predictions_release.csv",
    ):
        headers = set(read_csv(filename)[0])
        if headers & llm_forbidden_headers:
            raise AssertionError(f"restricted LLM columns in {filename}: {headers & llm_forbidden_headers}")

    print(json.dumps({
        "status": "PASS",
        "main_instances": len(main_labels),
        "sentence_groups": len({row["sentence_group"] for row in main_labels}),
        "grammar_points": len(points),
        "official_standard_points": len(standard_inventory),
        "compiled_diagnostic_units": len(crosswalk),
        "compiled_diagnostic_profiles": len({row["diagnostic_profile_id"] for row in crosswalk}),
        "compiled_runtime_bindings": len(bindings_v3["bindings"]),
        "operational_templates": len(templates),
        "binding_contract_tests": len(contract_tests),
        "normalized_regex_bindings": 503,
        "unit_specialized_split_bindings": 39,
        "source_literal_scaffold_bindings": 143,
        "taxonomy_paths_covered": 103,
        "handbook_alignments": len(handbook_alignment),
        "handbook_alignment_high": 569,
        "handbook_alignment_medium": 3,
        "catalog_field_corrections": len(corrections),
        "catalog_points_corrected": len({row["standard_point_id"] for row in corrections}),
        "standard_ocr_anchors": 362,
        "standard_appendix_span_anchors": 210,
        "positive_evidence_pairs": example_coverage["unique_sentence_point_pairs"],
        "points_with_annotated_positive": example_coverage["points_with_annotated_positive"],
        "legacy_regex_drafts_unreviewed": 474,
        "official_points_without_legacy_regex": 98,
        "diagnostic_rule_records": len(rules),
        "point_specific_runtime_bindings": 16,
        "page_anchor_audit": page_audit["all_points_verified"],
        "graph_nodes": graph["nodes"],
        "graph_edges": graph["edges"],
        "full_inventory_graph_nodes": graph_v2["total"]["nodes"],
        "full_inventory_graph_edges": graph_v2["total"]["edges"],
        "full_compilation_graph_nodes": graph_v3["total"]["nodes"],
        "full_compilation_graph_edges": graph_v3["total"]["edges"],
        "v2_nodes_preserved_in_v3": graph_v3["validation"]["v2_node_ids_preserved"],
        "v2_edges_preserved_in_v3": graph_v3["validation"]["v2_edges_preserved"],
        "release_audit": audit["pass"],
        "release_audited_files": audit["audited_files"],
        "release_cross_provider_files_audited": audit["cross_provider_files_audited"],
        "external_llm_test_instances": 600,
        "cross_provider_llm_systems": 2,
        "cross_provider_dev_predictions": 1196,
        "cross_provider_test_predictions": 1200,
        "external_llm_explanation_calls": 720,
        "external_llm_formal_cost_usd": llm_comparison["formal_api_cost_usd"],
        "headline_values_verified": 61,
    }, indent=2))


if __name__ == "__main__":
    main()
