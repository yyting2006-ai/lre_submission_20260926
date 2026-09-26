#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from run_grouped_kg_experiments import MAIN_CSV, read_csv, write_csv


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
TRADITIONAL = ROOT / "asgd_fast_experiments/outputs/exp_20260718_grouped_kg_v2/predictions_with_rule_traces.csv"
MACBERT = ROOT / "asgd_fast_experiments/outputs/exp_20260718_macbert_grouped_v1/predictions.csv"
LLM_DIRECT = ROOT / "asgd_fast_experiments/outputs/exp_20260718_openai_gpt56terra_seed42_v2"
LLM_STRONG = ROOT / "asgd_fast_experiments/outputs/exp_20260718_openai_gpt56terra_strong_seed42_v1"
LLM_GOVERNOR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_llm_kg_governor_v1"
AUX_RESULTS = ROOT / "asgd_fast_experiments/outputs/exp_20260718_grouped_auxiliary_v1/results_by_seed.csv"
BINDINGS = ROOT / "grammar_kg/kg/executable_rule_bindings_v1.json"
OUT_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_llm_comparison_v1"

SEED = 42
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20270719
NEGATIVE = "不是目标语法点"
POSITIVE = "是目标语法点"
MODELS = {
    "point_majority": "Point majority",
    "char_tfidf_svm": "Char SVM",
    "char_svm_plus_kg_constraints": "Char SVM + ExecKG",
    "macbert_base_finetuned": "MacBERT",
    "gpt_5_6_terra_global_6shot_low": "GPT-5.6 Terra global 6-shot",
    "gpt_5_6_terra_point_4shot_medium": "GPT-5.6 Terra point-matched 4-shot",
    "gpt_5_6_terra_point_4shot_plus_exec_kg_governor": "GPT-5.6 Terra + ExecKG governor",
}
PAIRS = [
    ("char_svm_plus_kg_constraints", "gpt_5_6_terra_point_4shot_medium"),
    ("char_svm_plus_kg_constraints", "gpt_5_6_terra_point_4shot_plus_exec_kg_governor"),
    ("macbert_base_finetuned", "gpt_5_6_terra_point_4shot_medium"),
    ("macbert_base_finetuned", "gpt_5_6_terra_point_4shot_plus_exec_kg_governor"),
    ("char_tfidf_svm", "gpt_5_6_terra_point_4shot_medium"),
    ("char_tfidf_svm", "gpt_5_6_terra_point_4shot_plus_exec_kg_governor"),
    ("gpt_5_6_terra_point_4shot_plus_exec_kg_governor", "gpt_5_6_terra_point_4shot_medium"),
    ("gpt_5_6_terra_point_4shot_medium", "gpt_5_6_terra_global_6shot_low"),
    ("char_svm_plus_kg_constraints", "macbert_base_finetuned"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalize_row(row: dict[str, str], model: str) -> dict[str, str]:
    return {
        **row,
        "model": model,
        "correct": str(int(row["gold"] == row["prediction"])),
    }


def load_models() -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(TRADITIONAL):
        if int(row["seed"]) == SEED and row["task"] == "target_label" and row["model"] in MODELS:
            output[row["model"]].append(normalize_row(row, row["model"]))
    for row in read_rows(MACBERT):
        if int(row["seed"]) == SEED:
            output["macbert_base_finetuned"].append(
                normalize_row(row, "macbert_base_finetuned")
            )
    for row in read_rows(LLM_DIRECT / "release_safe/main_predictions_release.csv"):
        output["gpt_5_6_terra_global_6shot_low"].append(
            normalize_row(row, "gpt_5_6_terra_global_6shot_low")
        )
    for row in read_rows(LLM_STRONG / "release_safe/selected_test_predictions_release.csv"):
        output["gpt_5_6_terra_point_4shot_medium"].append(
            normalize_row(row, "gpt_5_6_terra_point_4shot_medium")
        )
    for row in read_rows(LLM_GOVERNOR / "test_predictions_release.csv"):
        converted = {
            **row,
            "prediction": row["governed_prediction"],
            "confidence": row["llm_confidence"],
        }
        output["gpt_5_6_terra_point_4shot_plus_exec_kg_governor"].append(
            normalize_row(converted, "gpt_5_6_terra_point_4shot_plus_exec_kg_governor")
        )
    if set(output) != set(MODELS):
        raise ValueError(f"Missing models: {set(MODELS) - set(output)}")
    reference = {(row["编号"], row["sentence_group"], row["gold"]) for row in next(iter(output.values()))}
    for model, rows in output.items():
        keys = {(row["编号"], row["sentence_group"], row["gold"]) for row in rows}
        if keys != reference or len(rows) != 600:
            raise ValueError(f"Alignment failure for {model}: {len(rows)} rows")
    return output


def score(gold: list[str], pred: list[str]) -> dict[str, float]:
    labels = [NEGATIVE, POSITIVE]
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "hard_negative_accuracy": float(recall[0]),
        "target_precision": float(precision[1]),
        "target_recall": float(recall[1]),
        "target_f1": float(f1[1]),
        "non_target_precision": float(precision[0]),
        "non_target_recall": float(recall[0]),
        "non_target_f1": float(f1[0]),
    }


def bootstrap(
    rows_by_model: dict[str, list[dict[str, str]]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    keyed = {
        model: {(row["编号"], row["sentence_group"]): row for row in rows}
        for model, rows in rows_by_model.items()
    }
    keys = sorted(next(iter(keyed.values())))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key[1]].append(index)
    group_ids = sorted(groups)
    label_to_id = {NEGATIVE: 0, POSITIVE: 1}
    gold = np.asarray([label_to_id[keyed[next(iter(keyed))][key]["gold"]] for key in keys])
    predictions = {
        model: np.asarray([label_to_id[keyed[model][key]["prediction"]] for key in keys])
        for model in keyed
    }

    def fast(prediction: np.ndarray, indices: np.ndarray) -> dict[str, float]:
        y = gold[indices]
        p = prediction[indices]
        confusion = np.bincount(y * 2 + p, minlength=4).reshape(2, 2)
        true_positive = np.diag(confusion).astype(float)
        false_positive = confusion.sum(axis=0) - true_positive
        false_negative = confusion.sum(axis=1) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        per_f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator != 0,
        )
        negative_mask = y == 0
        return {
            "accuracy": float(np.mean(y == p)),
            "macro_f1": float(np.mean(per_f1)),
            "hard_negative_accuracy": float(np.mean(p[negative_mask] == 0)),
        }

    full = np.arange(len(keys), dtype=int)
    points = {model: fast(pred, full) for model, pred in predictions.items()}
    samples = {
        model: {metric: [] for metric in ("accuracy", "macro_f1", "hard_negative_accuracy")}
        for model in predictions
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
        indices = np.concatenate([np.asarray(groups[group_id], dtype=int) for group_id in sampled_groups])
        for model, pred in predictions.items():
            values = fast(pred, indices)
            for metric, value in values.items():
                samples[model][metric].append(value)

    ci_rows: list[dict[str, object]] = []
    for model in MODELS:
        for metric, values in samples[model].items():
            array = np.asarray(values)
            ci_rows.append(
                {
                    "model": model,
                    "model_display": MODELS[model],
                    "metric": metric,
                    "point": points[model][metric],
                    "ci_low": float(np.quantile(array, 0.025)),
                    "ci_high": float(np.quantile(array, 0.975)),
                    "bootstrap_sd": float(np.std(array, ddof=1)),
                    "n_sentence_groups": len(group_ids),
                    "n_candidates": len(keys),
                    "n_bootstrap": BOOTSTRAP_SAMPLES,
                }
            )

    pair_rows: list[dict[str, object]] = []
    mcnemar_rows: list[dict[str, object]] = []
    for model_a, model_b in PAIRS:
        for metric in ("accuracy", "macro_f1", "hard_negative_accuracy"):
            difference = np.asarray(samples[model_a][metric]) - np.asarray(samples[model_b][metric])
            pair_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "point_delta_a_minus_b": points[model_a][metric] - points[model_b][metric],
                    "delta_ci_low": float(np.quantile(difference, 0.025)),
                    "delta_ci_high": float(np.quantile(difference, 0.975)),
                    "bootstrap_p_two_sided": min(
                        1.0,
                        2.0
                        * min(
                            float(np.mean(difference <= 0.0)),
                            float(np.mean(difference >= 0.0)),
                        ),
                    ),
                    "n_bootstrap": BOOTSTRAP_SAMPLES,
                }
            )
        a_correct = {
            key: keyed[model_a][key]["gold"] == keyed[model_a][key]["prediction"] for key in keys
        }
        b_correct = {
            key: keyed[model_b][key]["gold"] == keyed[model_b][key]["prediction"] for key in keys
        }
        a_only = sum(a_correct[key] and not b_correct[key] for key in keys)
        b_only = sum(b_correct[key] and not a_correct[key] for key in keys)
        discordant = a_only + b_only
        p = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
        mcnemar_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "a_only_correct": a_only,
                "b_only_correct": b_only,
                "discordant": discordant,
                "exact_mcnemar_p": p,
                "note": "Candidate-level descriptive; sentence-group bootstrap is primary.",
            }
        )
    return ci_rows, pair_rows, mcnemar_rows


def subgroup_rows(
    rows_by_model: dict[str, list[dict[str, str]]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    reference = {row["编号"]: row for row in rows_by_model["char_svm_plus_kg_constraints"]}
    gold_rows = {row["编号"]: row for row in read_csv(MAIN_CSV)}
    bound_points = set(json.loads(BINDINGS.read_text(encoding="utf-8"))["rules"])
    family: list[dict[str, object]] = []
    confidence: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for model, rows in rows_by_model.items():
        by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_tier: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_coverage: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            ref = reference[row["编号"]]
            by_family[ref["语法家族"]].append(row)
            by_tier[gold_rows[row["编号"]]["金标置信层级"]].append(row)
            state = "bound_point" if ref["候选语法点"] in bound_points else "unbound_point"
            by_coverage[state].append(row)
        for value, items in sorted(by_family.items()):
            family.append({"model": model, "grammar_family": value, "n": len(items), **score([x["gold"] for x in items], [x["prediction"] for x in items])})
        for value, items in sorted(by_tier.items()):
            confidence.append({"model": model, "gold_confidence_tier": value, "n": len(items), **score([x["gold"] for x in items], [x["prediction"] for x in items])})
        for value, items in sorted(by_coverage.items()):
            coverage.append({"model": model, "coverage": value, "n": len(items), **score([x["gold"] for x in items], [x["prediction"] for x in items])})
    return family, confidence, coverage


def trace_state_rows(rows_by_model: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    reference = {row["编号"]: row for row in rows_by_model["char_svm_plus_kg_constraints"]}
    output: list[dict[str, object]] = []
    for model, rows in rows_by_model.items():
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            ref = reference[row["编号"]]
            if ref.get("kg_strong_reject") == "1":
                state = "strong_reject"
            elif ref.get("kg_weak_reject") == "1":
                state = "weak_reject"
            elif ref.get("kg_accepted") == "1":
                state = "accept"
            else:
                state = "no_decision"
            grouped[state].append(row)
        for state, items in sorted(grouped.items()):
            output.append(
                {
                    "model": model,
                    "trace_state": state,
                    "n": len(items),
                    **score([x["gold"] for x in items], [x["prediction"] for x in items]),
                }
            )
    return output


def complementarity(rows_by_model: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    kg = {row["编号"]: row for row in rows_by_model["char_svm_plus_kg_constraints"]}
    llm = {row["编号"]: row for row in rows_by_model["gpt_5_6_terra_point_4shot_medium"]}
    ids = sorted(kg)
    kg_correct = {row_id: kg[row_id]["gold"] == kg[row_id]["prediction"] for row_id in ids}
    llm_correct = {row_id: llm[row_id]["gold"] == llm[row_id]["prediction"] for row_id in ids}
    high_conf_wrong = [
        row_id
        for row_id in ids
        if not llm_correct[row_id] and float(llm[row_id].get("confidence") or 0.0) >= 0.9
    ]
    return {
        "both_correct": sum(kg_correct[row_id] and llm_correct[row_id] for row_id in ids),
        "kg_only_correct": sum(kg_correct[row_id] and not llm_correct[row_id] for row_id in ids),
        "llm_only_correct": sum(llm_correct[row_id] and not kg_correct[row_id] for row_id in ids),
        "both_wrong": sum(not kg_correct[row_id] and not llm_correct[row_id] for row_id in ids),
        "oracle_union_accuracy": float(np.mean([kg_correct[row_id] or llm_correct[row_id] for row_id in ids])),
        "llm_high_confidence_wrong": len(high_conf_wrong),
        "llm_high_confidence_wrong_corrected_by_kg": sum(kg_correct[row_id] for row_id in high_conf_wrong),
    }


def auxiliary_comparison() -> list[dict[str, object]]:
    rows = read_rows(AUX_RESULTS)
    selected: list[dict[str, object]] = []
    wanted = {
        ("structure_evidence_support", "char_tfidf_lr"),
        ("structure_evidence_support", "char_lr_plus_kg"),
        ("slot_status", "slot_char_tfidf_lr"),
        ("slot_span", "executable_slot_rules"),
    }
    for row in rows:
        if int(row["seed"]) == SEED and (row["task"], row["model"]) in wanted:
            selected.append(dict(row))
    llm = json.loads((LLM_DIRECT / "summary.json").read_text(encoding="utf-8"))
    selected.extend(
        [
            {
                "seed": SEED,
                "task": "structure_evidence_support",
                "model": "gpt_5_6_terra_global_6shot_low",
                "accuracy": llm["evidence"]["accuracy"],
                "macro_f1": llm["evidence"]["macro_f1"],
                "negative_accuracy": llm["evidence"]["negative_accuracy"],
                "test_n": llm["evidence"]["n"],
            },
            {
                "seed": SEED,
                "task": "slot_status",
                "model": "gpt_5_6_terra_global_6shot_low",
                "accuracy": llm["slot"]["status_accuracy"],
                "macro_f1": llm["slot"]["status_macro_f1"],
                "test_n": llm["slot"]["n"],
            },
            {
                "seed": SEED,
                "task": "slot_span",
                "model": "gpt_5_6_terra_global_6shot_low",
                "exact_span_f1": llm["slot"]["exact_span_f1"],
                "relaxed_span_f1": llm["slot"]["relaxed_span_f1"],
                "cue_grounding": llm["slot"]["span_grounding_rate"],
                "test_n": llm["slot"]["n"],
            },
        ]
    )
    return selected


def pairwise_jaccard(items: list[tuple[str, ...]]) -> float:
    scores: list[float] = []
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            left, right = set(items[left_index]), set(items[right_index])
            scores.append(len(left & right) / len(left | right) if left or right else 1.0)
    return float(np.mean(scores)) if scores else 1.0


def audit_candidate_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: int(row["repeat"]))
    return {
        "label_stability": float(len({row["prediction"] for row in ordered}) == 1),
        "cue_exact_stability": float(
            len({tuple(sorted(row["evidence_cues"])) for row in ordered}) == 1
        ),
        "cue_pairwise_jaccard": pairwise_jaccard(
            [tuple(sorted(row["evidence_cues"])) for row in ordered]
        ),
        "reason_exact_stability": float(len({row["reason"] for row in ordered}) == 1),
        "call_accuracy": float(np.mean([row["gold"] == row["prediction"] for row in ordered])),
        "overconfident_wrong_rate": float(
            np.mean(
                [
                    row["gold"] != row["prediction"] and float(row["confidence"]) >= 0.9
                    for row in ordered
                ]
            )
        ),
    }


def paired_audit_analysis() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    def grouped(path: Path) -> dict[str, dict[str, float]]:
        rows: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]:
            if item.get("valid_output") and not item.get("error"):
                rows[item["编号"]].append(item)
        return {row_id: audit_candidate_metrics(items) for row_id, items in rows.items() if len(items) == 3}

    direct = grouped(LLM_DIRECT / "private_direct_audit_raw.jsonl")
    rules = grouped(LLM_DIRECT / "private_rule_audit_raw.jsonl")
    if set(direct) != set(rules) or len(direct) != 120:
        raise ValueError("Direct/rule audit alignment failure")
    ids = sorted(direct)
    metrics = list(next(iter(direct.values())))
    rng = np.random.default_rng(20270720)
    difference_samples = {metric: [] for metric in metrics}
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        for metric in metrics:
            difference_samples[metric].append(
                float(np.mean([rules[row_id][metric] - direct[row_id][metric] for row_id in sampled]))
            )
    bootstrap_rows: list[dict[str, object]] = []
    for metric in metrics:
        differences = np.asarray(difference_samples[metric])
        point = float(np.mean([rules[row_id][metric] - direct[row_id][metric] for row_id in ids]))
        bootstrap_rows.append(
            {
                "metric": metric,
                "direct_point": float(np.mean([direct[row_id][metric] for row_id in ids])),
                "rule_shaped_point": float(np.mean([rules[row_id][metric] for row_id in ids])),
                "delta_rule_minus_direct": point,
                "delta_ci_low": float(np.quantile(differences, 0.025)),
                "delta_ci_high": float(np.quantile(differences, 0.975)),
                "bootstrap_p_two_sided": min(
                    1.0,
                    2.0
                    * min(
                        float(np.mean(differences <= 0.0)),
                        float(np.mean(differences >= 0.0)),
                    ),
                ),
                "n_candidates": len(ids),
                "n_bootstrap": BOOTSTRAP_SAMPLES,
            }
        )
    mcnemar_rows: list[dict[str, object]] = []
    for metric in ("label_stability", "cue_exact_stability", "reason_exact_stability"):
        direct_only = sum(direct[row_id][metric] == 1 and rules[row_id][metric] == 0 for row_id in ids)
        rules_only = sum(direct[row_id][metric] == 0 and rules[row_id][metric] == 1 for row_id in ids)
        discordant = direct_only + rules_only
        p = float(binomtest(min(direct_only, rules_only), discordant, 0.5).pvalue) if discordant else 1.0
        mcnemar_rows.append(
            {
                "metric": metric,
                "direct_only": direct_only,
                "rule_shaped_only": rules_only,
                "discordant": discordant,
                "exact_mcnemar_p": p,
            }
        )
    return bootstrap_rows, mcnemar_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_model = load_models()
    overall = []
    for model, rows in rows_by_model.items():
        overall.append(
            {
                "seed": SEED,
                "model": model,
                "model_display": MODELS[model],
                "n": len(rows),
                **score([row["gold"] for row in rows], [row["prediction"] for row in rows]),
            }
        )
    ci_rows, pair_rows, mcnemar_rows = bootstrap(rows_by_model)
    family_rows, confidence_rows, coverage_rows = subgroup_rows(rows_by_model)
    trace_rows = trace_state_rows(rows_by_model)
    auxiliary_rows = auxiliary_comparison()
    complement = complementarity(rows_by_model)
    audit_bootstrap_rows, audit_mcnemar_rows = paired_audit_analysis()

    write_csv(OUT_DIR / "main_metrics_seed42.csv", overall)
    write_csv(OUT_DIR / "group_bootstrap_confidence_intervals.csv", ci_rows)
    write_csv(OUT_DIR / "paired_group_bootstrap_differences.csv", pair_rows)
    write_csv(OUT_DIR / "mcnemar_descriptive.csv", mcnemar_rows)
    write_csv(OUT_DIR / "family_metrics.csv", family_rows)
    write_csv(OUT_DIR / "gold_confidence_tier_metrics.csv", confidence_rows)
    write_csv(OUT_DIR / "binding_coverage_metrics.csv", coverage_rows)
    write_csv(OUT_DIR / "trace_state_metrics.csv", trace_rows)
    write_csv(OUT_DIR / "auxiliary_comparison_seed42.csv", auxiliary_rows)
    write_csv(OUT_DIR / "paired_audit_bootstrap.csv", audit_bootstrap_rows)
    write_csv(OUT_DIR / "paired_audit_mcnemar.csv", audit_mcnemar_rows)

    direct_summary = json.loads((LLM_DIRECT / "summary.json").read_text(encoding="utf-8"))
    strong_summary = json.loads((LLM_STRONG / "summary.json").read_text(encoding="utf-8"))
    governor_summary = json.loads((LLM_GOVERNOR / "summary.json").read_text(encoding="utf-8"))
    formal_cost = sum(
        direct_summary[key]["estimated_standard_api_cost_usd"]
        for key in ("main", "evidence", "slot", "direct_audit", "rule_audit")
    ) + sum(
        item["estimated_standard_api_cost_usd"]
        for item in strong_summary["development"].values()
    ) + strong_summary["test"]["estimated_standard_api_cost_usd"]
    report = {
        "seed": SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "normalized sentence group",
        "main_metrics": {row["model"]: row for row in overall},
        "complementarity_exec_kg_vs_strong_llm": complement,
        "direct_explanation_audit": direct_summary["direct_audit"],
        "rule_shaped_explanation_audit": direct_summary["rule_audit"],
        "strong_prompt_selection": {
            "development": strong_summary["development"],
            "selected_variant": strong_summary["selected_variant"],
            "test": strong_summary["test"],
        },
        "post_hoc_exploratory_governor": governor_summary,
        "paired_explanation_audit": audit_bootstrap_rows,
        "formal_api_cost_usd": formal_cost,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
