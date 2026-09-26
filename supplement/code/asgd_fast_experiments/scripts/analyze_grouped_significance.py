#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from run_grouped_kg_experiments import NEGATIVE_LABEL, metric_bundle, write_csv


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
EXP_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_grouped_kg_v2"
PREDICTIONS = EXP_DIR / "predictions_with_rule_traces.csv"
MACBERT_PREDICTIONS = ROOT / "asgd_fast_experiments/outputs/exp_20260718_macbert_grouped_v1/predictions.csv"
OUT_DIR = EXP_DIR / "statistical_analysis"
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20270718
MODEL_PAIRS = [
    ("char_tfidf_lr", "char_lr_plus_kg_constraints"),
    ("char_tfidf_svm", "char_svm_plus_kg_constraints"),
    ("macbert_base_finetuned", "char_svm_plus_kg_constraints"),
    ("char_lr_plus_kg_markers", "char_lr_plus_kg_constraints"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def score(task: str, rows: list[dict[str, str]], prediction_key: str = "prediction") -> dict[str, float]:
    bundle = metric_bundle(
        task,
        [row["gold"] for row in rows],
        [row[prediction_key] for row in rows],
    )
    return {
        "accuracy": float(bundle["accuracy"]),
        "macro_f1": float(bundle["macro_f1"]),
        "hard_negative_accuracy": float(bundle["hard_negative_accuracy"] or 0.0),
    }


def group_bootstrap(
    task: str,
    rows_by_model: dict[str, list[dict[str, str]]],
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[dict]]:
    model_names = sorted(rows_by_model)
    keyed = {
        model: {(row["编号"], row["sentence_group"]): row for row in rows}
        for model, rows in rows_by_model.items()
    }
    reference_keys = set(keyed[model_names[0]])
    for model in model_names[1:]:
        if set(keyed[model]) != reference_keys:
            raise ValueError(f"Prediction alignment mismatch for {task}: {model}")

    ordered_keys = sorted(reference_keys)
    key_to_index = {key: index for index, key in enumerate(ordered_keys)}
    groups: dict[str, list[int]] = defaultdict(list)
    for key in ordered_keys:
        groups[key[1]].append(key_to_index[key])
    group_ids = sorted(groups)
    labels = sorted({keyed[model_names[0]][key]["gold"] for key in ordered_keys})
    label_to_id = {label: index for index, label in enumerate(labels)}
    gold_ids = np.asarray([label_to_id[keyed[model_names[0]][key]["gold"]] for key in ordered_keys])
    prediction_ids = {
        model: np.asarray([label_to_id[keyed[model][key]["prediction"]] for key in ordered_keys])
        for model in model_names
    }
    negative_id = label_to_id[NEGATIVE_LABEL[task]]

    def fast_score(pred_ids: np.ndarray, indices: np.ndarray) -> dict[str, float]:
        gold = gold_ids[indices]
        pred = pred_ids[indices]
        n_labels = len(labels)
        confusion = np.bincount(
            gold * n_labels + pred, minlength=n_labels * n_labels
        ).reshape(n_labels, n_labels)
        true_positive = np.diag(confusion).astype(float)
        false_positive = confusion.sum(axis=0) - true_positive
        false_negative = confusion.sum(axis=1) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        per_label_f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator != 0,
        )
        negative_mask = gold == negative_id
        return {
            "accuracy": float(np.mean(gold == pred)),
            "macro_f1": float(np.mean(per_label_f1)),
            "hard_negative_accuracy": float(np.mean(pred[negative_mask] == negative_id)) if np.any(negative_mask) else 0.0,
        }

    full_indices = np.arange(len(ordered_keys), dtype=int)
    point_scores = {
        model: fast_score(prediction_ids[model], full_indices)
        for model in model_names
    }
    boot_scores = {
        model: {metric: [] for metric in ("accuracy", "macro_f1", "hard_negative_accuracy")}
        for model in model_names
    }
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled_indices = np.concatenate([np.asarray(groups[group_id], dtype=int) for group_id in sampled_groups])
        for model in model_names:
            values = fast_score(prediction_ids[model], sampled_indices)
            for metric, value in values.items():
                boot_scores[model][metric].append(value)

    ci_rows: list[dict] = []
    for model in model_names:
        for metric, values in boot_scores[model].items():
            array = np.asarray(values)
            ci_rows.append({
                "task": task,
                "model": model,
                "metric": metric,
                "point": point_scores[model][metric],
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
                "bootstrap_sd": float(np.std(array, ddof=1)),
                "n_sentence_groups": len(group_ids),
                "n_candidates": len(reference_keys),
                "n_bootstrap": BOOTSTRAP_SAMPLES,
            })

    pair_rows: list[dict] = []
    mcnemar_rows: list[dict] = []
    for baseline, kg_model in MODEL_PAIRS:
        if baseline not in keyed or kg_model not in keyed:
            continue
        for metric in ("accuracy", "macro_f1", "hard_negative_accuracy"):
            diff = np.asarray(boot_scores[kg_model][metric]) - np.asarray(boot_scores[baseline][metric])
            pair_rows.append({
                "task": task,
                "baseline": baseline,
                "kg_model": kg_model,
                "metric": metric,
                "point_delta": point_scores[kg_model][metric] - point_scores[baseline][metric],
                "delta_ci_low": float(np.quantile(diff, 0.025)),
                "delta_ci_high": float(np.quantile(diff, 0.975)),
                "bootstrap_p_two_sided": min(
                    1.0,
                    2.0 * min(float(np.mean(diff <= 0.0)), float(np.mean(diff >= 0.0))),
                ),
                "n_sentence_groups": len(group_ids),
                "n_bootstrap": BOOTSTRAP_SAMPLES,
            })

        baseline_correct = {
            key: keyed[baseline][key]["gold"] == keyed[baseline][key]["prediction"]
            for key in reference_keys
        }
        kg_correct = {
            key: keyed[kg_model][key]["gold"] == keyed[kg_model][key]["prediction"]
            for key in reference_keys
        }
        baseline_only = sum(baseline_correct[key] and not kg_correct[key] for key in reference_keys)
        kg_only = sum(kg_correct[key] and not baseline_correct[key] for key in reference_keys)
        discordant = baseline_only + kg_only
        p_value = float(binomtest(min(baseline_only, kg_only), discordant, 0.5).pvalue) if discordant else 1.0
        mcnemar_rows.append({
            "task": task,
            "baseline": baseline,
            "kg_model": kg_model,
            "baseline_only_correct": baseline_only,
            "kg_only_correct": kg_only,
            "discordant": discordant,
            "exact_mcnemar_p": p_value,
            "note": "Candidate-level descriptive test; grouped bootstrap is the primary uncertainty analysis.",
        })
    return ci_rows, pair_rows, mcnemar_rows


def main() -> None:
    rows = read_rows(PREDICTIONS)
    if MACBERT_PREDICTIONS.exists():
        for row in read_rows(MACBERT_PREDICTIONS):
            row["task"] = "target_label"
            row["model"] = "macbert_base_finetuned"
            rows.append(row)
    grouped: dict[tuple[int, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(int(row["seed"]), row["task"])][row["model"]].append(row)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ci_rows: list[dict] = []
    pair_rows: list[dict] = []
    mcnemar_rows: list[dict] = []
    for (seed, task), rows_by_model in sorted(grouped.items()):
        keep = {
            model: model_rows
            for model, model_rows in rows_by_model.items()
            if model in {name for pair in MODEL_PAIRS for name in pair}
        }
        cis, pairs, mcnemars = group_bootstrap(task, keep, rng)
        for row in cis + pairs + mcnemars:
            row["seed"] = seed
        ci_rows.extend(cis)
        pair_rows.extend(pairs)
        mcnemar_rows.extend(mcnemars)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "group_bootstrap_confidence_intervals.csv", ci_rows)
    write_csv(OUT_DIR / "paired_group_bootstrap_differences.csv", pair_rows)
    write_csv(OUT_DIR / "mcnemar_descriptive.csv", mcnemar_rows)

    primary = [
        row for row in pair_rows
        if row["baseline"] == "char_tfidf_svm"
        and row["kg_model"] == "char_svm_plus_kg_constraints"
        and row["metric"] == "macro_f1"
    ]
    report = {
        "bootstrap_unit": "normalized sentence group",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "random_seed": BOOTSTRAP_SEED,
        "primary_macro_f1_differences": primary,
    }
    (OUT_DIR / "statistical_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
