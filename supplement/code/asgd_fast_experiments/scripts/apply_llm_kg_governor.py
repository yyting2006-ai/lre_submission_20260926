#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from run_grouped_kg_experiments import KG, MAIN_CSV, read_csv, sentence_group_id, write_csv


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
LLM_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_openai_gpt56terra_strong_seed42_v1"
OUT_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_llm_kg_governor_v1"
DEV_RELEASE = LLM_DIR / "release_safe/dev_point_retrieved_predictions_release.csv"
TEST_RELEASE = LLM_DIR / "release_safe/selected_test_predictions_release.csv"

POSITIVE = "是目标语法点"
NEGATIVE = "不是目标语法点"
ACCEPT_WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
WEAK_WEIGHTS = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0]
STRONG_WEIGHTS = [-6.0, -4.0, -3.0, -2.0, -1.0, 0.0]
LOCAL_THRESHOLDS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def load_release(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        if int(row.get("valid_output") or 0):
            output[row["编号"]] = row
    return output


def target_logit(row: dict[str, Any]) -> float:
    confidence = min(0.9999, max(0.0001, float(row["confidence"])))
    probability = confidence if row["prediction"] == POSITIVE else 1.0 - confidence
    return math.log(probability / (1.0 - probability))


def state_features(row: dict[str, str]) -> tuple[int, int, int, list[str]]:
    result = KG.evaluate(row)
    return (
        int(result["accepted"]),
        int(result["weak_reject"]),
        int(result["strong_reject"]),
        list(result["trace"]),
    )


def metric(gold: list[str], prediction: list[str]) -> dict[str, float]:
    negative = [index for index, label in enumerate(gold) if label == NEGATIVE]
    return {
        "accuracy": float(accuracy_score(gold, prediction)),
        "macro_f1": float(f1_score(gold, prediction, average="macro", zero_division=0)),
        "hard_negative_accuracy": float(
            sum(prediction[index] == NEGATIVE for index in negative) / len(negative)
        )
        if negative
        else 0.0,
    }


def predict(
    llm_rows: dict[str, dict[str, Any]],
    source_rows: dict[str, dict[str, str]],
    weights: tuple[float, float, float, float],
) -> tuple[list[str], list[float]]:
    accept_weight, weak_weight, strong_weight, threshold = weights
    predictions: list[str] = []
    scores: list[float] = []
    for row_id, llm in llm_rows.items():
        accepted, weak, strong, _ = state_features(source_rows[row_id])
        score = (
            target_logit(llm)
            + accept_weight * accepted
            + weak_weight * weak
            + strong_weight * strong
        )
        predictions.append(POSITIVE if score >= threshold else NEGATIVE)
        scores.append(score)
    return predictions, scores


def select_threshold_only(dev: dict[str, dict[str, Any]]) -> tuple[float, dict[str, float]]:
    gold = [row["gold"] for row in dev.values()]
    best: tuple[tuple[float, float, float], float, dict[str, float]] | None = None
    for threshold in LOCAL_THRESHOLDS:
        prediction = [POSITIVE if target_logit(row) >= threshold else NEGATIVE for row in dev.values()]
        values = metric(gold, prediction)
        ranking = (values["macro_f1"], values["accuracy"], -abs(threshold))
        if best is None or ranking > best[0]:
            best = (ranking, threshold, values)
    assert best is not None
    return best[1], best[2]


def select_governor(
    dev: dict[str, dict[str, Any]], source_rows: dict[str, dict[str, str]]
) -> tuple[tuple[float, float, float, float], dict[str, float]]:
    gold = [row["gold"] for row in dev.values()]
    best: tuple[
        tuple[float, float, float, tuple[float, float, float, float]],
        tuple[float, float, float, float],
        dict[str, float],
    ] | None = None
    for weights in itertools.product(
        ACCEPT_WEIGHTS, WEAK_WEIGHTS, STRONG_WEIGHTS, LOCAL_THRESHOLDS
    ):
        prediction, _ = predict(dev, source_rows, weights)
        values = metric(gold, prediction)
        ranking = (
            values["macro_f1"],
            values["accuracy"],
            -sum(abs(value) for value in weights),
            tuple(-value for value in weights),
        )
        if best is None or ranking > best[0]:
            best = (ranking, weights, values)
    assert best is not None
    return best[1], best[2]


def evaluate_split(
    name: str,
    llm_rows: dict[str, dict[str, Any]],
    source_rows: dict[str, dict[str, str]],
    threshold: float,
    weights: tuple[float, float, float, float],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    gold = [row["gold"] for row in llm_rows.values()]
    raw_prediction = [row["prediction"] for row in llm_rows.values()]
    calibrated_prediction = [
        POSITIVE if target_logit(row) >= threshold else NEGATIVE for row in llm_rows.values()
    ]
    governed_prediction, scores = predict(llm_rows, source_rows, weights)
    metrics = {
        "raw_llm": metric(gold, raw_prediction),
        "local_threshold_only": metric(gold, calibrated_prediction),
        "llm_plus_exec_kg_governor": metric(gold, governed_prediction),
    }
    output: list[dict[str, Any]] = []
    for (row_id, llm), calibrated, governed, score in zip(
        llm_rows.items(), calibrated_prediction, governed_prediction, scores
    ):
        source = source_rows[row_id]
        accepted, weak, strong, trace = state_features(source)
        output.append(
            {
                "split": name,
                "编号": row_id,
                "sentence_group": sentence_group_id(source["句子"]),
                "gold": llm["gold"],
                "llm_prediction": llm["prediction"],
                "llm_confidence": llm["confidence"],
                "threshold_only_prediction": calibrated,
                "governed_prediction": governed,
                "governed_score": score,
                "kg_accepted": accepted,
                "kg_weak_reject": weak,
                "kg_strong_reject": strong,
                "kg_rule_ids": ";".join(item for item in trace if item.startswith("rule:")),
                "governed_correct": int(governed == llm["gold"]),
            }
        )
    return metrics, output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = {row["编号"]: row for row in read_csv(MAIN_CSV)}
    dev = load_release(DEV_RELEASE)
    test = load_release(TEST_RELEASE)
    threshold, threshold_dev = select_threshold_only(dev)
    weights, governor_dev = select_governor(dev, source_rows)
    dev_metrics, dev_rows = evaluate_split("dev", dev, source_rows, threshold, weights)
    test_metrics, test_rows = evaluate_split("test", test, source_rows, threshold, weights)
    report = {
        "status": "post_hoc_exploratory",
        "selection_split": "development",
        "selection_metric": "macro_f1",
        "confidence_transform": "self-reported target probability clipped to [0.0001, 0.9999], then logit",
        "threshold_grid": LOCAL_THRESHOLDS,
        "accept_weight_grid": ACCEPT_WEIGHTS,
        "weak_weight_grid": WEAK_WEIGHTS,
        "strong_weight_grid": STRONG_WEIGHTS,
        "selected_threshold_only": threshold,
        "selected_governor_weights": {
            "accept": weights[0],
            "weak_reject": weights[1],
            "strong_reject": weights[2],
            "threshold": weights[3],
        },
        "selection_scores": {
            "threshold_only": threshold_dev,
            "governor": governor_dev,
        },
        "dev": dev_metrics,
        "test": test_metrics,
    }
    write_csv(OUT_DIR / "dev_predictions_release.csv", dev_rows)
    write_csv(OUT_DIR / "test_predictions_release.csv", test_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
