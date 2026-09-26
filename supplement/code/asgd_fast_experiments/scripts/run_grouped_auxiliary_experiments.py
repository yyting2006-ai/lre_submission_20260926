#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from run_grouped_kg_experiments import (
    C_GRID,
    KG,
    read_csv,
    split_grouped,
    train_tuned_model,
    write_csv,
)


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
GOLD_DIR = ROOT / "annotation_qc/outputs/formal_qc_20260718"
EVIDENCE_CSV = GOLD_DIR / "gold_evidence_by_filename_majority_annotator2_20260718.csv"
SLOT_CSV = GOLD_DIR / "gold_slot_span_by_filename_majority_annotator2_20260718.csv"
OUT_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_grouped_auxiliary_v1"
SEEDS = [13, 42, 2027]
EVIDENCE_LABEL = "最终结构槽位是否支持"
SLOT_STATUS_LABEL = "最终槽位状态"
SLOT_SPAN_LABEL = "最终槽位片段（从句子中复制；条件必填）"


def norm(value: object | None) -> str:
    return "" if value is None else str(value).strip()


def evaluate_labels(
    y_true: list[str], y_pred: list[str], labels: list[str], negative_label: str | None = None
) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    if negative_label is not None:
        indices = [index for index, label in enumerate(y_true) if label == negative_label]
        result["negative_accuracy"] = (
            sum(y_true[index] == y_pred[index] for index in indices) / len(indices)
            if indices else 0.0
        )
    return result


def majority(train: list[dict[str, str]], label_col: str) -> str:
    return Counter(norm(row[label_col]) for row in train).most_common(1)[0][0]


def family_majorities(train: list[dict[str, str]], label_col: str) -> tuple[str, dict[str, str]]:
    global_label = majority(train, label_col)
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        counts[row["语法家族"]][norm(row[label_col])] += 1
    return global_label, {family: labels.most_common(1)[0][0] for family, labels in counts.items()}


def evidence_rule_prediction(row: dict[str, str]) -> str:
    result = KG.evaluate(row)
    if result["strong_reject"]:
        return "不支持"
    if result["accepted"] and result["structure_present"]:
        return "支持"
    return "部分支持"


def run_evidence() -> tuple[list[dict], list[dict], list[dict]]:
    all_rows = [row for row in read_csv(EVIDENCE_CSV) if norm(row[EVIDENCE_LABEL]) != "不确定"]
    labels = ["不支持", "部分支持", "支持"]
    result_rows: list[dict] = []
    prediction_rows: list[dict] = []
    split_rows: list[dict] = []
    for seed in SEEDS:
        train, dev, test, manifest = split_grouped(all_rows, seed)
        global_label, family_label = family_majorities(train, EVIDENCE_LABEL)
        models = [
            train_tuned_model("char_tfidf_lr", train, dev, "evidence_support", EVIDENCE_LABEL, "lr", True, None),
            train_tuned_model("char_tfidf_svm", train, dev, "evidence_support", EVIDENCE_LABEL, "svm", True, None),
            train_tuned_model("char_lr_plus_kg", train, dev, "evidence_support", EVIDENCE_LABEL, "lr", True, "constraints"),
            train_tuned_model("char_svm_plus_kg", train, dev, "evidence_support", EVIDENCE_LABEL, "svm", True, "constraints"),
        ]
        predictions: dict[str, list[str]] = {
            "global_majority": [global_label] * len(test),
            "family_majority": [family_label.get(row["语法家族"], global_label) for row in test],
            "executable_kg_rules": [evidence_rule_prediction(row) for row in test],
        }
        selected_c = {model.name: model.selected_c for model in models}
        for model in models:
            predictions[model.name] = model.predict(test)

        gold = [norm(row[EVIDENCE_LABEL]) for row in test]
        for model_name, pred in predictions.items():
            metrics = evaluate_labels(gold, pred, labels, negative_label="不支持")
            result_rows.append({
                "seed": seed,
                "task": "structure_evidence_support",
                "model": model_name,
                **metrics,
                "selected_C": selected_c.get(model_name, ""),
                "train_n": len(train),
                "dev_n": len(dev),
                "test_n": len(test),
                "zero_group_overlap": int(
                    manifest["group_overlap_train_dev"] == 0
                    and manifest["group_overlap_train_test"] == 0
                    and manifest["group_overlap_dev_test"] == 0
                ),
            })
            for row, gold_label, prediction in zip(test, gold, pred):
                runtime = KG.evaluate(row)
                prediction_rows.append({
                    "seed": seed,
                    "task": "structure_evidence_support",
                    "model": model_name,
                    "编号": row["编号"],
                    "gold": gold_label,
                    "prediction": prediction,
                    "correct": int(gold_label == prediction),
                    "kg_rule_trace": ";".join(runtime["trace"]),
                })
        split_rows.append({"seed": seed, "task": "structure_evidence_support", **manifest})
    return result_rows, prediction_rows, split_rows


def slot_text(row: dict[str, str]) -> str:
    return " ".join([
        "句子=" + norm(row.get("句子")),
        "候选=" + norm(row.get("候选片段")),
        "语法点=" + norm(row.get("候选语法点")),
        "家族=" + norm(row.get("语法家族")),
        "槽位ID=" + norm(row.get("槽位ID")),
        "槽位名=" + norm(row.get("槽位名称")),
        "标注要求=" + norm(row.get("标注要求")),
    ])


@dataclass
class SlotModel:
    name: str
    vectorizer: TfidfVectorizer
    classifier: object
    selected_c: float

    def predict(self, rows: list[dict[str, str]]) -> list[str]:
        return list(self.classifier.predict(self.vectorizer.transform([slot_text(row) for row in rows])))


def train_slot_model(
    train: list[dict[str, str]], dev: list[dict[str, str]], model_type: str
) -> SlotModel:
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 4), min_df=2, sublinear_tf=True, max_features=100_000
    )
    x_train = vectorizer.fit_transform([slot_text(row) for row in train])
    x_dev = vectorizer.transform([slot_text(row) for row in dev])
    y_train = [norm(row[SLOT_STATUS_LABEL]) for row in train]
    y_dev = [norm(row[SLOT_STATUS_LABEL]) for row in dev]
    best_model = None
    best_score = -math.inf
    best_c = C_GRID[0]
    for c_value in C_GRID:
        if model_type == "lr":
            classifier = LogisticRegression(
                C=c_value, max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=0
            )
        else:
            classifier = LinearSVC(
                C=c_value, max_iter=20_000, class_weight="balanced", random_state=0
            )
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_dev)
        score = f1_score(y_dev, prediction, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_c = c_value
            best_model = classifier
    assert best_model is not None
    return SlotModel(f"slot_char_tfidf_{model_type}", vectorizer, best_model, best_c)


def first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def grounded(text: str, sentence: str) -> str:
    value = text.strip(" ，。！？；,:;!?()（）")
    return value if value and value in sentence else ""


def slot_span_rule(row: dict[str, str]) -> str:
    sentence = norm(row["句子"])
    candidate = norm(row["候选片段"])
    family = norm(row["语法家族"])
    slot = norm(row["槽位ID"])
    value = ""

    if slot == "marker":
        value = "把" if "把" in sentence else ""
    elif slot == "passive_marker":
        value = first_match(r"被|叫|让|给|为.+所", candidate or sentence)
    elif slot == "compare_marker":
        value = first_match(r"不如|没有|比|跟|和|一样", candidate or sentence)
    elif slot == "complement_marker":
        value = first_match(r"得", candidate or sentence)
    elif slot == "prep_marker":
        value = first_match(r"关于|对于|从|往|向|朝|对|给|为|在|以|由", candidate or sentence)
    elif slot == "connective_1":
        value = first_match(r"之所以|因为|虽然|如果|只要|既然|不但|即使|尽管|由于|无论", sentence)
    elif slot == "connective_2":
        value = first_match(r"所以|但是|就|那么|而且|也|因此|从而|进而|还是|都", sentence)
    elif slot == "clause_1":
        value = re.split(r"[，；。！？]", sentence, maxsplit=1)[0]
    elif slot == "clause_2":
        parts = re.split(r"[，；]", sentence, maxsplit=1)
        value = parts[1] if len(parts) == 2 else ""
    elif slot == "time_or_location_expression":
        value = candidate
    elif family == "时间/处所功能结构" and slot in {"predicate", "function_relation"}:
        after = sentence.split(candidate, 1)[1] if candidate and candidate in sentence else ""
        value = re.split(r"[，。！？；]", after.lstrip("，, "), maxsplit=1)[0]
    elif family == "介词/框架结构":
        marker_match = re.search(r"关于|对于|从|往|向|朝|对|给|为|在|以|由", candidate)
        if marker_match and slot == "prep_object":
            tail = candidate[marker_match.end():]
            value = re.split(r"[，。！？；]", tail, maxsplit=1)[0]
        elif slot in {"predicate", "function_scope"}:
            value = candidate
    elif family == "补语结构":
        if "得" in candidate:
            left, right = candidate.split("得", 1)
            value = left[-4:] if slot == "predicate" else right if slot == "complement" else ""
        else:
            match = re.search(
                r"(?P<p>[一-龥]{1,4})(?P<c>起来|下来|出来|进去|过来|过去|进|出|上|下|一遍|一次|一下|[一二三四五六七八九十半两0-9]+(?:次|遍|年|月|天|小时|分钟))",
                candidate,
            )
            if match:
                value = match.group("p") if slot == "predicate" else match.group("c") if slot in {"complement", "object_or_measure"} else ""
    elif family == "比较构式":
        match = re.search(
            r"(?P<a>[^比没有不如跟和，。！？；]+)(?P<m>比|没有|不如|跟|和)(?P<b>[^，。！？；]+)",
            candidate,
        )
        if match:
            if slot == "compare_subject":
                value = match.group("a")
            elif slot == "standard":
                value = match.group("b")
            elif slot == "property":
                value = candidate[match.end("b"):]
    elif family == "处置/受事重组构式" and "把" in sentence:
        before, after = sentence.split("把", 1)
        if slot == "agent":
            value = re.split(r"[，。！？；]", before)[-1]
        elif slot == "patient":
            value = re.split(r"[，。！？；了着过]", after, maxsplit=1)[0]
        elif slot in {"predicate", "result_component", "affected_result"}:
            value = re.split(r"[，。！？；]", after, maxsplit=1)[0]
    elif family == "被动/受事凸显构式":
        match = re.search(r"被|叫|让|给", sentence)
        if match:
            before, after = sentence[:match.start()], sentence[match.end():]
            if slot == "patient_subject":
                value = re.split(r"[，。！？；]", before)[-1]
            elif slot in {"agent", "predicate"}:
                value = re.split(r"[，。！？；]", after, maxsplit=1)[0]
    elif family == "多谓词/论元结构":
        chunks = [part for part in re.split(r"[，。！？；、 ]", candidate) if part]
        if slot == "predicate_1" and chunks:
            value = chunks[0]
        elif slot == "predicate_2" and len(chunks) > 1:
            value = chunks[1]
        elif slot in {"object_or_recipient", "shared_argument", "predicate"}:
            value = candidate

    return grounded(value, sentence)


def span_scores(rows: list[dict[str, str]], predicted: list[str], relaxed: bool) -> dict[str, float]:
    tp = fp = fn = 0
    grounded_count = 0
    predicted_count = 0
    for row, pred in zip(rows, predicted):
        gold = norm(row[SLOT_SPAN_LABEL]) if norm(row[SLOT_STATUS_LABEL]) == "已标出" else ""
        if pred:
            predicted_count += 1
            grounded_count += int(pred in norm(row["句子"]))
        match = bool(pred and gold and (pred == gold if not relaxed else (pred in gold or gold in pred)))
        tp += int(match)
        fp += int(bool(pred) and not match)
        fn += int(bool(gold) and not match)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "grounding_rate": grounded_count / predicted_count if predicted_count else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run_slots() -> tuple[list[dict], list[dict], list[dict]]:
    all_rows = read_csv(SLOT_CSV)
    status_labels = ["不适用", "候选错误", "句中省略", "已标出"]
    result_rows: list[dict] = []
    prediction_rows: list[dict] = []
    split_rows: list[dict] = []
    for seed in SEEDS:
        train, dev, test, manifest = split_grouped(all_rows, seed)
        global_label, family_label = family_majorities(train, SLOT_STATUS_LABEL)
        models = [train_slot_model(train, dev, "lr"), train_slot_model(train, dev, "svm")]
        status_predictions: dict[str, list[str]] = {
            "global_majority": [global_label] * len(test),
            "family_majority": [family_label.get(row["语法家族"], global_label) for row in test],
        }
        for model in models:
            status_predictions[model.name] = model.predict(test)
        gold_status = [norm(row[SLOT_STATUS_LABEL]) for row in test]
        for model_name, prediction in status_predictions.items():
            result_rows.append({
                "seed": seed,
                "task": "slot_status",
                "model": model_name,
                **evaluate_labels(gold_status, prediction, status_labels),
                "selected_C": next((model.selected_c for model in models if model.name == model_name), ""),
                "train_n": len(train),
                "dev_n": len(dev),
                "test_n": len(test),
                "zero_group_overlap": int(
                    manifest["group_overlap_train_dev"] == 0
                    and manifest["group_overlap_train_test"] == 0
                    and manifest["group_overlap_dev_test"] == 0
                ),
            })
            for row, gold, pred in zip(test, gold_status, prediction):
                prediction_rows.append({
                    "seed": seed,
                    "task": "slot_status",
                    "model": model_name,
                    "编号": row["编号"],
                    "槽位ID": row["槽位ID"],
                    "gold": gold,
                    "prediction": pred,
                    "correct": int(gold == pred),
                })

        span_prediction = [slot_span_rule(row) for row in test]
        exact = span_scores(test, span_prediction, relaxed=False)
        relaxed = span_scores(test, span_prediction, relaxed=True)
        result_rows.append({
            "seed": seed,
            "task": "slot_span",
            "model": "executable_slot_rules",
            "exact_span_f1": exact["f1"],
            "relaxed_span_f1": relaxed["f1"],
            "exact_precision": exact["precision"],
            "exact_recall": exact["recall"],
            "relaxed_precision": relaxed["precision"],
            "relaxed_recall": relaxed["recall"],
            "cue_grounding": exact["grounding_rate"],
            "test_n": len(test),
            "zero_group_overlap": int(
                manifest["group_overlap_train_dev"] == 0
                and manifest["group_overlap_train_test"] == 0
                and manifest["group_overlap_dev_test"] == 0
            ),
        })
        for row, pred in zip(test, span_prediction):
            prediction_rows.append({
                "seed": seed,
                "task": "slot_span",
                "model": "executable_slot_rules",
                "编号": row["编号"],
                "槽位ID": row["槽位ID"],
                "gold_status": row[SLOT_STATUS_LABEL],
                "gold_span": row[SLOT_SPAN_LABEL],
                "prediction_span": pred,
                "grounded": int(not pred or pred in row["句子"]),
            })
        split_rows.append({"seed": seed, "task": "slot", **manifest})
    return result_rows, prediction_rows, split_rows


def aggregate(rows: list[dict]) -> list[dict]:
    metrics = [
        "accuracy", "macro_f1", "negative_accuracy", "exact_span_f1", "relaxed_span_f1",
        "exact_precision", "exact_recall", "relaxed_precision", "relaxed_recall", "cue_grounding",
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["model"])].append(row)
    output: list[dict] = []
    for (task, model), items in sorted(grouped.items()):
        for metric in metrics:
            values = [float(row[metric]) for row in items if row.get(metric, "") not in {"", None}]
            if values:
                output.append({
                    "task": task,
                    "model": model,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "seeds": len(values),
                })
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence_results, evidence_predictions, evidence_splits = run_evidence()
    slot_results, slot_predictions, slot_splits = run_slots()
    results = evidence_results + slot_results
    predictions = evidence_predictions + slot_predictions
    aggregate_rows = aggregate(results)
    write_csv(OUT_DIR / "results_by_seed.csv", results)
    write_csv(OUT_DIR / "results_aggregate.csv", aggregate_rows)
    write_csv(OUT_DIR / "predictions.csv", predictions)
    (OUT_DIR / "split_manifest.json").write_text(
        json.dumps(evidence_splits + slot_splits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "out_dir": str(OUT_DIR),
        "result_rows": len(results),
        "prediction_rows": len(predictions),
        "all_splits_zero_group_overlap": all(int(row["zero_group_overlap"]) == 1 for row in results),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
