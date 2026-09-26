#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
GOLD_DIR = ROOT / "annotation_qc/outputs/formal_qc_20260718"
MAIN_CSV = GOLD_DIR / "gold_main_by_filename_majority_annotator2_20260718.csv"
EVIDENCE_CSV = GOLD_DIR / "gold_evidence_by_filename_majority_annotator2_20260718.csv"
SLOT_CSV = GOLD_DIR / "gold_slot_span_by_filename_majority_annotator2_20260718.csv"
KG_DIR = ROOT / "grammar_kg/kg"
KG_BINDINGS = KG_DIR / "executable_rule_bindings_v1.json"
GRAMMAR_POINTS = KG_DIR / "grammar_points_v0_1.csv"
RULE_EVIDENCE = KG_DIR / "high_frequency_rule_evidence_v0_3.csv"
OUT_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_grouped_kg_v2"

SEEDS = [13, 42, 2027]
TEST_FOLDS = 5
DEV_FOLDS_WITHIN_REMAINDER = 8
C_GRID = [0.25, 1.0, 4.0]

TASKS = {
    "target_label": "最终主标签：是否为目标语法点",
    "extraction": "最终候选提取是否正确",
    "function_type": "最终功能/问题类型",
}

NEGATIVE_LABEL = {
    "target_label": "不是目标语法点",
    "extraction": "候选错误",
    "function_type": "表层触发但非目标",
}

POSITIVE_LABEL = {
    "target_label": "是目标语法点",
    "extraction": "正确",
    "function_type": "目标用例",
}

BOUNDARY_LABEL = {
    "extraction": "边界不完整",
    "function_type": "边界问题",
}

DECISIVE_LABELS = {
    "target_label": {"是目标语法点", "不是目标语法点"},
    "extraction": {"正确", "边界不完整", "边界过长", "候选错误"},
    "function_type": {"目标用例", "边界问题", "表层触发但非目标"},
}


def norm(value: object | None) -> str:
    return "" if value is None else str(value).strip()


def canonical_label(task: str, value: object | None) -> str:
    label = norm(value)
    if task == "function_type":
        return {
            "功能误判": "表层触发但非目标",
            "不确定": "不确定/需复核",
        }.get(label, label)
    return label


def normalize_sentence(text: str) -> str:
    text = unicodedata.normalize("NFKC", norm(text)).lower()
    text = text.translate(str.maketrans({
        "，": ",", "。": ".", "！": "!", "？": "?", "；": ";",
        "：": ":", "（": "(", "）": ")", "“": '"', "”": '"',
        "‘": "'", "’": "'",
    }))
    return re.sub(r"\s+", "", text)


def sentence_group_id(text: str) -> str:
    normalized = normalize_sentence(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_grouped(rows: list[dict[str, str]], seed: int) -> tuple[list, list, list, dict]:
    indices = np.arange(len(rows))
    strata = np.array([norm(row["语法家族"]) for row in rows])
    groups = np.array([sentence_group_id(row["句子"]) for row in rows])

    outer = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=seed)
    train_dev_idx, test_idx = next(outer.split(indices, strata, groups))

    inner_indices = indices[train_dev_idx]
    inner_strata = strata[train_dev_idx]
    inner_groups = groups[train_dev_idx]
    inner = StratifiedGroupKFold(
        n_splits=DEV_FOLDS_WITHIN_REMAINDER,
        shuffle=True,
        random_state=seed + 100_003,
    )
    train_rel, dev_rel = next(inner.split(inner_indices, inner_strata, inner_groups))
    train_idx = train_dev_idx[train_rel]
    dev_idx = train_dev_idx[dev_rel]

    split_groups = {
        "train": set(groups[train_idx]),
        "dev": set(groups[dev_idx]),
        "test": set(groups[test_idx]),
    }
    assert split_groups["train"].isdisjoint(split_groups["dev"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["dev"].isdisjoint(split_groups["test"])

    def take(idx: np.ndarray) -> list[dict[str, str]]:
        return [rows[int(i)] for i in idx]

    manifest = {
        "seed": seed,
        "strategy": "StratifiedGroupKFold by normalized sentence SHA1",
        "train_rows": len(train_idx),
        "dev_rows": len(dev_idx),
        "test_rows": len(test_idx),
        "train_groups": len(split_groups["train"]),
        "dev_groups": len(split_groups["dev"]),
        "test_groups": len(split_groups["test"]),
        "group_overlap_train_dev": 0,
        "group_overlap_train_test": 0,
        "group_overlap_dev_test": 0,
        "train_ids": [rows[int(i)]["编号"] for i in train_idx],
        "dev_ids": [rows[int(i)]["编号"] for i in dev_idx],
        "test_ids": [rows[int(i)]["编号"] for i in test_idx],
    }
    return take(train_idx), take(dev_idx), take(test_idx), manifest


class ExecutableGrammarKG:
    def __init__(self) -> None:
        self.bindings = json.loads(KG_BINDINGS.read_text(encoding="utf-8"))
        self.point_rows = read_csv(GRAMMAR_POINTS)
        self.rule_rows = read_csv(RULE_EVIDENCE)
        self.point_by_candidate = {row["candidate_label"]: row for row in self.point_rows}
        self.rule_by_id = {row["规则ID"]: row for row in self.rule_rows}
        self.meta_patterns = [re.compile(pattern) for pattern in self.bindings["global_meta_patterns"]]

    @staticmethod
    def _literal_markers(raw: str) -> list[str]:
        conceptual = {"数量", "时量", "趋向", "结果", "双宾", "给予", "兼语", "可能", "将字处置"}
        markers = [x.strip() for x in re.split(r"[；;]", raw) if x.strip()]
        return [x for x in markers if x not in conceptual and "..." not in x and "等" not in x]

    def evaluate(self, row: dict[str, str]) -> dict:
        sent = norm(row.get("句子"))
        cand = norm(row.get("候选片段"))
        point = norm(row.get("候选语法点"))
        family = norm(row.get("语法家族"))
        binding = self.bindings["rules"].get(point)
        point_meta = self.point_by_candidate.get(point, {})

        candidate_in_sentence = bool(cand and cand in sent)
        meta_trigger = any(pattern.search(cand) for pattern in self.meta_patterns)
        empty_candidate = not cand
        marker_present = False
        structure_present = False
        hard_lexical_negative = False
        fired_positive: list[str] = []
        fired_weak: list[str] = []
        fired_strong: list[str] = []
        reasons: list[str] = []

        if binding:
            marker_present = bool(re.search(binding["marker_pattern"], cand))
            structure_present = bool(re.search(binding["structure_pattern"], cand))
            hard_lexical_negative = any(
                re.search(pattern, cand) for pattern in binding.get("hard_negative_patterns", [])
            )
            if marker_present:
                fired_positive.extend(binding.get("positive_rule_ids", [])[:1])
                reasons.append("marker_present")
            else:
                fired_weak.extend(binding.get("weak_rule_ids", [])[:1])
                reasons.append("required_marker_missing")
            if structure_present:
                fired_positive.extend(binding.get("positive_rule_ids", [])[1:])
                reasons.append("surface_structure_present")
            else:
                fired_weak.extend(binding.get("weak_rule_ids", [])[:1])
                reasons.append("surface_structure_incomplete")
            if hard_lexical_negative:
                fired_strong.extend(binding.get("strong_rule_ids", [])[-1:])
                reasons.append("lexical_hard_negative")
        else:
            markers = self._literal_markers(norm(point_meta.get("markers")))
            marker_present = any(marker in cand for marker in markers) if markers else False
            structure_present = marker_present and len(cand) >= max(2, min((len(x) for x in markers), default=1) + 1)
            if marker_present:
                reasons.append("fallback_marker_present")
            else:
                reasons.append("no_executable_point_binding")

        if empty_candidate:
            fired_strong.append("GLOBAL-EMPTY-CANDIDATE")
            reasons.append("empty_candidate")
        if meta_trigger:
            strong_ids = binding.get("strong_rule_ids", []) if binding else []
            fired_strong.extend(strong_ids[:1] or ["GLOBAL-META-MENTION"])
            reasons.append("metalinguistic_or_ocr")
        if cand and not candidate_in_sentence:
            fired_weak.extend(binding.get("weak_rule_ids", [])[:1] if binding else ["GLOBAL-OFFSET-MISMATCH"])
            reasons.append("candidate_not_exact_substring")

        fired_positive = list(dict.fromkeys(fired_positive))
        fired_weak = list(dict.fromkeys(fired_weak))
        fired_strong = list(dict.fromkeys(fired_strong))
        score = len(fired_positive) - len(fired_weak) - 2 * len(fired_strong)
        accepted = bool(marker_present and structure_present and not fired_strong)
        weak_reject = bool(fired_weak and not fired_strong)
        strong_reject = bool(fired_strong)

        authority_rows = [self.rule_by_id[rid] for rid in fired_positive + fired_weak + fired_strong if rid in self.rule_by_id]
        source_ids: set[str] = set()
        page_anchors: set[str] = set()
        for authority in authority_rows:
            source_ids.update(x for x in authority["支撑文献ID"].split("；") if x)
            page_anchors.update(x for x in authority["规则说明页码"].split("；") if x)

        if strong_reject:
            diagnosis = "表层触发但非目标"
        elif weak_reject:
            diagnosis = "边界问题"
        elif accepted:
            diagnosis = "目标用例"
        else:
            diagnosis = "不确定/需复核"

        trace = [f"candidate:{point}", f"family:{family}"]
        trace.extend(f"rule:{rid}" for rid in fired_positive + fired_weak + fired_strong)
        trace.extend(f"source:{sid}" for sid in sorted(source_ids))

        return {
            "accepted": accepted,
            "weak_reject": weak_reject,
            "strong_reject": strong_reject,
            "marker_present": marker_present,
            "structure_present": structure_present,
            "hard_lexical_negative": hard_lexical_negative,
            "meta_trigger": meta_trigger,
            "candidate_in_sentence": candidate_in_sentence,
            "binding_covered": binding is not None,
            "evidence_score": score,
            "positive_rule_ids": fired_positive,
            "weak_rule_ids": fired_weak,
            "strong_rule_ids": fired_strong,
            "authority_rule_count": len(authority_rows),
            "authority_source_count": len(source_ids),
            "authority_page_anchor_count": len(page_anchors),
            "diagnosis": diagnosis,
            "reasons": reasons,
            "trace": trace,
        }

    def features(self, row: dict[str, str], mode: str = "full") -> dict[str, object]:
        result = self.evaluate(row)
        cand_len = len(norm(row.get("候选片段")))
        sent_len = max(1, len(norm(row.get("句子"))))
        features: dict[str, object] = {
            "family": norm(row.get("语法家族")),
            "grammar_point": norm(row.get("候选语法点")),
            "candidate_length_norm": min(cand_len, 100) / 100.0,
            "candidate_sentence_ratio": cand_len / sent_len,
            "binding_covered": int(result["binding_covered"]),
            "marker_present": int(result["marker_present"]),
            "structure_present": int(result["structure_present"]),
            "candidate_in_sentence": int(result["candidate_in_sentence"]),
        }
        if mode in {"constraints", "full"}:
            features.update({
                "kg_accepted": int(result["accepted"]),
                "kg_weak_reject": int(result["weak_reject"]),
                "kg_strong_reject": int(result["strong_reject"]),
                "kg_meta_trigger": int(result["meta_trigger"]),
                "kg_lexical_hard_negative": int(result["hard_lexical_negative"]),
                "kg_evidence_score_norm": max(-10, min(10, result["evidence_score"])) / 10.0,
                "kg_diagnosis": result["diagnosis"],
            })
            for reason in result["reasons"]:
                features[f"reason={reason}"] = 1
            for rule_id in result["positive_rule_ids"]:
                features[f"positive_rule={rule_id}"] = 1
            for rule_id in result["weak_rule_ids"]:
                features[f"weak_rule={rule_id}"] = 1
            for rule_id in result["strong_rule_ids"]:
                features[f"strong_rule={rule_id}"] = 1
        if mode == "full":
            features.update({
                "authority_rule_count_norm": min(result["authority_rule_count"], 10) / 10.0,
                "authority_source_count_norm": min(result["authority_source_count"], 10) / 10.0,
                "authority_page_anchor_count_norm": min(result["authority_page_anchor_count"], 20) / 20.0,
            })
        return features


KG = ExecutableGrammarKG()


def row_text(row: dict[str, str]) -> str:
    return " ".join([
        "句子=" + norm(row.get("句子")),
        "候选=" + norm(row.get("候选片段")),
        "候选语法点=" + norm(row.get("候选语法点")),
        "语法家族=" + norm(row.get("语法家族")),
    ])


def build_matrix(
    rows: list[dict[str, str]],
    text_vectorizer: TfidfVectorizer | None,
    kg_vectorizer: DictVectorizer | None,
    kg_mode: str | None,
    fit: bool,
):
    matrices = []
    if text_vectorizer is not None:
        texts = [row_text(row) for row in rows]
        matrices.append(text_vectorizer.fit_transform(texts) if fit else text_vectorizer.transform(texts))
    if kg_vectorizer is not None and kg_mode is not None:
        structured = [KG.features(row, kg_mode) for row in rows]
        matrices.append(kg_vectorizer.fit_transform(structured) if fit else kg_vectorizer.transform(structured))
    if not matrices:
        raise ValueError("At least one feature family is required")
    return matrices[0] if len(matrices) == 1 else hstack(matrices, format="csr")


@dataclass
class TrainedModel:
    name: str
    classifier: object
    text_vectorizer: TfidfVectorizer | None
    kg_vectorizer: DictVectorizer | None
    kg_mode: str | None
    selected_c: float

    def matrix(self, rows: list[dict[str, str]]):
        return build_matrix(rows, self.text_vectorizer, self.kg_vectorizer, self.kg_mode, fit=False)

    def predict(self, rows: list[dict[str, str]]) -> list[str]:
        return list(self.classifier.predict(self.matrix(rows)))

    def confidence(self, rows: list[dict[str, str]]) -> list[float | None]:
        matrix = self.matrix(rows)
        if hasattr(self.classifier, "predict_proba"):
            return list(np.max(self.classifier.predict_proba(matrix), axis=1))
        if hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(matrix)
            if scores.ndim == 1:
                return list(1.0 / (1.0 + np.exp(-np.abs(scores))))
            shifted = scores - np.max(scores, axis=1, keepdims=True)
            probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            return list(np.max(probs, axis=1))
        return [None] * len(rows)


def train_tuned_model(
    name: str,
    train: list[dict[str, str]],
    dev: list[dict[str, str]],
    task: str,
    label_col: str,
    model_type: str,
    use_text: bool,
    kg_mode: str | None,
) -> TrainedModel:
    text_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        min_df=2,
        sublinear_tf=True,
        max_features=100_000,
    ) if use_text else None
    kg_vectorizer = DictVectorizer(sparse=True) if kg_mode else None
    x_train = build_matrix(train, text_vectorizer, kg_vectorizer, kg_mode, fit=True)
    x_dev = build_matrix(dev, text_vectorizer, kg_vectorizer, kg_mode, fit=False)
    y_train = [canonical_label(task, row[label_col]) for row in train]
    y_dev = [canonical_label(task, row[label_col]) for row in dev]

    best_classifier = None
    best_score = -math.inf
    best_c = C_GRID[0]
    for c_value in C_GRID:
        if model_type == "lr":
            classifier = LogisticRegression(
                C=c_value,
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=0,
            )
        elif model_type == "svm":
            classifier = LinearSVC(
                C=c_value,
                class_weight="balanced",
                random_state=0,
                max_iter=20_000,
            )
        else:
            raise ValueError(model_type)
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_dev)
        score = f1_score(y_dev, prediction, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_classifier = classifier
            best_c = c_value
    assert best_classifier is not None
    return TrainedModel(name, best_classifier, text_vectorizer, kg_vectorizer, kg_mode, best_c)


def metric_bundle(task: str, y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    negative = NEGATIVE_LABEL[task]
    hard_indices = [i for i, label in enumerate(y_true) if label == negative]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hard_negative_accuracy": (
            sum(y_true[i] == y_pred[i] for i in hard_indices) / len(hard_indices)
            if hard_indices else None
        ),
        "per_label": {
            label: {"precision": p[i], "recall": r[i], "f1": f1[i], "support": int(support[i])}
            for i, label in enumerate(labels)
        },
    }


def grouped_majorities(
    train: list[dict[str, str]], task: str, label_col: str
) -> tuple[str, dict[str, str], dict[str, str]]:
    global_counts = Counter(canonical_label(task, row[label_col]) for row in train)
    by_family: dict[str, Counter] = defaultdict(Counter)
    by_point: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        label = canonical_label(task, row[label_col])
        by_family[row["语法家族"]][label] += 1
        by_point[row["候选语法点"]][label] += 1
    return (
        global_counts.most_common(1)[0][0],
        {family: counts.most_common(1)[0][0] for family, counts in by_family.items()},
        {point: counts.most_common(1)[0][0] for point, counts in by_point.items()},
    )


def kg_rule_predict(row: dict[str, str], task: str) -> str:
    result = KG.evaluate(row)
    if task == "target_label":
        return "是目标语法点" if result["accepted"] else "不是目标语法点"
    if task == "extraction":
        return "正确" if result["accepted"] else (
            "候选错误" if result["strong_reject"] else "边界不完整"
        )
    return "目标用例" if result["accepted"] else (
        "表层触发但非目标" if result["strong_reject"] else "边界问题"
    )


def hard_reject_prediction(row: dict[str, str], task: str, prediction: str) -> str:
    return NEGATIVE_LABEL[task] if KG.evaluate(row)["strong_reject"] else prediction


def signal_label(task: str, signal_type: str) -> str | None:
    if signal_type == "positive":
        return POSITIVE_LABEL[task]
    if signal_type == "strong":
        return NEGATIVE_LABEL[task]
    if signal_type == "weak":
        return BOUNDARY_LABEL.get(task)
    return None


def estimate_rule_reliability(
    train: list[dict[str, str]], task: str, label_col: str, seed: int
) -> tuple[dict[str, dict], list[dict]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    rule_type: dict[str, str] = {}
    for row in train:
        gold = canonical_label(task, row[label_col])
        evaluation = KG.evaluate(row)
        for signal_type, key in (
            ("positive", "positive_rule_ids"),
            ("weak", "weak_rule_ids"),
            ("strong", "strong_rule_ids"),
        ):
            predicted = signal_label(task, signal_type)
            if predicted is None:
                continue
            for rule_id in evaluation[key]:
                rule_type[rule_id] = signal_type
                counts[rule_id]["support"] += 1
                counts[rule_id]["correct"] += int(predicted == gold)

    reliability: dict[str, dict] = {}
    rows: list[dict] = []
    for rule_id, stats in sorted(counts.items()):
        support = stats["support"]
        correct = stats["correct"]
        posterior = (correct + 1.0) / (support + 2.0)
        if support >= 20 and posterior >= 0.80:
            grade = "A_KEEP"
            weight = posterior
        elif support >= 5 and posterior >= 0.65:
            grade = "B_KEEP_BUT_MONITOR"
            weight = 0.65 * posterior
        elif posterior < 0.50:
            grade = "D_HIGH_RISK"
            weight = 0.0
        else:
            grade = "INSUFFICIENT_EVIDENCE"
            weight = 0.15
        item = {
            "seed": seed,
            "task": task,
            "rule_id": rule_id,
            "signal_type": rule_type[rule_id],
            "support": support,
            "correct": correct,
            "posterior_reliability": posterior,
            "grade": grade,
            "effective_weight": weight,
        }
        reliability[rule_id] = item
        rows.append(item)
    return reliability, rows


def aligned_probabilities(model: TrainedModel, rows: list[dict[str, str]], classes: list[str]) -> np.ndarray:
    matrix = model.matrix(rows)
    probabilities = model.classifier.predict_proba(matrix)
    model_classes = list(model.classifier.classes_)
    aligned = np.full((len(rows), len(classes)), 1e-9, dtype=float)
    for index, label in enumerate(classes):
        if label in model_classes:
            aligned[:, index] = probabilities[:, model_classes.index(label)]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def rerank_probabilities(
    rows: list[dict[str, str]],
    local_probabilities: np.ndarray,
    classes: list[str],
    reliability: dict[str, dict],
    alpha: float,
    task: str,
) -> tuple[list[str], list[float]]:
    predictions: list[str] = []
    confidences: list[float] = []
    for row, local_probs in zip(rows, local_probabilities):
        evaluation = KG.evaluate(row)
        kg_scores = {label: 0.0 for label in classes}
        kg_counts = Counter()
        for signal_type, key in (
            ("positive", "positive_rule_ids"),
            ("weak", "weak_rule_ids"),
            ("strong", "strong_rule_ids"),
        ):
            label = signal_label(task, signal_type)
            if label not in kg_scores:
                continue
            for rule_id in evaluation[key]:
                weight = reliability.get(rule_id, {}).get("effective_weight", 0.15)
                kg_scores[label] += float(weight)
                kg_counts[label] += 1
        for label in classes:
            if kg_counts[label]:
                kg_scores[label] /= kg_counts[label]
        logits = np.log(np.maximum(local_probs, 1e-9)) + alpha * np.array(
            [kg_scores[label] for label in classes], dtype=float
        )
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        best = int(np.argmax(probs))
        predictions.append(classes[best])
        confidences.append(float(probs[best]))
    return predictions, confidences


def tune_reliability_reranker(
    local_model: TrainedModel,
    train: list[dict[str, str]],
    dev: list[dict[str, str]],
    test: list[dict[str, str]],
    task: str,
    label_col: str,
    seed: int,
) -> tuple[list[str], list[float], float, list[dict], list[dict]]:
    reliability, reliability_rows = estimate_rule_reliability(train, task, label_col, seed)
    classes = sorted(DECISIVE_LABELS[task])
    dev_probabilities = aligned_probabilities(local_model, dev, classes)
    test_probabilities = aligned_probabilities(local_model, test, classes)
    dev_gold = [canonical_label(task, row[label_col]) for row in dev]
    tuning_rows = []
    best_alpha = 0.0
    best_key = (-1.0, -1.0, -1.0)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        dev_pred, _ = rerank_probabilities(dev, dev_probabilities, classes, reliability, alpha, task)
        bundle = metric_bundle(task, dev_gold, dev_pred)
        key = (
            bundle["macro_f1"],
            bundle["hard_negative_accuracy"] or 0.0,
            bundle["accuracy"],
        )
        tuning_rows.append({
            "seed": seed,
            "task": task,
            "alpha": alpha,
            "dev_accuracy": bundle["accuracy"],
            "dev_macro_f1": bundle["macro_f1"],
            "dev_hard_negative_accuracy": bundle["hard_negative_accuracy"],
        })
        if key > best_key:
            best_key = key
            best_alpha = alpha
    test_pred, test_confidence = rerank_probabilities(
        test, test_probabilities, classes, reliability, best_alpha, task
    )
    return test_pred, test_confidence, best_alpha, reliability_rows, tuning_rows


def evaluate_predictions(
    rows: list[dict[str, str]],
    task: str,
    label_col: str,
    model_name: str,
    predictions: list[str],
    confidences: list[float | None] | None,
    seed: int,
) -> tuple[dict, list[dict], list[dict]]:
    y_true = [canonical_label(task, row[label_col]) for row in rows]
    metrics = metric_bundle(task, y_true, predictions)
    result = {
        "seed": seed,
        "task": task,
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "hard_negative_accuracy": metrics["hard_negative_accuracy"],
        "n": len(rows),
    }
    detailed: list[dict] = []
    family_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, (row, gold, pred) in enumerate(zip(rows, y_true, predictions)):
        kg_result = KG.evaluate(row)
        confidence = confidences[index] if confidences else None
        detailed.append({
            "seed": seed,
            "task": task,
            "model": model_name,
            "编号": row["编号"],
            "sentence_group": sentence_group_id(row["句子"]),
            "语法家族": row["语法家族"],
            "候选语法点": row["候选语法点"],
            "gold": gold,
            "prediction": pred,
            "correct": int(gold == pred),
            "confidence": confidence,
            "kg_accepted": int(kg_result["accepted"]),
            "kg_weak_reject": int(kg_result["weak_reject"]),
            "kg_strong_reject": int(kg_result["strong_reject"]),
            "kg_rule_trace": ";".join(kg_result["trace"]),
        })
        family_pairs[row["语法家族"]].append((gold, pred))
    family_results = []
    for family, pairs in sorted(family_pairs.items()):
        family_metric = metric_bundle(task, [x[0] for x in pairs], [x[1] for x in pairs])
        family_results.append({
            "seed": seed,
            "task": task,
            "model": model_name,
            "语法家族": family,
            "accuracy": family_metric["accuracy"],
            "macro_f1": family_metric["macro_f1"],
            "hard_negative_accuracy": family_metric["hard_negative_accuracy"],
            "n": len(pairs),
        })
    return result, detailed, family_results


def selective_metrics(prediction_rows: list[dict], review_budget: float = 0.10) -> dict:
    scored = []
    for row in prediction_rows:
        confidence = row["confidence"] if row["confidence"] not in {None, ""} else 0.5
        conflict = (
            row["kg_strong_reject"] == 1
            and row["prediction"] not in {"不是目标语法点", "候选错误", "表层触发但非目标"}
        )
        risk = (1.0 - float(confidence)) + (1.0 if conflict else 0.0) + (0.25 if row["kg_weak_reject"] else 0.0)
        scored.append((risk, row))
    scored.sort(key=lambda item: (-item[0], item[1]["编号"]))
    review_n = max(1, round(len(scored) * review_budget))
    reviewed = [row for _, row in scored[:review_n]]
    retained = [row for _, row in scored[review_n:]]
    all_errors = sum(1 - row["correct"] for _, row in scored)
    captured_errors = sum(1 - row["correct"] for row in reviewed)
    return {
        "review_budget": review_budget,
        "review_rate": len(reviewed) / len(scored),
        "retained_accuracy": sum(row["correct"] for row in retained) / len(retained) if retained else 0.0,
        "error_capture_rate": captured_errors / all_errors if all_errors else 0.0,
        "review_precision": captured_errors / len(reviewed) if reviewed else 0.0,
    }


def aggregate(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, items in sorted(grouped.items()):
        base = dict(zip(keys, group_key))
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) not in {None, ""}]
            if not values:
                continue
            output.append({
                **base,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "seeds": len(values),
            })
    return output


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(MAIN_CSV)
    results: list[dict] = []
    predictions: list[dict] = []
    family_results: list[dict] = []
    hyperparameters: list[dict] = []
    split_manifests: list[dict] = []
    selective_rows: list[dict] = []
    excluded_rows: list[dict] = []
    rule_reliability_rows: list[dict] = []
    reranker_tuning_rows: list[dict] = []

    for seed in SEEDS:
        train, dev, test, manifest = split_grouped(rows, seed)
        split_manifests.append(manifest)
        for task, label_col in TASKS.items():
            train_task = [
                row for row in train
                if canonical_label(task, row[label_col]) in DECISIVE_LABELS[task]
            ]
            dev_task = [
                row for row in dev
                if canonical_label(task, row[label_col]) in DECISIVE_LABELS[task]
            ]
            test_task = [
                row for row in test
                if canonical_label(task, row[label_col]) in DECISIVE_LABELS[task]
            ]
            excluded_rows.append({
                "seed": seed,
                "task": task,
                "train_excluded_as_review": len(train) - len(train_task),
                "dev_excluded_as_review": len(dev) - len(dev_task),
                "test_excluded_as_review": len(test) - len(test_task),
                "test_decisive_coverage": len(test_task) / len(test),
            })
            global_label, family_labels, point_labels = grouped_majorities(
                train_task, task, label_col
            )
            baseline_predictions = {
                "global_majority": [global_label] * len(test_task),
                "family_majority": [family_labels.get(row["语法家族"], global_label) for row in test_task],
                "point_majority": [point_labels.get(row["候选语法点"], global_label) for row in test_task],
                "executable_kg_rules": [kg_rule_predict(row, task) for row in test_task],
            }

            trained_models = [
                train_tuned_model("char_tfidf_lr", train_task, dev_task, task, label_col, "lr", True, None),
                train_tuned_model("char_tfidf_svm", train_task, dev_task, task, label_col, "svm", True, None),
                train_tuned_model("kg_only_lr", train_task, dev_task, task, label_col, "lr", False, "full"),
                train_tuned_model("char_lr_plus_kg_markers", train_task, dev_task, task, label_col, "lr", True, "markers"),
                train_tuned_model("char_lr_plus_kg_constraints", train_task, dev_task, task, label_col, "lr", True, "constraints"),
                train_tuned_model("char_lr_plus_kg_full", train_task, dev_task, task, label_col, "lr", True, "full"),
                train_tuned_model("char_svm_plus_kg_constraints", train_task, dev_task, task, label_col, "svm", True, "constraints"),
                train_tuned_model("char_svm_plus_kg_full", train_task, dev_task, task, label_col, "svm", True, "full"),
            ]
            trained_by_name = {model.name: model for model in trained_models}

            for model_name, preds in baseline_predictions.items():
                result, detail, family_detail = evaluate_predictions(
                    test_task, task, label_col, model_name, preds, None, seed
                )
                results.append(result)
                predictions.extend(detail)
                family_results.extend(family_detail)

            for model in trained_models:
                preds = model.predict(test_task)
                confidences = model.confidence(test_task)
                result, detail, family_detail = evaluate_predictions(
                    test_task, task, label_col, model.name, preds, confidences, seed
                )
                results.append(result)
                predictions.extend(detail)
                family_results.extend(family_detail)
                hyperparameters.append({
                    "seed": seed,
                    "task": task,
                    "model": model.name,
                    "selected_C": model.selected_c,
                })
                if model.name in {"char_tfidf_lr", "char_lr_plus_kg_full"}:
                    selective_rows.append({
                        "seed": seed,
                        "task": task,
                        "model": model.name,
                        **selective_metrics(detail, review_budget=0.10),
                    })
                if model.name == "char_lr_plus_kg_full":
                    rejected = [hard_reject_prediction(row, task, pred) for row, pred in zip(test_task, preds)]
                    reject_name = "char_lr_plus_kg_full_hard_reject"
                    reject_result, reject_detail, reject_family = evaluate_predictions(
                        test_task, task, label_col, reject_name, rejected, confidences, seed
                    )
                    results.append(reject_result)
                    predictions.extend(reject_detail)
                    family_results.extend(reject_family)

            reranked, rerank_confidence, selected_alpha, reliability_detail, tuning_detail = tune_reliability_reranker(
                trained_by_name["char_tfidf_lr"],
                train_task,
                dev_task,
                test_task,
                task,
                label_col,
                seed,
            )
            rerank_name = "reliability_weighted_kg_reranker"
            rerank_result, rerank_detail, rerank_family = evaluate_predictions(
                test_task,
                task,
                label_col,
                rerank_name,
                reranked,
                rerank_confidence,
                seed,
            )
            results.append(rerank_result)
            predictions.extend(rerank_detail)
            family_results.extend(rerank_family)
            hyperparameters.append({
                "seed": seed,
                "task": task,
                "model": rerank_name,
                "selected_alpha": selected_alpha,
            })
            rule_reliability_rows.extend(reliability_detail)
            reranker_tuning_rows.extend(tuning_detail)

    aggregate_results = aggregate(
        results,
        ["task", "model"],
        ["accuracy", "macro_f1", "hard_negative_accuracy"],
    )
    selective_aggregate = aggregate(
        selective_rows,
        ["task", "model"],
        ["retained_accuracy", "error_capture_rate", "review_precision"],
    )

    write_csv(OUT_DIR / "results_by_seed.csv", results)
    write_csv(OUT_DIR / "results_aggregate.csv", aggregate_results)
    write_csv(OUT_DIR / "predictions_with_rule_traces.csv", predictions)
    write_csv(OUT_DIR / "family_results_by_seed.csv", family_results)
    write_csv(OUT_DIR / "selected_hyperparameters.csv", hyperparameters)
    write_csv(OUT_DIR / "selective_review_by_seed.csv", selective_rows)
    write_csv(OUT_DIR / "selective_review_aggregate.csv", selective_aggregate)
    write_csv(OUT_DIR / "review_exclusion_by_seed.csv", excluded_rows)
    write_csv(OUT_DIR / "rule_reliability_by_seed.csv", rule_reliability_rows)
    write_csv(OUT_DIR / "reranker_alpha_tuning.csv", reranker_tuning_rows)
    (OUT_DIR / "split_manifest.json").write_text(
        json.dumps(split_manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "main_gold": str(MAIN_CSV),
        "kg_bindings": str(KG_BINDINGS),
        "grammar_points": str(GRAMMAR_POINTS),
        "rule_evidence": str(RULE_EVIDENCE),
        "seeds": SEEDS,
        "split": {
            "outer": f"StratifiedGroupKFold(n_splits={TEST_FOLDS})",
            "inner": f"StratifiedGroupKFold(n_splits={DEV_FOLDS_WITHIN_REMAINDER})",
            "group": "SHA1(NFKC+lower+whitespace-stripped sentence)",
        },
        "tfidf": {"analyzer": "char", "ngram_range": [1, 4], "min_df": 2, "max_features": 100000},
        "C_grid": C_GRID,
        "label_canonicalization": {"功能误判": "表层触发但非目标", "不确定": "不确定/需复核"},
        "decisive_label_policy": {
            task: sorted(labels) for task, labels in DECISIVE_LABELS.items()
        },
    }
    (OUT_DIR / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# Grouped Executable-KG Experiments v2",
        "",
        "All splits group rows by normalized sentence. The split manifest asserts zero group overlap.",
        "KG features are read from an executable rule-binding resource and every prediction row stores fired rule/source paths.",
        "",
        "| Task | Model | Metric | Mean | Std |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate_results:
        report.append(
            f"| {row['task']} | {row['model']} | {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} |"
        )
    (OUT_DIR / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(OUT_DIR),
        "result_rows": len(results),
        "aggregate_rows": len(aggregate_results),
        "prediction_rows": len(predictions),
        "zero_group_overlap": all(
            manifest["group_overlap_train_dev"] == 0
            and manifest["group_overlap_train_test"] == 0
            and manifest["group_overlap_dev_test"] == 0
            for manifest in split_manifests
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
