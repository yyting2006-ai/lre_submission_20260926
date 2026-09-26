#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from run_grouped_kg_experiments import (
    DECISIVE_LABELS,
    MAIN_CSV,
    TASKS,
    canonical_label,
    read_csv,
    split_grouped,
    train_tuned_model,
)
from run_macbert_grouped_baseline import Collator, GrammarDataset, MODEL_NAME, choose_device


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
OUT_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_efficiency_v1"
MACBERT_RESULTS = ROOT / "asgd_fast_experiments/outputs/exp_20260718_macbert_grouped_v1/results_by_seed.csv"
SEED = 42


def mib(value: int) -> float:
    return value / (1024 * 1024)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(MAIN_CSV)
    train, dev, test, _ = split_grouped(rows, SEED)
    label_col = TASKS["target_label"]
    decisive = DECISIVE_LABELS["target_label"]
    train = [row for row in train if canonical_label("target_label", row[label_col]) in decisive]
    dev = [row for row in dev if canonical_label("target_label", row[label_col]) in decisive]
    test = [row for row in test if canonical_label("target_label", row[label_col]) in decisive]

    started = time.perf_counter()
    svm = train_tuned_model(
        "char_svm_plus_kg_constraints", train, dev, "target_label", label_col, "svm", True, "constraints"
    )
    svm_train_seconds = time.perf_counter() - started
    svm.predict(test[:32])
    svm_times = []
    for _ in range(20):
        started = time.perf_counter()
        svm.predict(test)
        svm_times.append(time.perf_counter() - started)
    with tempfile.NamedTemporaryFile(suffix=".joblib") as handle:
        joblib.dump(svm, handle.name, compress=3)
        svm_size = Path(handle.name).stat().st_size

    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, local_files_only=True
    ).to(device)
    model.eval()
    loader = DataLoader(
        GrammarDataset(test, tokenizer, 128),
        batch_size=32,
        shuffle=False,
        collate_fn=Collator(tokenizer),
        num_workers=0,
    )

    def macbert_pass() -> None:
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels")
                batch = {key: value.to(device) for key, value in batch.items()}
                model(**batch)
        if device.type == "mps":
            torch.mps.synchronize()

    macbert_pass()
    macbert_times = []
    for _ in range(3):
        started = time.perf_counter()
        macbert_pass()
        macbert_times.append(time.perf_counter() - started)

    cache_root = Path.home() / ".cache/huggingface/hub/models--hfl--chinese-macbert-base/blobs"
    weight_files = [path for path in cache_root.glob("*") if path.is_file() and not path.name.endswith(".incomplete")]
    macbert_disk_size = max((path.stat().st_size for path in weight_files), default=0)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    macbert_train_rows = read_csv(MACBERT_RESULTS)
    macbert_train_seconds = statistics.mean(float(row["duration_seconds"]) for row in macbert_train_rows)

    report = {
        "benchmark_seed": SEED,
        "test_candidates": len(test),
        "device": str(device),
        "lightweight_kg_svm": {
            "training_seconds_including_C_tuning": svm_train_seconds,
            "median_inference_seconds": statistics.median(svm_times),
            "candidates_per_second": len(test) / statistics.median(svm_times),
            "serialized_size_mib": mib(svm_size),
            "text_feature_count": len(svm.text_vectorizer.vocabulary_) if svm.text_vectorizer else 0,
            "kg_feature_count": len(svm.kg_vectorizer.vocabulary_) if svm.kg_vectorizer else 0,
            "selected_C": svm.selected_c,
        },
        "macbert_base": {
            "mean_finetuning_seconds_over_three_seeds": macbert_train_seconds,
            "median_inference_seconds": statistics.median(macbert_times),
            "candidates_per_second": len(test) / statistics.median(macbert_times),
            "parameter_count": parameter_count,
            "largest_cached_weight_file_mib": mib(macbert_disk_size),
            "batch_size": 32,
            "max_length": 128,
            "latency_note": "Random classification head; architecture-level inference cost only. Model load time excluded.",
        },
    }
    (OUT_DIR / "efficiency_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
