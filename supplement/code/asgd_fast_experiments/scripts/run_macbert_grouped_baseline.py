#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from run_grouped_kg_experiments import (
    DECISIVE_LABELS,
    KG,
    MAIN_CSV,
    canonical_label,
    read_csv,
    sentence_group_id,
    split_grouped,
    write_csv,
)


ROOT = Path(os.environ.get("ASGD_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
OUT_DIR = ROOT / "asgd_fast_experiments/outputs/exp_20260718_macbert_grouped_v1"
MODEL_NAME = "hfl/chinese-macbert-base"
LABEL_COL = "最终主标签：是否为目标语法点"
LABEL_TO_ID = {"不是目标语法点": 0, "是目标语法点": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2027])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_with_retry(device: torch.device, attempts: int = 5):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=2,
                id2label=ID_TO_LABEL,
                label2id=LABEL_TO_ID,
            ).to(device)
        except Exception as error:  # Network interruptions are common on first download.
            last_error = error
            if attempt == attempts:
                break
            wait_seconds = 15 * attempt
            print(
                json.dumps(
                    {
                        "event": "model_load_retry",
                        "attempt": attempt,
                        "wait_seconds": wait_seconds,
                        "error": type(error).__name__,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


def format_input(row: dict[str, str]) -> str:
    return (
        f"语法点：{row['候选语法点']}。"
        f"语法家族：{row['语法家族']}。"
        f"候选片段：{row['候选片段']}。"
        f"句子：{row['句子']}"
    )


class GrammarDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            format_input(row),
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = LABEL_TO_ID[canonical_label("target_label", row[LABEL_COL])]
        return {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}


class Collator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        labels = torch.stack([item.pop("labels") for item in batch])
        padded = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    gold: list[int] = []
    pred: list[int] = []
    confidence: list[float] = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits.detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        gold.extend(labels.tolist())
        pred.extend(torch.argmax(probs, dim=-1).tolist())
        confidence.extend(torch.max(probs, dim=-1).values.tolist())
    return gold, pred, confidence


def metrics(gold: list[int], pred: list[int]) -> dict[str, float]:
    negative_indices = [index for index, label in enumerate(gold) if label == 0]
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
        "hard_negative_accuracy": (
            sum(gold[index] == pred[index] for index in negative_indices) / len(negative_indices)
            if negative_indices else 0.0
        ),
    }


def class_weights(rows: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    counts = np.bincount(
        [LABEL_TO_ID[canonical_label("target_label", row[LABEL_COL])] for row in rows],
        minlength=2,
    )
    weights = len(rows) / (2.0 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_seed(args: argparse.Namespace, seed: int, tokenizer, device: torch.device) -> tuple[dict, list[dict], list[dict]]:
    set_seed(seed)
    all_rows = read_csv(MAIN_CSV)
    train, dev, test, manifest = split_grouped(all_rows, seed)
    train = [row for row in train if canonical_label("target_label", row[LABEL_COL]) in DECISIVE_LABELS["target_label"]]
    dev = [row for row in dev if canonical_label("target_label", row[LABEL_COL]) in DECISIVE_LABELS["target_label"]]
    test = [row for row in test if canonical_label("target_label", row[LABEL_COL]) in DECISIVE_LABELS["target_label"]]

    collator = Collator(tokenizer)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        GrammarDataset(train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
    )
    dev_loader = DataLoader(
        GrammarDataset(dev, tokenizer, args.max_length),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    test_loader = DataLoader(
        GrammarDataset(test, tokenizer, args.max_length),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = load_model_with_retry(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights(train, device))

    best_state = None
    best_epoch = 0
    best_dev_f1 = -1.0
    epoch_rows: list[dict] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach().cpu())
        dev_gold, dev_pred, _ = predict(model, dev_loader, device)
        dev_metrics = metrics(dev_gold, dev_pred)
        epoch_rows.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
        })
        print(json.dumps(epoch_rows[-1], ensure_ascii=False), flush=True)
        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    test_gold, test_pred, confidence = predict(model, test_loader, device)
    test_metrics = metrics(test_gold, test_pred)
    duration = time.time() - started

    prediction_rows = []
    for row, gold_id, pred_id, conf in zip(test, test_gold, test_pred, confidence):
        kg_result = KG.evaluate(row)
        prediction_rows.append({
            "seed": seed,
            "编号": row["编号"],
            "sentence_group": sentence_group_id(row["句子"]),
            "语法家族": row["语法家族"],
            "候选语法点": row["候选语法点"],
            "gold": ID_TO_LABEL[gold_id],
            "prediction": ID_TO_LABEL[pred_id],
            "correct": int(gold_id == pred_id),
            "confidence": conf,
            "kg_strong_reject": int(kg_result["strong_reject"]),
            "kg_rule_trace": ";".join(kg_result["trace"]),
        })

    result = {
        "seed": seed,
        "task": "target_label",
        "model": "macbert_base_finetuned",
        **test_metrics,
        "best_dev_macro_f1": best_dev_f1,
        "best_epoch": best_epoch,
        "train_n": len(train),
        "dev_n": len(dev),
        "test_n": len(test),
        "duration_seconds": duration,
        "device": str(device),
        "zero_group_overlap": (
            manifest["group_overlap_train_dev"] == 0
            and manifest["group_overlap_train_test"] == 0
            and manifest["group_overlap_dev_test"] == 0
        ),
    }
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return result, prediction_rows, epoch_rows


def run() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    requested_seeds = set(args.seeds)
    results = [
        row for row in read_csv(OUT_DIR / "results_by_seed.csv")
        if int(row["seed"]) not in requested_seeds
    ] if (OUT_DIR / "results_by_seed.csv").exists() else []
    predictions = [
        row for row in read_csv(OUT_DIR / "predictions.csv")
        if int(row["seed"]) not in requested_seeds
    ] if (OUT_DIR / "predictions.csv").exists() else []
    epochs = [
        row for row in read_csv(OUT_DIR / "training_log.csv")
        if int(row["seed"]) not in requested_seeds
    ] if (OUT_DIR / "training_log.csv").exists() else []
    for seed in args.seeds:
        result, pred_rows, epoch_rows = train_one_seed(args, seed, tokenizer, device)
        results.append(result)
        predictions.extend(pred_rows)
        epochs.extend(epoch_rows)
        write_csv(OUT_DIR / "results_by_seed.csv", results)
        write_csv(OUT_DIR / "predictions.csv", predictions)
        write_csv(OUT_DIR / "training_log.csv", epochs)

    numeric_results = [
        {
            **row,
            **{
                metric: float(row[metric])
                for metric in ("accuracy", "macro_f1", "hard_negative_accuracy")
            },
        }
        for row in results
    ]
    aggregate = {
        metric: {
            "mean": float(np.mean([row[metric] for row in numeric_results])),
            "std": float(np.std([row[metric] for row in numeric_results], ddof=1)) if len(numeric_results) > 1 else 0.0,
            "min": min(row[metric] for row in numeric_results),
            "max": max(row[metric] for row in numeric_results),
        }
        for metric in ("accuracy", "macro_f1", "hard_negative_accuracy")
    }
    completed_seeds = sorted(int(row["seed"]) for row in numeric_results)
    environment = {
        "model": MODEL_NAME,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "config": {
            **vars(args),
            "seeds": completed_seeds,
            "requested_seeds_in_last_invocation": args.seeds,
        },
        "aggregate": aggregate,
    }
    (OUT_DIR / "experiment_config_and_summary.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()
