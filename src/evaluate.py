"""Metrics computation, full evaluation, confusion matrix, and error analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import EvalPrediction, PreTrainedModel, PreTrainedTokenizerBase

from src.config import Config
from src.data_loader import ClickbaitDataset

logger = logging.getLogger("clickbait")


# ---------------------------------------------------------------------------
# HF Trainer callback metric
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Called by HF Trainer after each eval step."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def full_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_dataset: ClickbaitDataset,
    config: Config,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run complete evaluation and save artifacts."""
    output_dir = Path(output_dir or config.training.output_dir) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    logger.info("[Step 1/6] Building DataLoader (batch_size=%d, samples=%d)",
                config.inference.batch_size, len(test_dataset))
    loader = DataLoader(
        test_dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
    )

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []

    logger.info("[Step 2/6] Running inference over %d batches", len(loader))
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            print(batch["input_ids"].shape)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].tolist()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).tolist()
            confidence = probs[:, 1].tolist()  # probability for class 1 (clickbait)

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(confidence)
    logger.info("Inference complete: %d samples processed", len(all_preds))

    # --- Compute metrics ---
    logger.info("[Step 3/6] Computing metrics")
    label_names = list(config.data.label_map.keys())
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec,
        "recall_macro": rec,
    }
    logger.info("Evaluation results: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

    # --- Classification report ---
    logger.info("[Step 4/6] Generating classification report")
    report_str = classification_report(
        all_labels, all_preds, target_names=label_names, zero_division=0
    )
    report_dict = classification_report(
        all_labels, all_preds, target_names=label_names, zero_division=0, output_dict=True
    )
    print(report_str)
    (output_dir / "classification_report.txt").write_text(report_str)
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # --- Save eval_metrics.json ---
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics and report saved to %s", output_dir)

    # --- Confusion matrix ---
    logger.info("[Step 5/6] Saving confusion matrix")
    _save_confusion_matrix(all_labels, all_preds, label_names, output_dir)

    # --- Confidence histogram ---
    _save_confidence_histogram(all_labels, all_preds, all_probs, output_dir)

    # --- Test predictions CSV ---
    logger.info("[Step 6/6] Saving predictions CSV")
    _save_predictions_csv(test_dataset.df, all_preds, all_probs, config, output_dir)

    logger.info("Full evaluation complete. Results saved to %s", output_dir)
    return metrics


def _save_confusion_matrix(
    labels: list[int],
    preds: list[int],
    label_names: list[str],
    output_dir: Path,
) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_dir / "confusion_matrix.png")


def _save_confidence_histogram(
    labels: list[int],
    preds: list[int],
    probs: list[float],
    output_dir: Path,
) -> None:
    correct = [p for p, l, pr in zip(probs, labels, preds) if p == l]  # noqa: E741
    incorrect = [p for p, l, pr in zip(probs, labels, preds) if p != l]  # noqa: E741

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correct, bins=20, alpha=0.6, label="Correct", color="green")
    ax.hist(incorrect, bins=20, alpha=0.6, label="Incorrect", color="red")
    ax.set_xlabel("Confidence (P(clickbait))")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_histogram.png", dpi=150)
    plt.close(fig)


def _save_predictions_csv(
    df: pd.DataFrame,
    preds: list[int],
    probs: list[float],
    config: Config,
    output_dir: Path,
) -> None:
    inv_label_map = {v: k for k, v in config.data.label_map.items()}
    out = df.copy()
    out["pred_label"] = [inv_label_map[p] for p in preds]
    out["confidence"] = [round(pr, 4) for pr in probs]

    pred_dir = Path(config.training.output_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "test_predictions.csv"
    out.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def error_analysis(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_dataset: ClickbaitDataset,
    config: Config,
) -> pd.DataFrame:
    """Return a DataFrame of misclassified samples with confidence scores."""
    device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(test_dataset, batch_size=config.inference.batch_size, shuffle=False)

    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Error analysis", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).tolist()
            confidence = probs[:, 1].tolist()
            all_preds.extend(preds)
            all_probs.extend(confidence)

    inv_label_map = {v: k for k, v in config.data.label_map.items()}
    df = test_dataset.df.copy()
    df["true_label_id"] = test_dataset.labels.tolist()
    df["pred_label_id"] = all_preds
    df["confidence"] = [round(p, 4) for p in all_probs]
    df["true_label"] = df["true_label_id"].map(inv_label_map)
    df["pred_label"] = df["pred_label_id"].map(inv_label_map)

    errors = df[df["true_label_id"] != df["pred_label_id"]].copy()
    logger.info("Misclassified samples: %d / %d", len(errors), len(df))
    return errors
