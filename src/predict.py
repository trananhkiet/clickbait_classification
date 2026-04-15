"""Single-sample and batch inference."""

from __future__ import annotations

import logging

import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import Config
from src.data_loader import format_input

logger = logging.getLogger("clickbait")

_INV_LABEL_MAP_CACHE: dict[str, dict[int, str]] = {}


def _get_inv_label_map(config: Config) -> dict[int, str]:
    key = str(id(config))
    if key not in _INV_LABEL_MAP_CACHE:
        _INV_LABEL_MAP_CACHE[key] = {v: k for k, v in config.data.label_map.items()}
    return _INV_LABEL_MAP_CACHE[key]


# ---------------------------------------------------------------------------
# seq_cls inference
# ---------------------------------------------------------------------------


def _predict_seq_cls(
    texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> list[dict]:
    device = next(model.parameters()).device
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config.model.max_length,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1).tolist()
    confidences = probs.max(dim=-1).values.tolist()

    inv_label_map = _get_inv_label_map(config)
    return [
        {"label": inv_label_map[pid], "confidence": round(conf, 4)}
        for pid, conf in zip(pred_ids, confidences)
    ]


# ---------------------------------------------------------------------------
# sft (generative) inference
# ---------------------------------------------------------------------------

_SFT_LABEL_TOKENS = {"clickbait", "non-clickbait", "non_clickbait"}


def _predict_sft(
    texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> list[dict]:
    device = next(model.parameters()).device
    results = []

    for text in texts:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.model.max_length,
        )
        input_ids = encodings["input_ids"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip().lower()

        label = "non-clickbait"
        for token in _SFT_LABEL_TOKENS:
            if token in generated:
                label = "clickbait" if "clickbait" == token else "non-clickbait"
                if token == "clickbait" and "non" not in generated:
                    label = "clickbait"
                break

        results.append({"label": label, "confidence": 1.0})

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_single(
    title: str,
    paragraph: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> dict:
    """Predict a single article.

    Returns: {"label": "clickbait"|"non-clickbait", "confidence": float}
    """
    row = pd.Series({"title": title, "lead_paragraph": paragraph})
    text = format_input(row, config.prompt.template)

    if config.training.approach == "seq_cls":
        return _predict_seq_cls([text], model, tokenizer, config)[0]
    return _predict_sft([text], model, tokenizer, config)[0]


def predict_batch(
    input_csv: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> pd.DataFrame:
    """Predict all rows in input_csv. Returns DataFrame with added columns."""
    df = pd.read_csv(input_csv)
    texts = [
        format_input(row, config.prompt.template) for _, row in df.iterrows()
    ]

    batch_size = config.inference.batch_size
    all_results: list[dict] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i : i + batch_size]
        if config.training.approach == "seq_cls":
            batch_results = _predict_seq_cls(batch_texts, model, tokenizer, config)
        else:
            batch_results = _predict_sft(batch_texts, model, tokenizer, config)
        all_results.extend(batch_results)

    df["pred_label"] = [r["label"] for r in all_results]
    df["confidence"] = [r["confidence"] for r in all_results]
    return df
