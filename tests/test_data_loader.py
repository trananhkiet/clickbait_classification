"""Unit tests for src/data_loader.py."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from unittest.mock import MagicMock

from src.data_loader import (
    ClickbaitDataset,
    compute_class_weights,
    format_input,
    load_raw_data,
    split_data,
)
from src.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(100),
            "title": [f"Title {i}" for i in range(100)],
            "lead_paragraph": [f"Paragraph {i}" for i in range(100)],
            "label": (["clickbait"] * 50) + (["non-clickbait"] * 50),
            "category": ["news"] * 100,
        }
    )


@pytest.fixture()
def raw_csv(tmp_path, sample_df) -> str:
    p = tmp_path / "articles.csv"
    sample_df.to_csv(p, index=False)
    return str(p)


@pytest.fixture()
def mock_tokenizer():
    """Minimal tokenizer mock that mimics HF tokenizer output."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0

    def fake_call(texts, padding, truncation, max_length, return_tensors):
        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.zeros(n, max_length, dtype=torch.long),
            "attention_mask": torch.ones(n, max_length, dtype=torch.long),
        }

    tokenizer.side_effect = fake_call
    tokenizer.__call__ = fake_call
    return tokenizer


# ---------------------------------------------------------------------------
# Tests — load_raw_data
# ---------------------------------------------------------------------------


def test_load_raw_data(raw_csv):
    df = load_raw_data(raw_csv)
    assert len(df) == 100
    assert "title" in df.columns
    assert "lead_paragraph" in df.columns
    assert "label" in df.columns


def test_load_raw_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_raw_data("does_not_exist.csv")


def test_load_raw_data_drops_nulls(tmp_path):
    df = pd.DataFrame(
        {
            "title": ["Good", None, "Also good"],
            "lead_paragraph": ["Para", "Para", "Para"],
            "label": ["clickbait", "non-clickbait", "clickbait"],
        }
    )
    p = tmp_path / "articles.csv"
    df.to_csv(p, index=False)
    result = load_raw_data(str(p))
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests — split_data
# ---------------------------------------------------------------------------


def test_split_sizes(sample_df):
    train, val, test = split_data(sample_df, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    total = len(train) + len(val) + len(test)
    assert total == len(sample_df)
    assert abs(len(train) / total - 0.8) < 0.05


def test_split_stratification(sample_df):
    train, val, test = split_data(
        sample_df, {"train": 0.8, "val": 0.1, "test": 0.1}, stratified=True, seed=42
    )
    for split in [train, val, test]:
        counts = split["label"].value_counts(normalize=True)
        assert abs(counts.get("clickbait", 0) - 0.5) < 0.1


def test_split_reproducibility(sample_df):
    t1, v1, te1 = split_data(sample_df, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    t2, v2, te2 = split_data(sample_df, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    pd.testing.assert_frame_equal(t1, t2)


# ---------------------------------------------------------------------------
# Tests — format_input
# ---------------------------------------------------------------------------


def test_format_input():
    template = "Tiêu đề: {title}\nĐoạn mở đầu: {lead_paragraph}\nNhãn:"
    row = pd.Series({"title": "Hello", "lead_paragraph": "World"})
    result = format_input(row, template)
    assert "Hello" in result
    assert "World" in result
    assert "Nhãn:" in result


# ---------------------------------------------------------------------------
# Tests — ClickbaitDataset
# ---------------------------------------------------------------------------


def test_dataset_length(sample_df, mock_tokenizer):
    config = Config()
    ds = ClickbaitDataset(sample_df, mock_tokenizer, config)
    assert len(ds) == len(sample_df)


def test_dataset_item_keys(sample_df, mock_tokenizer):
    config = Config()
    ds = ClickbaitDataset(sample_df, mock_tokenizer, config)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item


def test_dataset_tensor_shapes(sample_df, mock_tokenizer):
    config = Config()
    ds = ClickbaitDataset(sample_df, mock_tokenizer, config)
    item = ds[0]
    assert item["input_ids"].shape == (config.model.max_length,)
    assert item["attention_mask"].shape == (config.model.max_length,)
    assert item["labels"].dim() == 0  # scalar


def test_dataset_label_encoding(sample_df, mock_tokenizer):
    config = Config()
    ds = ClickbaitDataset(sample_df, mock_tokenizer, config)
    labels = ds.labels.tolist()
    assert set(labels) == {0, 1}


# ---------------------------------------------------------------------------
# Tests — compute_class_weights
# ---------------------------------------------------------------------------


def test_class_weights_balanced():
    labels = [0, 0, 1, 1]
    weights = compute_class_weights(labels, num_classes=2)
    assert abs(weights[0].item() - weights[1].item()) < 1e-4


def test_class_weights_imbalanced():
    labels = [0, 0, 0, 1]
    weights = compute_class_weights(labels, num_classes=2)
    # Minority class (1) should have higher weight
    assert weights[1] > weights[0]
