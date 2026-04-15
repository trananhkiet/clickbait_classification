"""CSV parsing, text preprocessing, stratified splits, and PyTorch Dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config import Config

logger = logging.getLogger("clickbait")


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and do minimal validation."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"title", "lead_paragraph", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Drop rows with null in key columns
    before = len(df)
    df = df.dropna(subset=list(required_cols))
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with null values in key columns.", dropped)

    logger.info("Loaded %d samples from %s", len(df), csv_path)
    _log_class_distribution(df, "label")
    return df


def _log_class_distribution(df: pd.DataFrame, label_col: str) -> None:
    counts = df[label_col].value_counts()
    total = len(df)
    for label, count in counts.items():
        logger.info("  %s: %d (%.1f%%)", label, count, 100.0 * count / total)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    ratios: dict[str, float],
    stratified: bool = True,
    label_col: str = "label",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val, test) according to ratios."""
    train_ratio = ratios["train"]
    val_ratio = ratios["val"]
    # test_ratio = ratios["test"]

    stratify = df[label_col] if stratified else None

    train_df, temp_df = train_test_split(
        df,
        test_size=1.0 - train_ratio,
        stratify=stratify,
        random_state=seed,
    )

    # Split temp into val / test
    val_fraction_of_temp = val_ratio / (1.0 - train_ratio)
    stratify_temp = temp_df[label_col] if stratified else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_fraction_of_temp,
        stratify=stratify_temp,
        random_state=seed,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: str | Path,
) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    logger.info("Splits saved to %s", processed_dir)


def load_splits(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    logger.info("Loaded cached splits from %s", processed_dir)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_input(row: pd.Series, template: str) -> str:
    """Apply the prompt template to a DataFrame row."""
    return template.format(
        title=str(row.get("title", "")),
        lead_paragraph=str(row.get("lead_paragraph", "")),
    )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class ClickbaitDataset(Dataset):
    """Tokenized PyTorch dataset for clickbait classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        config: Config,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config

        label_map = config.data.label_map
        template = config.prompt.template
        max_length = config.model.max_length

        texts = [format_input(row, template) for _, row in df.iterrows()]
        labels = [label_map[str(row[config.data.label_column])] for _, row in df.iterrows()]

        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        self.input_ids: torch.Tensor = encodings["input_ids"]
        self.attention_mask: torch.Tensor = encodings["attention_mask"]
        self.labels: torch.Tensor = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Class weight utility
# ---------------------------------------------------------------------------


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    counts = torch.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * num_classes
    return weights


# ---------------------------------------------------------------------------
# High-level pipeline helper
# ---------------------------------------------------------------------------


def prepare_datasets(
    config: Config,
    tokenizer: PreTrainedTokenizerBase,
    force_reprocess: bool = False,
) -> tuple[ClickbaitDataset, ClickbaitDataset, ClickbaitDataset]:
    """Full pipeline: load → split → cache → tokenize → return datasets."""
    processed_dir = Path(config.data.processed_dir)
    splits_exist = all(
        (processed_dir / f"{s}.csv").exists() for s in ("train", "val", "test")
    )

    if splits_exist and not force_reprocess:
        train_df, val_df, test_df = load_splits(processed_dir)
    else:
        df = load_raw_data(config.data.raw_csv_path)
        ratios = {
            "train": config.data.split_ratios.train,
            "val": config.data.split_ratios.val,
            "test": config.data.split_ratios.test,
        }
        train_df, val_df, test_df = split_data(
            df,
            ratios=ratios,
            stratified=config.data.stratified,
            label_col=config.data.label_column,
            seed=config.data.seed,
        )
        save_splits(train_df, val_df, test_df, processed_dir)

    train_ds = ClickbaitDataset(train_df, tokenizer, config)
    val_ds = ClickbaitDataset(val_df, tokenizer, config)
    test_ds = ClickbaitDataset(test_df, tokenizer, config)
    return train_ds, val_ds, test_ds
