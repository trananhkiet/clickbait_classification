"""CSV parsing, text preprocessing, stratified splits, and PyTorch Dataset."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config import AugmentationConfig, Config

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
# Text augmentation (EDA-style)
# ---------------------------------------------------------------------------


def _random_deletion(words: list[str], p: float) -> list[str]:
    if len(words) == 1:
        return words
    result = [w for w in words if random.random() > p]
    return result if result else [random.choice(words)]


def _random_swap(words: list[str], n: int) -> list[str]:
    words = words.copy()
    for _ in range(n):
        if len(words) < 2:
            break
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words


def augment_text(text: str, aug_cfg: AugmentationConfig) -> str:
    words = text.split()
    if not words:
        return text
    words = _random_deletion(words, aug_cfg.deletion_prob)
    words = _random_swap(words, aug_cfg.n_swaps)
    return " ".join(words)


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
        augment: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config

        label_map = config.data.label_map
        template = config.prompt.template
        max_length = config.model.max_length

        texts = [format_input(row, template) for _, row in df.iterrows()]
        labels = [label_map[str(row[config.data.label_column])] for _, row in df.iterrows()]

        if augment and config.augmentation.enabled:
            aug_cfg = config.augmentation
            if aug_cfg.minority_only:
                counts: dict[int, int] = {}
                for lbl in labels:
                    counts[lbl] = counts.get(lbl, 0) + 1
                minority_label = min(counts, key=lambda k: counts[k])
                aug_indices = [i for i, lbl in enumerate(labels) if lbl == minority_label]
            else:
                aug_indices = list(range(len(texts)))

            aug_texts = [augment_text(texts[i], aug_cfg) for i in aug_indices]
            aug_labels = [labels[i] for i in aug_indices]
            texts = texts + aug_texts
            labels = labels + aug_labels
            logger.info(
                "Augmentation: added %d samples (minority_only=%s). New total: %d",
                len(aug_texts), aug_cfg.minority_only, len(texts),
            )

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
