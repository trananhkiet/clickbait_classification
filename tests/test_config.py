"""Unit tests for src/config.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config import Config, SplitRatios, get_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_yaml(tmp_path: Path) -> Path:
    cfg = {
        "model": {"name": "Qwen/Qwen2.5-0.5B"},
        "training": {"approach": "seq_cls", "num_epochs": 3},
        "data": {
            "raw_csv_path": "data/raw/articles.csv",
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


@pytest.fixture()
def profile_yaml(tmp_path: Path) -> Path:
    cfg = {
        "model": {"name": "Qwen/Qwen2.5-0.5B", "max_length": 512},
        "training": {"per_device_train_batch_size": 8, "bf16": True, "num_epochs": 5},
        "data": {
            "raw_csv_path": "data/raw/articles.csv",
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "device_profiles": {
            "low_vram": {
                "model.max_length": 256,
                "training.per_device_train_batch_size": 2,
            }
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_config():
    """Config can be instantiated with defaults."""
    cfg = Config()
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B"
    assert cfg.model.num_labels == 2
    assert cfg.training.approach == "seq_cls"


def test_split_ratios_must_sum_to_one():
    with pytest.raises(Exception):
        SplitRatios(train=0.7, val=0.2, test=0.2)  # sum = 1.1


def test_split_ratios_valid():
    sr = SplitRatios(train=0.8, val=0.1, test=0.1)
    assert abs(sr.train + sr.val + sr.test - 1.0) < 1e-6


def test_load_yaml(minimal_yaml: Path):
    cfg = get_config(minimal_yaml)
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B"
    assert cfg.training.num_epochs == 3


def test_profile_override(profile_yaml: Path):
    cfg = get_config(profile_yaml, profile="low_vram")
    assert cfg.model.max_length == 256
    assert cfg.training.per_device_train_batch_size == 2


def test_dotted_override(minimal_yaml: Path):
    cfg = get_config(minimal_yaml, **{"training.num_epochs": 10})
    assert cfg.training.num_epochs == 10


def test_invalid_profile(minimal_yaml: Path):
    with pytest.raises(ValueError, match="not found"):
        get_config(minimal_yaml, profile="nonexistent_profile")


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        get_config("nonexistent_config.yaml")


def test_label_map_defaults():
    cfg = Config()
    assert cfg.data.label_map["clickbait"] == 1
    assert cfg.data.label_map["non-clickbait"] == 0
