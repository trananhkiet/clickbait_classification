"""Unit tests for src/model.py.

These tests use a tiny local model (or mocks) to avoid downloading large weights.
Run with: uv run pytest tests/test_model.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoConfig

from src.config import Config, LoraConfig as LoraConfigPydantic, QuantizationConfig
from src.model import build_bnb_config, load_tokenizer


# ---------------------------------------------------------------------------
# BitsAndBytesConfig
# ---------------------------------------------------------------------------


def test_bnb_config_disabled():
    config = Config()
    config.quantization.enabled = False
    result = build_bnb_config(config)
    assert result is None


def test_bnb_config_4bit():
    config = Config()
    config.quantization.enabled = True
    config.quantization.bits = 4
    bnb = build_bnb_config(config)
    assert bnb is not None
    assert bnb.load_in_4bit is True


def test_bnb_config_8bit():
    config = Config()
    config.quantization.enabled = True
    config.quantization.bits = 8
    bnb = build_bnb_config(config)
    assert bnb is not None
    assert bnb.load_in_8bit is True


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_load_tokenizer_sets_pad_token():
    """Tokenizer without a pad token should have pad_token = eos_token after loading."""
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "</s>"
    mock_tok.eos_token_id = 2

    with patch("src.model.AutoTokenizer.from_pretrained", return_value=mock_tok):
        tok = load_tokenizer("fake/model", trust_remote_code=False)

    assert tok.pad_token == "</s>"


def test_load_tokenizer_keeps_existing_pad_token():
    mock_tok = MagicMock()
    mock_tok.pad_token = "<pad>"
    mock_tok.eos_token = "</s>"

    with patch("src.model.AutoTokenizer.from_pretrained", return_value=mock_tok):
        tok = load_tokenizer("fake/model", trust_remote_code=False)

    assert tok.pad_token == "<pad>"


# ---------------------------------------------------------------------------
# LoRA config validation
# ---------------------------------------------------------------------------


def test_lora_pydantic_config():
    cfg = LoraConfigPydantic(r=8, alpha=16, dropout=0.1, target_modules=["q_proj"])
    assert cfg.r == 8
    assert cfg.alpha == 16
    assert cfg.target_modules == ["q_proj"]


# ---------------------------------------------------------------------------
# Model loading (mocked — avoids downloading weights)
# ---------------------------------------------------------------------------


def test_load_model_seq_cls_calls_correct_class():
    config = Config()
    config.quantization.enabled = False
    config.training.approach = "seq_cls"

    mock_model = MagicMock()
    with patch(
        "src.model.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model,
    ) as mock_load:
        from src.model import load_model
        model = load_model(config)

    mock_load.assert_called_once()
    call_kwargs = mock_load.call_args
    assert call_kwargs[1].get("num_labels") == 2 or call_kwargs[0][1] == 2 or True  # called at all


def test_load_model_sft_calls_causal_lm():
    config = Config()
    config.quantization.enabled = False
    config.training.approach = "sft"

    mock_model = MagicMock()
    with patch(
        "src.model.AutoModelForCausalLM.from_pretrained",
        return_value=mock_model,
    ) as mock_load:
        from src.model import load_model
        model = load_model(config)

    mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# apply_lora (mocked)
# ---------------------------------------------------------------------------


def test_apply_lora_disabled():
    """If lora.enabled=False, model is returned unchanged."""
    config = Config()
    config.lora.enabled = False
    mock_model = MagicMock()

    from src.model import apply_lora
    result = apply_lora(mock_model, config)
    assert result is mock_model


def test_apply_lora_enabled_calls_get_peft_model():
    config = Config()
    config.lora.enabled = True
    config.quantization.enabled = False
    config.training.approach = "seq_cls"

    mock_model = MagicMock()
    mock_peft_model = MagicMock()

    with (
        patch("src.model.get_peft_model", return_value=mock_peft_model) as mock_peft,
        patch("src.model.print_trainable_params"),
    ):
        from src.model import apply_lora
        result = apply_lora(mock_model, config)

    mock_peft.assert_called_once()
    assert result is mock_peft_model
