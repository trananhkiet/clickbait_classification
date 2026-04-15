"""Pydantic config loader for configs/config.yaml with CLI override and device-profile support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("clickbait")


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-0.5B"
    num_labels: int = 2
    max_length: int = 512
    trust_remote_code: bool = True


class QuantizationConfig(BaseModel):
    enabled: bool = True
    bits: Literal[4, 8] = 4
    quant_type: Literal["nf4", "fp4"] = "nf4"
    use_double_quant: bool = True
    compute_dtype: Literal["float16", "bfloat16"] = "bfloat16"


class LoraConfig(BaseModel):
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


class SplitRatios(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    @model_validator(mode="after")
    def _check_sum(self) -> "SplitRatios":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {total}")
        return self


class DataConfig(BaseModel):
    raw_csv_path: str = "data/raw/articles.csv"
    processed_dir: str = "data/processed"
    text_columns: list[str] = Field(default_factory=lambda: ["title", "lead_paragraph"])
    label_column: str = "label"
    label_map: dict[str, int] = Field(default_factory=lambda: {"non-clickbait": 0, "clickbait": 1})
    split_ratios: SplitRatios = Field(default_factory=SplitRatios)
    stratified: bool = True
    seed: int = 42


class PromptConfig(BaseModel):
    template: str = (
        'Phân loại bài báo sau đây là "clickbait" hoặc "non-clickbait".\n\n'
        "Tiêu đề: {title}\n"
        "Đoạn mở đầu: {lead_paragraph}\n\n"
        "Nhãn:\n"
    )


class TrainingConfig(BaseModel):
    approach: Literal["seq_cls", "sft"] = "seq_cls"
    output_dir: str = "outputs"
    num_epochs: int = 5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    report_to: str = "none"
    seed: int = 42


class InferenceConfig(BaseModel):
    checkpoint_path: str = "outputs/checkpoints/best"
    batch_size: int = 32
    device: str = "auto"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    # device_profiles are only for reference; not parsed into typed models
    device_profiles: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_dotted_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply flat dotted-key overrides (e.g. {"training.num_epochs": 3}) to nested dict."""
    for key, value in overrides.items():
        parts = key.split(".")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return data


def _apply_profile(data: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profiles = data.get("device_profiles", {})
    if profile_name not in profiles:
        raise ValueError(
            f"Profile '{profile_name}' not found. Available: {list(profiles.keys())}"
        )
    profile_overrides = profiles[profile_name]
    return _apply_dotted_overrides(data, profile_overrides)


def _auto_adjust_precision(config: Config) -> Config:
    """Disable bf16 on CPU/MPS; enable fp16 on CUDA if bf16 unsupported."""
    device_str = config.inference.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

    if device_str in ("cpu", "mps"):
        config.training.bf16 = False
        config.training.fp16 = False
    elif device_str == "cuda":
        # Keep user settings; just log
        pass
    return config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_config(
    config_path: str | Path = "configs/config.yaml",
    profile: str | None = None,
    **overrides: Any,
) -> Config:
    """Load config from YAML, optionally apply a device profile and CLI overrides."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    if profile:
        raw = _apply_profile(raw, profile)

    if overrides:
        raw = _apply_dotted_overrides(raw, overrides)

    config = Config.model_validate(raw)
    config = _auto_adjust_precision(config)
    logger.info("Config loaded from %s (profile=%s)", config_path, profile)
    return config
