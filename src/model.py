"""Model + tokenizer loading, quantization (BitsAndBytes), and LoRA (PEFT)."""

from __future__ import annotations

import logging

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config import Config
from src.utils import print_trainable_params

logger = logging.getLogger("clickbait")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    # Many decoder-only models lack a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)
    return tokenizer


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def build_bnb_config(config: Config) -> BitsAndBytesConfig | None:
    qcfg = config.quantization
    if not qcfg.enabled:
        return None

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    compute_dtype = dtype_map.get(qcfg.compute_dtype, torch.bfloat16)

    if qcfg.bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qcfg.quant_type,
            bnb_4bit_use_double_quant=qcfg.use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif qcfg.bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(config: Config) -> PreTrainedModel:
    """Load the base model (seq_cls or causal LM) with optional quantization."""
    mcfg = config.model
    bnb_config = build_bnb_config(config)

    kwargs: dict = {
        "trust_remote_code": mcfg.trust_remote_code,
        "torch_dtype": torch.bfloat16 if config.training.bf16 else torch.float16,
    }
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = {"": 0}  # pin to GPU 0; avoids DataParallel conflict

    if config.training.approach == "seq_cls":
        model = AutoModelForSequenceClassification.from_pretrained(
            mcfg.name,
            num_labels=mcfg.num_labels,
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(mcfg.name, **kwargs)

    # Ensure pad_token_id is set (decoder-only models often lack it)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        logger.info("Set model.config.pad_token_id = eos_token_id (%s)", model.config.eos_token_id)

    logger.info(
        "Loaded model '%s' (approach=%s, quantization=%s)",
        mcfg.name,
        config.training.approach,
        f"{config.quantization.bits}-bit" if config.quantization.enabled else "none",
    )
    return model


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


def apply_lora(model: PreTrainedModel, config: Config) -> PreTrainedModel:
    """Attach LoRA adapters using PEFT."""
    if not config.lora.enabled:
        return model

    lcfg = config.lora
    task_type = (
        TaskType.SEQ_CLS if config.training.approach == "seq_cls" else TaskType.CAUSAL_LM
    )

    lora_config = LoraConfig(
        r=lcfg.r,
        lora_alpha=lcfg.alpha,
        lora_dropout=lcfg.dropout,
        bias=lcfg.bias,
        target_modules=lcfg.target_modules,
        task_type=task_type,
    )

    # Prepare for k-bit training when quantization is enabled
    if config.quantization.enabled:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    model = get_peft_model(model, lora_config)
    logger.info(
        "LoRA applied — r=%d, alpha=%d, target_modules=%s",
        lcfg.r,
        lcfg.alpha,
        lcfg.target_modules,
    )
    print_trainable_params(model)
    return model


# ---------------------------------------------------------------------------
# Load fine-tuned model from checkpoint
# ---------------------------------------------------------------------------


def load_finetuned_model(
    config: Config,
    checkpoint_path: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a fine-tuned model (LoRA adapter merged or standalone) for inference."""
    from peft import PeftModel

    ckpt = checkpoint_path or config.inference.checkpoint_path
    tokenizer = load_tokenizer(config.model.name, config.model.trust_remote_code)

    base_model = load_model(config)

    # Check if it's a PEFT/LoRA adapter directory
    import os
    if os.path.exists(os.path.join(ckpt, "adapter_config.json")):
        model = PeftModel.from_pretrained(base_model, ckpt)
        logger.info("Loaded LoRA adapter from %s", ckpt)
    else:
        # Assume it's a full model directory
        if config.training.approach == "seq_cls":
            model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        else:
            model = AutoModelForCausalLM.from_pretrained(ckpt)
        logger.info("Loaded full model from %s", ckpt)

    model.eval()
    return model, tokenizer
