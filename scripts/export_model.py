"""Export fine-tuned LoRA model for deployment.

Usage:
    # Merge LoRA adapter into base model
    uv run python scripts/export_model.py --merge --output outputs/merged_model

    # Export merged model to ONNX
    uv run python scripts/export_model.py --format onnx --output outputs/model.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.model import load_finetuned_model
from src.utils import setup_logging

logger = logging.getLogger("clickbait")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fine-tuned model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA into base model")
    parser.add_argument(
        "--format",
        choices=["pytorch", "onnx"],
        default="pytorch",
        help="Export format",
    )
    parser.add_argument("--output", default="outputs/exported_model", help="Output path")
    return parser.parse_args()


def merge_and_save(config, checkpoint_path: str | None, output_dir: str) -> None:
    """Merge LoRA adapters into the base model and save."""
    from peft import PeftModel
    from src.model import build_bnb_config, load_tokenizer
    from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
    import torch

    logger.info("Loading base model for merge...")
    tokenizer = load_tokenizer(config.model.name, config.model.trust_remote_code)

    # Load base WITHOUT quantization (merge requires full precision)
    base_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": torch.float16,
    }
    if config.training.approach == "seq_cls":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model.name, num_labels=config.model.num_labels, **base_kwargs
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(config.model.name, **base_kwargs)

    ckpt = checkpoint_path or config.inference.checkpoint_path
    peft_model = PeftModel.from_pretrained(base_model, ckpt)
    logger.info("Merging LoRA weights...")
    merged = peft_model.merge_and_unload()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    logger.info("Merged model saved to %s", out_path)


def export_onnx(config, output_path: str, checkpoint_path: str | None) -> None:
    """Export merged model to ONNX."""
    import torch

    merged_dir = str(Path(output_path).parent / "_merged_tmp")
    merge_and_save(config, checkpoint_path, merged_dir)

    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(merged_dir)
    model.eval()

    from src.model import load_tokenizer
    tokenizer = load_tokenizer(merged_dir, config.model.trust_remote_code)

    dummy = tokenizer(
        "dummy input",
        return_tensors="pt",
        padding="max_length",
        max_length=config.model.max_length,
        truncation=True,
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=14,
    )
    logger.info("ONNX model saved to %s", output_path)


def main() -> None:
    args = parse_args()
    config = get_config(args.config)
    setup_logging(config.training.output_dir)

    if args.format == "onnx":
        export_onnx(config, args.output, args.checkpoint)
    elif args.merge or args.format == "pytorch":
        merge_and_save(config, args.checkpoint, args.output)
    else:
        logger.error("Specify --merge or --format onnx")
        sys.exit(1)


if __name__ == "__main__":
    main()
