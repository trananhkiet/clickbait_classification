"""CLI entry point for local training.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --config configs/config.yaml
    uv run python scripts/train.py --profile low_vram
    uv run python scripts/train.py --model.name Qwen/Qwen2.5-1.5B --training.num_epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data_loader import prepare_datasets
from src.evaluate import full_evaluation
from src.model import apply_lora, load_model, load_tokenizer
from src.trainer import build_trainer, run_training
from src.utils import set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clickbait detection model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--profile", default=None, help="Device profile name from config")
    parser.add_argument("--force-reprocess", action="store_true", help="Re-split and re-tokenize data")

    # Allow arbitrary dotted overrides, e.g. --model.name Qwen/Qwen2.5-1.5B
    args, unknown = parser.parse_known_args()

    overrides: dict = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token.startswith("--") and "." in token:
            key = token[2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                value: str | int | float | bool = unknown[i + 1]
                # Try numeric / bool coercion
                try:
                    value = int(value)  # type: ignore[assignment]
                except ValueError:
                    try:
                        value = float(value)  # type: ignore[assignment]
                    except ValueError:
                        if value in ("true", "True"):
                            value = True
                        elif value in ("false", "False"):
                            value = False
                overrides[key] = value
                i += 2
            else:
                i += 1
        else:
            i += 1

    args.overrides = overrides
    return args


def main() -> None:
    args = parse_args()
    config = get_config(args.config, profile=args.profile, **args.overrides)

    setup_logging(config.training.output_dir)
    set_seed(config.training.seed)

    # 1. Load tokenizer
    tokenizer = load_tokenizer(config.model.name, config.model.trust_remote_code)

    # 2. Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(
        config, tokenizer, force_reprocess=args.force_reprocess
    )

    # 3. Load model + quantization + LoRA
    model = load_model(config)
    model = apply_lora(model, config)

    # 4. Train
    trainer = build_trainer(model, tokenizer, train_ds, val_ds, config)
    run_training(trainer, config)

    # 5. Evaluate on test set
    from pathlib import Path as _Path
    best_ckpt = str(_Path(config.training.output_dir) / "checkpoints" / "best")
    from src.model import load_finetuned_model
    eval_model, eval_tokenizer = load_finetuned_model(config, checkpoint_path=best_ckpt)
    full_evaluation(eval_model, eval_tokenizer, test_ds, config)


if __name__ == "__main__":
    main()
