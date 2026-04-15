"""Utility functions: seed setting, device detection, logging, parameter counting."""

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """Return the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def setup_logging(output_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger to console (and optionally a file in output_dir)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if output_dir:
        log_path = Path(output_dir) / "logs" / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=handlers,
    )
    return logging.getLogger("clickbait")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_trainable_params(model: torch.nn.Module) -> None:
    total, trainable = count_parameters(model)
    pct = 100.0 * trainable / total if total else 0.0
    logger = logging.getLogger("clickbait")
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )
