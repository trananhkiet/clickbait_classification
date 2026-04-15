"""CLI entry point for evaluation and inference.

Usage:
    # Full test-set evaluation
    uv run python scripts/test.py --mode evaluate

    # Single prediction
    uv run python scripts/test.py --mode predict \\
        --title "Sốc: Bí mật kinh hoàng" \\
        --paragraph "Bạn sẽ không tin được..."

    # Batch prediction from CSV
    uv run python scripts/test.py --mode predict_batch \\
        --input data/new_articles.csv \\
        --output outputs/predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data_loader import ClickbaitDataset, load_splits
from src.evaluate import full_evaluation
from src.model import load_finetuned_model
from src.predict import predict_batch, predict_single
from src.utils import set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or run inference")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--mode",
        choices=["evaluate", "predict", "predict_batch"],
        default="evaluate",
    )
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    # predict mode
    parser.add_argument("--title", default=None)
    parser.add_argument("--paragraph", default=None)
    # predict_batch mode
    parser.add_argument("--input", default=None, help="Input CSV for batch prediction")
    parser.add_argument("--output", default=None, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config(args.config, profile=args.profile)
    setup_logging(config.training.output_dir)
    set_seed(config.training.seed)

    model, tokenizer = load_finetuned_model(config, checkpoint_path=args.checkpoint)

    if args.mode == "evaluate":
        _, _, test_df = load_splits(config.data.processed_dir)
        test_ds = ClickbaitDataset(test_df, tokenizer, config)
        metrics = full_evaluation(model, tokenizer, test_ds, config)
        print("\nTest Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    elif args.mode == "predict":
        if not args.title or not args.paragraph:
            print("--title and --paragraph are required for predict mode.", file=sys.stderr)
            sys.exit(1)
        result = predict_single(args.title, args.paragraph, model, tokenizer, config)
        print(f"\nLabel:      {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")

    elif args.mode == "predict_batch":
        if not args.input:
            print("--input is required for predict_batch mode.", file=sys.stderr)
            sys.exit(1)
        out_df = predict_batch(args.input, model, tokenizer, config)
        output_path = args.output or "outputs/predictions/batch_predictions.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
