"""Training loop: builds HF Trainer (seq_cls) or SFTTrainer (sft) and runs training."""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.config import Config
from src.data_loader import ClickbaitDataset
from src.evaluate import compute_metrics

logger = logging.getLogger("clickbait")


def _build_training_args(config: Config) -> TrainingArguments:
    tcfg = config.training
    checkpoints_dir = str(Path(tcfg.output_dir) / "checkpoints")

    return TrainingArguments(
        output_dir=checkpoints_dir,
        num_train_epochs=tcfg.num_epochs,
        per_device_train_batch_size=tcfg.per_device_train_batch_size,
        per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        learning_rate=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
        warmup_ratio=tcfg.warmup_ratio,
        lr_scheduler_type=tcfg.lr_scheduler_type,
        fp16=tcfg.fp16,
        bf16=tcfg.bf16,
        logging_steps=tcfg.logging_steps,
        eval_strategy=tcfg.eval_strategy,
        eval_steps=tcfg.eval_steps,
        save_strategy=tcfg.save_strategy,
        save_steps=tcfg.save_steps,
        save_total_limit=tcfg.save_total_limit,
        load_best_model_at_end=tcfg.load_best_model_at_end,
        metric_for_best_model=tcfg.metric_for_best_model,
        greater_is_better=tcfg.greater_is_better,
        report_to=tcfg.report_to,
        seed=tcfg.seed,
        remove_unused_columns=False,
    )


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: ClickbaitDataset,
    val_dataset: ClickbaitDataset,
    config: Config,
) -> Trainer:
    training_args = _build_training_args(config)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)]

    if config.training.approach == "seq_cls":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
    else:
        # SFT approach — use trl SFTTrainer
        from trl import SFTConfig, SFTTrainer

        sft_args = SFTConfig(
            output_dir=training_args.output_dir,
            num_train_epochs=training_args.num_train_epochs,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            warmup_steps=training_args.warmup_steps,
            lr_scheduler_type=training_args.lr_scheduler_type,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            logging_steps=training_args.logging_steps,
            eval_strategy=training_args.eval_strategy,
            eval_steps=training_args.eval_steps,
            save_strategy=training_args.save_strategy,
            save_steps=training_args.save_steps,
            save_total_limit=training_args.save_total_limit,
            load_best_model_at_end=training_args.load_best_model_at_end,
            metric_for_best_model=training_args.metric_for_best_model,
            greater_is_better=training_args.greater_is_better,
            report_to=training_args.report_to,
            seed=training_args.seed,
            max_seq_length=config.model.max_length,
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

    logger.info("Trainer built (approach=%s)", config.training.approach)
    return trainer


def run_training(trainer: Trainer, config: Config) -> None:
    """Run training and save the best adapter."""
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # Save the best model adapter
    best_dir = str(Path(config.training.output_dir) / "checkpoints" / "best")
    trainer.save_model(best_dir)

    # Save tokenizer alongside
    if hasattr(trainer, "processing_class") and trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(best_dir)

    logger.info("Best model saved to %s", best_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
