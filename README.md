# Vietnamese Clickbait Detector

Fine-tune a small language model (Qwen2.5, Gemma, Phi-2, PhoGPT) to classify Vietnamese news articles as **clickbait** or **non-clickbait** using LoRA + optional 4-bit quantization.

---

## Quick Start

### 1. Install dependencies (requires [uv](https://github.com/astral-sh/uv))

```bash
uv sync                          # base dependencies
uv sync --extra dev --extra logging  # + pytest, ruff, tensorboard, wandb
```

### 2. Add your dataset

Place your CSV at `data/raw/articles.csv`. Required columns:

| Column | Description |
|---|---|
| `title` | Article headline |
| `lead_paragraph` | Opening paragraph |
| `label` | `clickbait` or `non-clickbait` |

### 3. Train

```bash
# Default (Qwen2.5-0.5B, seq_cls, config.yaml)
uv run python scripts/train.py

# With a hardware preset
uv run python scripts/train.py --profile low_vram

# Override individual hyperparameters
uv run python scripts/train.py --training.num_epochs 3 --model.name Qwen/Qwen2.5-1.5B
```

### 4. Evaluate / Infer

```bash
# Full test-set metrics
uv run python scripts/test.py --mode evaluate

# Single article
uv run python scripts/test.py --mode predict \
    --title "Sốc: Bí mật kinh hoàng đằng sau vụ việc này" \
    --paragraph "Bạn sẽ không tin được những gì đã xảy ra..."

# Batch inference from CSV
uv run python scripts/test.py --mode predict_batch \
    --input data/new_articles.csv \
    --output outputs/predictions.csv
```

### 5. Export model

```bash
# Merge LoRA into base model for standalone deployment
uv run python scripts/export_model.py --merge --output outputs/merged_model

# Export to ONNX (for CPU inference)
uv run python scripts/export_model.py --format onnx --output outputs/model.onnx
```

---

## Hardware Profiles

Pass `--profile <name>` to `scripts/train.py`:

| Profile | VRAM | Batch | Notes |
|---|---|---|---|
| `colab_free` | 16 GB (T4) | 4 | Google Colab free tier |
| `low_vram` | 4–6 GB | 2 | GTX 1650 / MX450 |
| `medium_vram` | 8–12 GB | 8 | RTX 3060/3070 |
| `high_vram` | 24 GB+ | 16 | RTX 3090/4090, A100 |

---

## Supported Base Models

| Model | Params | VRAM (4-bit) |
|---|---|---|
| `Qwen/Qwen2.5-0.5B` | 0.5B | ~2 GB |
| `Qwen/Qwen2.5-1.5B` | 1.5B | ~3 GB |
| `google/gemma-2-2b` | 2B | ~4 GB |
| `microsoft/phi-2` | 2.7B | ~5 GB |
| `vinai/PhoGPT-1B5` | 1.5B | ~3 GB |

---

## Outputs

```
outputs/
├── checkpoints/best/          # Best LoRA adapter (by F1)
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── confidence_histogram.png
│   └── eval_metrics.json
└── predictions/
    └── test_predictions.csv
```

---

## Google Colab

Open `notebooks/train_colab.ipynb` to train on Colab.
Open `notebooks/test_colab.ipynb` to evaluate and run inference interactively.

---

## Run Tests

```bash
uv run pytest tests/ -v
```

---

## Configuration

All settings live in `configs/config.yaml`. Key sections:

- `model` — which HuggingFace model to fine-tune
- `quantization` — 4-bit/8-bit BitsAndBytes settings
- `lora` — LoRA rank, alpha, target modules
- `training` — epochs, batch size, learning rate, scheduler
- `data` — CSV paths, split ratios, prompt template
- `inference` — checkpoint path, batch size

See `SPEC.md` for full documentation.
