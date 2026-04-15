# SPEC: Vietnamese Clickbait Detection using Small Language Models

## 1. Project Overview

**Goal:** Fine-tune a small language model to classify Vietnamese news articles as `clickbait` or `non-clickbait` based on their title and lead paragraph.

**Input Data:** A CSV file containing Vietnamese news articles with columns:
`id, url, title, lead_paragraph, category, publish_datetime, source, thumbnail_url, label`

**Labels:** Binary classification — `clickbait` / `non-clickbait`

**Base Models (configurable via config):**

| Model | Params | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B` | 0.5B | **Default.** Best balance of speed and quality for low-resource devices. |
| `Qwen/Qwen2.5-1.5B` | 1.5B | Better accuracy, needs ~8GB VRAM. |
| `google/gemma-2-2b` | 2B | Strong multilingual. Needs ~10GB VRAM. |
| `microsoft/phi-2` | 2.7B | Good reasoning. Needs ~12GB VRAM. |
| `vinai/PhoGPT-1B5` | 1.5B | Vietnamese-specialized. Good if available. |

> **Recommendation:** Start with `Qwen/Qwen2.5-0.5B` for fast iteration, then scale up to 1.5B for production.

---

## 2. Folder Structure

```
current_folder/
├── configs/
│   └── config.yaml              # All training/model/data configurations
├── data/
│   ├── raw/
│   │   └── articles.csv         # Original dataset
│   └── processed/               # Auto-generated train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── __init__.py
│   ├── config.py                # Pydantic config loader (reads config.yaml)
│   ├── data_loader.py           # CSV parsing, text preprocessing, Dataset class
│   ├── model.py                 # Model + tokenizer loading, LoRA setup, quantization
│   ├── trainer.py               # Training loop (wraps HF Trainer)
│   ├── evaluate.py              # Metrics: accuracy, F1, precision, recall, confusion matrix
│   ├── predict.py               # Single-sample and batch inference
│   └── utils.py                 # Logging, seed setting, device detection
├── scripts/
│   ├── train.py                 # CLI entry point for local training
│   ├── test.py                  # CLI entry point for local evaluation/inference
│   └── export_model.py          # Export fine-tuned model for deployment
├── notebooks/
│   ├── train_colab.ipynb        # Google Colab training notebook
│   └── test_colab.ipynb         # Google Colab testing/inference notebook
├── outputs/                     # Auto-created: checkpoints, logs, metrics
│   ├── checkpoints/
│   ├── logs/
│   └── results/
├── tests/                       # Unit tests
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_config.py
├── pyproject.toml               # Project metadata + dependencies (uv-compatible)
├── .python-version              # Python version pin for uv
├── SPEC.md                      # This file
└── README.md                    # Quick-start guide
```

---

## 3. General Flow

### 3.1 Training Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Load Config │────▶│  Load & Split │────▶│  Preprocess &  │────▶│  Load Model  │
│  (YAML)      │     │  CSV Data     │     │  Tokenize      │     │  + LoRA/Quant│
└─────────────┘     └──────────────┘     └────────────────┘     └──────┬───────┘
                                                                       │
                    ┌──────────────┐     ┌────────────────┐            │
                    │  Save Model  │◀────│  Train with    │◀───────────┘
                    │  + Metrics   │     │  HF Trainer    │
                    └──────────────┘     └────────────────┘
```

**Step-by-step:**

1. **Config Loading** — Read `configs/config.yaml` via `src/config.py`. All hyperparameters, model choice, data paths, and optimization flags are centralized here.

2. **Data Loading & Splitting** — `src/data_loader.py` reads the raw CSV, constructs the input text from `title` and `lead_paragraph` using a prompt template, and splits into train/val/test sets based on configured ratios. Splits are stratified by label to maintain class balance. Processed splits are cached to `data/processed/`.

3. **Text Preprocessing & Tokenization** — Each sample is formatted as:
   ```
   Phân loại bài báo sau đây là "clickbait" hoặc "non-clickbait".

   Tiêu đề: {title}
   Đoạn mở đầu: {lead_paragraph}

   Nhãn:
   ```
   Tokenized with the model's tokenizer, padded/truncated to `max_length` from config.

4. **Model Loading** — `src/model.py` handles:
   - Loading the base model from HuggingFace
   - Applying **4-bit or 8-bit quantization** (via `bitsandbytes`) if configured
   - Attaching **LoRA adapters** (via `peft`) to reduce trainable parameters
   - Setting up the classification head (SequenceClassification or custom head)

5. **Training** — `src/trainer.py` wraps HuggingFace `Trainer` / `SFTTrainer` with:
   - Configurable epochs, batch size, learning rate, warmup, scheduler
   - Gradient accumulation for effective larger batches on small GPUs
   - Early stopping based on validation loss
   - Logging to console + TensorBoard/W&B (optional)

6. **Evaluation** — `src/evaluate.py` runs on the test split and produces:
   - Accuracy, Precision, Recall, F1 (macro + per-class)
   - Confusion matrix
   - Classification report saved to `outputs/results/`

7. **Saving** — The fine-tuned LoRA adapter (or full model) is saved to `outputs/checkpoints/`. Optionally merged with the base model for standalone deployment.

### 3.2 Inference Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Load Config │────▶│  Load Model  │────▶│  Preprocess   │────▶│  Predict │
│  + Checkpoint│     │  + LoRA      │     │  Input Text   │     │  Label   │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────┘
```

---

## 4. Configuration (`configs/config.yaml`)

```yaml
# ============================================================
# MODEL CONFIGURATION
# ============================================================
model:
  # Which base model to fine-tune (HuggingFace model ID)
  # Recommendations:
  #   - "Qwen/Qwen2.5-0.5B"    → fastest, ~2GB VRAM, good baseline
  #   - "Qwen/Qwen2.5-1.5B"    → better accuracy, ~8GB VRAM
  #   - "google/gemma-2-2b"     → strong multilingual, ~10GB VRAM
  #   - "vinai/PhoGPT-1B5"     → Vietnamese-specialized, ~8GB VRAM
  name: "Qwen/Qwen2.5-0.5B"
  num_labels: 2
  max_length: 512          # Max token length for input
  trust_remote_code: true

# ============================================================
# QUANTIZATION — reduces memory footprint for training/inference
# ============================================================
quantization:
  enabled: true
  bits: 4                  # 4 or 8. Use 4 for <6GB VRAM devices.
  quant_type: "nf4"        # "nf4" (recommended) or "fp4"
  use_double_quant: true   # Nested quantization, saves extra ~0.4GB
  compute_dtype: "bfloat16"  # "float16" or "bfloat16"

# ============================================================
# LoRA — Low-Rank Adaptation for parameter-efficient fine-tuning
# ============================================================
lora:
  enabled: true
  r: 16                    # LoRA rank. Lower = fewer params. Try 8/16/32.
  alpha: 32                # Scaling factor. Typically 2x of r.
  dropout: 0.05
  bias: "none"             # "none", "all", or "lora_only"
  # Which modules to attach LoRA to (model-specific):
  #   Qwen2.5:  ["q_proj", "v_proj"]  (minimal) or
  #             ["q_proj", "k_proj", "v_proj", "o_proj"] (recommended)
  #   Gemma:    ["q_proj", "v_proj"]
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
  # task_type is set automatically to SEQ_CLS

# ============================================================
# DATA
# ============================================================
data:
  raw_csv_path: "data/raw/articles.csv"
  processed_dir: "data/processed"
  # Input columns used for classification
  text_columns:
    - "title"
    - "lead_paragraph"
  label_column: "label"
  label_map:
    non-clickbait: 0
    clickbait: 1
  # Train/Validation/Test split ratios (must sum to 1.0)
  split_ratios:
    train: 0.8
    val: 0.1
    test: 0.1
  # Stratified split preserves label distribution
  stratified: true
  seed: 42

# ============================================================
# PROMPT TEMPLATE — how title + paragraph are combined
# ============================================================
prompt:
  template: |
    Phân loại bài báo sau đây là "clickbait" hoặc "non-clickbait".

    Tiêu đề: {title}
    Đoạn mở đầu: {lead_paragraph}

    Nhãn:
  # For sequence classification, we use the template to create input text.
  # For generative classification (SFT), model generates "clickbait" or "non-clickbait".

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
training:
  approach: "seq_cls"       # "seq_cls" (SequenceClassification) or "sft" (SFTTrainer generative)
  output_dir: "outputs"
  num_epochs: 5
  per_device_train_batch_size: 8    # Reduce to 2-4 for <6GB VRAM
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 4    # Effective batch = batch_size * grad_accum
  learning_rate: 2.0e-4             # Common range for LoRA: 1e-4 to 5e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"       # "cosine", "linear", "constant_with_warmup"
  fp16: false                       # Auto-set based on device. Don't use with bf16.
  bf16: true                        # Preferred on Ampere+ GPUs and Colab T4.
  logging_steps: 10
  eval_strategy: "steps"
  eval_steps: 50
  save_strategy: "steps"
  save_steps: 50
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "f1"
  greater_is_better: true
  early_stopping_patience: 3
  report_to: "none"                 # "tensorboard", "wandb", or "none"
  seed: 42

# ============================================================
# DEVICE PROFILES — quick presets for different hardware
# ============================================================
# Uncomment ONE profile or ignore to use the settings above.
# These override specific fields above when activated.
device_profiles:
  # --- Google Colab Free Tier (T4 16GB) ---
  colab_free:
    quantization.bits: 4
    training.per_device_train_batch_size: 4
    training.gradient_accumulation_steps: 8
    training.bf16: true
    model.max_length: 256

  # --- Low VRAM (4-6GB, e.g. GTX 1650, MX450) ---
  low_vram:
    quantization.bits: 4
    quantization.use_double_quant: true
    training.per_device_train_batch_size: 2
    training.gradient_accumulation_steps: 16
    model.max_length: 256
    lora.r: 8

  # --- Medium VRAM (8-12GB, e.g. RTX 3060/3070) ---
  medium_vram:
    quantization.bits: 4
    training.per_device_train_batch_size: 8
    training.gradient_accumulation_steps: 4
    model.max_length: 512

  # --- High VRAM (24GB+, e.g. RTX 3090/4090, A100) ---
  high_vram:
    quantization.enabled: false
    training.per_device_train_batch_size: 16
    training.gradient_accumulation_steps: 2
    model.max_length: 512
    lora.r: 32

# ============================================================
# INFERENCE
# ============================================================
inference:
  checkpoint_path: "outputs/checkpoints/best"   # Path to saved LoRA adapter or merged model
  batch_size: 32
  device: "auto"               # "auto", "cuda", "cpu", "mps"
```

---

## 5. Dependencies & Installation (using `uv`)

### 5.1 `pyproject.toml`

```toml
[project]
name = "clickbait-detector"
version = "0.1.0"
description = "Vietnamese clickbait detection via fine-tuned small LMs"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.44.0",
    "datasets>=3.0.0",
    "peft>=0.13.0",
    "bitsandbytes>=0.44.0",
    "trl>=0.12.0",
    "accelerate>=1.0.0",
    "scikit-learn>=1.5.0",
    "pandas>=2.2.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "tqdm>=4.66.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.6.0",
    "ipykernel>=6.29",
]
logging = [
    "tensorboard>=2.18",
    "wandb>=0.18",
]

[project.scripts]
train = "scripts.train:main"
test = "scripts.test:main"
```

### 5.2 `.python-version`

```
3.11
```

### 5.3 Installation Commands

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the project
git clone <repo-url> && cd clickbait-detector

# Create venv + install all dependencies
uv sync

# With optional dev/logging extras
uv sync --extra dev --extra logging

# Verify installation
uv run python -c "import torch; import transformers; import peft; print('OK')"
```

### 5.4 Google Colab Installation Cell

```python
# --- Run this as the first cell in Colab notebooks ---
!pip install -q torch transformers datasets peft bitsandbytes trl \
    accelerate scikit-learn pandas pyyaml pydantic tqdm matplotlib seaborn

# Clone repo
!git clone <repo-url> && cd clickbait-detector

# Upload your CSV to data/raw/articles.csv or mount Google Drive
```

---

## 6. Entry Points

### 6.1 Local Training — `scripts/train.py`

```bash
# Default (uses configs/config.yaml)
uv run python scripts/train.py

# Override config path
uv run python scripts/train.py --config configs/config.yaml

# Apply a device profile
uv run python scripts/train.py --profile low_vram

# Override specific values via CLI
uv run python scripts/train.py --model.name "Qwen/Qwen2.5-1.5B" --training.num_epochs 3
```

**What `scripts/train.py` does:**
1. Parses CLI args and loads config
2. Calls `src/data_loader.py` → split + tokenize
3. Calls `src/model.py` → load base model + quantize + attach LoRA
4. Calls `src/trainer.py` → train with HF Trainer
5. Calls `src/evaluate.py` → evaluate on val/test sets
6. Saves adapter + metrics to `outputs/`

### 6.2 Colab Training — `notebooks/train_colab.ipynb`

Notebook structure:
1. **Setup** — Install deps, mount Drive, upload CSV
2. **Config** — Inline YAML or load from file; apply `colab_free` profile
3. **Data** — Preview data, run split, show class distribution chart
4. **Train** — Execute training with live loss/metric plots
5. **Evaluate** — Confusion matrix, classification report, sample predictions
6. **Save** — Download adapter or save to Google Drive

### 6.3 Local Testing/Inference — `scripts/test.py`

```bash
# Evaluate on test split
uv run python scripts/test.py --mode evaluate

# Predict a single sample
uv run python scripts/test.py --mode predict \
    --title "Sốc: Bí mật kinh hoàng đằng sau vụ việc này" \
    --paragraph "Bạn sẽ không tin được những gì đã xảy ra..."

# Predict from a CSV file
uv run python scripts/test.py --mode predict_batch --input data/new_articles.csv --output outputs/predictions.csv
```

**What `scripts/test.py` does:**
1. Loads config + checkpoint from `inference.checkpoint_path`
2. Depending on `--mode`:
   - `evaluate`: loads test split, computes all metrics, prints report
   - `predict`: takes `--title` and `--paragraph`, returns label + confidence
   - `predict_batch`: reads input CSV, predicts all rows, writes output CSV

### 6.4 Colab Testing — `notebooks/test_colab.ipynb`

Notebook structure:
1. **Setup** — Install deps, load checkpoint from Drive
2. **Evaluate** — Run full test set evaluation with visual metrics
3. **Interactive Predict** — Text input widget for live predictions
4. **Batch Predict** — Upload CSV → get predictions CSV
5. **Error Analysis** — Show misclassified examples, confidence distributions

---

## 7. Module Specifications

### 7.1 `src/config.py`

- Uses **Pydantic** models to define and validate all config fields
- Loads from YAML file, with CLI override support
- Device profile application: merge profile overrides onto base config
- Auto-detects device (CUDA/MPS/CPU) and sets `fp16`/`bf16` accordingly
- Exposes a `get_config(config_path, profile, **overrides) -> Config` function

### 7.2 `src/data_loader.py`

- `load_raw_data(csv_path) -> pd.DataFrame` — read and validate CSV
- `split_data(df, ratios, stratified, seed) -> (train_df, val_df, test_df)` — stratified split using `sklearn.model_selection.train_test_split`
- `format_input(row, template) -> str` — apply prompt template
- `ClickbaitDataset(torch.utils.data.Dataset)` — tokenized dataset class
  - Handles padding, truncation, attention masks
  - Returns `input_ids`, `attention_mask`, `labels`
- Class imbalance handling: log class distribution, optionally compute class weights

### 7.3 `src/model.py`

- `load_tokenizer(model_name) -> AutoTokenizer` — loads tokenizer, sets pad token if missing
- `load_model(config) -> PreTrainedModel` — handles two approaches:
  - **Sequence Classification** (`seq_cls`): `AutoModelForSequenceClassification` with `num_labels=2`
  - **Generative/SFT** (`sft`): `AutoModelForCausalLM` — model generates label text
- `apply_quantization(model, quant_config) -> model` — BitsAndBytes 4/8-bit via `BitsAndBytesConfig`
- `apply_lora(model, lora_config) -> PeftModel` — attach LoRA adapters via `peft`
- `print_trainable_params(model)` — log total vs trainable parameter count

### 7.4 `src/trainer.py`

- `build_trainer(model, tokenizer, train_dataset, val_dataset, config) -> Trainer`
  - Uses HF `Trainer` for `seq_cls`, `SFTTrainer` from `trl` for `sft`
  - Configures `TrainingArguments` from config
  - Attaches `EarlyStoppingCallback`
  - Attaches `compute_metrics` function for eval
- `run_training(trainer) -> TrainOutput` — runs `.train()` and handles checkpointing

### 7.5 `src/evaluate.py`

- `compute_metrics(eval_pred) -> dict` — for HF Trainer callback: accuracy, F1, precision, recall
- `full_evaluation(model, tokenizer, test_dataset, config) -> dict` — complete evaluation:
  - All metrics (macro/weighted/per-class)
  - Confusion matrix (saved as PNG via matplotlib/seaborn)
  - Classification report (saved as text + JSON)
  - Confidence histogram
- `error_analysis(model, tokenizer, test_df, config) -> pd.DataFrame` — returns misclassified samples with confidence scores for debugging

### 7.6 `src/predict.py`

- `predict_single(title, paragraph, model, tokenizer, config) -> dict` — returns `{"label": "clickbait", "confidence": 0.93}`
- `predict_batch(input_csv, model, tokenizer, config) -> pd.DataFrame` — batch prediction with progress bar
- Handles both `seq_cls` (logits → softmax) and `sft` (generate → parse label) approaches

### 7.7 `src/utils.py`

- `set_seed(seed)` — reproducibility across torch/numpy/random
- `get_device()` — auto-detect best available device
- `setup_logging(output_dir)` — configure Python logging + optional TensorBoard
- `count_parameters(model)` — total and trainable param count

---

## 8. Training Approach Details

### 8.1 Approach A: Sequence Classification (Recommended Default)

- Adds a classification head (`Linear(hidden_dim, 2)`) on top of the base LM
- Input: tokenized prompt → model outputs logits → softmax → predicted class
- Loss: CrossEntropyLoss (with optional class weights)
- **Pros:** Faster training, simpler inference, well-suited for binary classification
- **Cons:** Less flexible than generative approach

### 8.2 Approach B: Supervised Fine-Tuning (Generative)

- Model learns to generate the label token (`"clickbait"` or `"non-clickbait"`) after the prompt
- Uses `SFTTrainer` from `trl`
- **Pros:** Can leverage model's language understanding more fully; easier to extend to multi-class later
- **Cons:** Slower inference (autoregressive generation), harder to constrain outputs

> **Recommendation:** Use `seq_cls` for production. Use `sft` for experimentation or if extending to more nuanced classification.

---

## 9. Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions / total |
| **F1 (macro)** | Harmonic mean of precision and recall, averaged across classes — **primary metric** |
| **Precision** | Of all predicted clickbait, how many were correct |
| **Recall** | Of all actual clickbait, how many were detected |
| **Confusion Matrix** | Visual 2×2 matrix of TP/FP/TN/FN |
| **Confidence Distribution** | Histogram of prediction probabilities for correct vs. incorrect |

---

## 10. Expected Outputs

After training completes, the `outputs/` folder will contain:

```
outputs/
├── checkpoints/
│   ├── best/                    # Best model (by F1) — LoRA adapter files
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── tokenizer/
│   ├── checkpoint-50/
│   └── checkpoint-100/
├── logs/
│   └── tensorboard/             # Training curves (if enabled)
├── results/
│   ├── classification_report.txt
│   ├── classification_report.json
│   ├── confusion_matrix.png
│   ├── confidence_histogram.png
│   ├── training_loss_curve.png
│   └── eval_metrics.json        # {"accuracy": 0.92, "f1": 0.91, ...}
└── predictions/
    └── test_predictions.csv     # id, true_label, pred_label, confidence
```

---

## 11. Export & Deployment

`scripts/export_model.py` supports:

```bash
# Merge LoRA adapter into base model for standalone use
uv run python scripts/export_model.py --merge --output outputs/merged_model

# Export to ONNX for faster CPU inference
uv run python scripts/export_model.py --format onnx --output outputs/model.onnx

# Export to GGUF for llama.cpp (generative approach only)
uv run python scripts/export_model.py --format gguf --output outputs/model.gguf
```

---

## 12. Testing (Unit Tests)

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_data_loader.py -v
```

**Test coverage:**

- `tests/test_config.py` — YAML loading, profile merging, validation of invalid configs
- `tests/test_data_loader.py` — CSV parsing, split ratios correctness, stratification verification, tokenization shape checks
- `tests/test_model.py` — Model loads correctly, LoRA attaches to expected modules, trainable param count is reduced, forward pass produces correct output shape

---

## 13. Notes & Considerations

1. **Vietnamese text:** Qwen2.5 has reasonable Vietnamese tokenization out of the box. No external Vietnamese tokenizer (e.g., VnCoreNLP) is required, but could improve results for larger models.

2. **Class imbalance:** If the dataset is heavily skewed toward `non-clickbait`, consider:
   - Weighted loss (`class_weight` in config)
   - Oversampling the minority class
   - Using F1 as the primary metric (already default)

3. **Data augmentation:** For small datasets (<1000 samples), consider:
   - Back-translation (Vietnamese → English → Vietnamese)
   - Synonym replacement
   - Title truncation/shuffling

4. **Reproducibility:** All random seeds are fixed via `src/utils.py`. Deterministic mode can be enabled in config but may slow training.

5. **Memory estimation for Qwen2.5-0.5B + 4-bit LoRA(r=16):**
   - Model: ~0.35GB (quantized)
   - LoRA params: ~2MB
   - Training overhead (optimizer, gradients): ~1.5GB
   - **Total: ~2-3GB VRAM** — fits on most modern GPUs and Colab free tier

6. **Colab session management:** Notebooks include Drive-saving cells after each epoch to guard against session disconnects.