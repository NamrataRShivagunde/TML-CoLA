# TML-CoLA

**Chain of Low-rank Adapters (CoLA)**: An efficient parameter-efficient fine-tuning method for large language models.

## Overview

This repository contains the implementation of CoLA and CoLA-M, methods for efficient training of large language models using low-rank adapters. The repository includes:

- **Baseline models**: Standard LLaMA models trained from scratch
- **CoLA**: Chain of Low-rank Adapters
- **CoLA-M**: Memory-efficient CoLA with gradient checkpointing
- **Offline mode**: Support for training with pre-downloaded datasets

## Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Training](#training)
  - [Baseline Models](#baseline-models)
  - [CoLA Models](#cola-models)
  - [CoLA-M Models](#cola-m-models)
- [Offline Mode](#offline-mode)
- [Configuration](#configuration)
- [Model Sizes](#model-sizes)

## Installation

```bash
# Clone the repository
git clone https://github.com/NamrataRShivagunde/TML-CoLA.git
cd TML-CoLA

# Create and activate conda environment
conda create -n cola python=3.11
conda activate cola

# Install dependencies
pip install torch transformers datasets wandb loguru tqdm
```

## Dataset Setup

The training uses the C4 (Colossal Clean Crawled Corpus) dataset from HuggingFace.

### Online Mode (Default)
By default, the training script streams data from HuggingFace, which requires an internet connection:

```bash
cd CoLA
torchrun --nproc_per_node=1 main.py \
    --model_config baseline_configs/llama_60m.json \
    --model_type llama \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 10000
```

### Offline Mode (Recommended for Servers)

For server environments or to avoid repeated downloads, you can pre-download and tokenize the dataset:

#### Step 1: Download the Dataset

```bash
# Quick test (50k tokens, ~2-3 minutes)
python scripts/download_c4_offline.py --target_tokens 50000 --val_tokens 5000

# Full download (10M tokens, ~10-30 minutes)
python scripts/download_c4_offline.py --target_tokens 10000000 --val_tokens 100000
```

This creates `datasets/c4/tokenized/` with train and validation splits.

#### Step 2: Train with Offline Data

```bash
cd CoLA
torchrun --nproc_per_node=1 main.py \
    --model_config baseline_configs/llama_60m.json \
    --model_type llama \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --offline_mode \
    --offline_data_path ../datasets/c4/tokenized
```

**Benefits of offline mode:**
- ✅ No internet required during training
- ✅ Faster data loading
- ✅ Reproducible datasets
- ✅ Suitable for air-gapped servers

See [scripts/OFFLINE_MODE_INSTRUCTIONS.md](scripts/OFFLINE_MODE_INSTRUCTIONS.md) for detailed instructions.

## Quick Start

### Test Run (Single GPU, 10 steps)

```bash
cd CoLA

# Test with a small model
torchrun --nproc_per_node=1 main.py \
    --model_config baseline_configs/llama_60m.json \
    --model_type llama \
    --batch_size 4 \
    --total_batch_size 16 \
    --num_training_steps 10 \
    --eval_every 5 \
    --save_dir ./test_run \
    --single_gpu
```

## Training

All training scripts are located in `CoLA/scripts/`. You can run them from the `CoLA/` directory.

### Baseline Models

Train standard LLaMA models without low-rank adapters:

```bash
cd CoLA

# 60M parameters
bash scripts/baseline_scripts/baseline60m.sh

# 130M parameters
bash scripts/baseline_scripts/baseline130m.sh
```

**Configuration files**: `CoLA/baseline_configs/llama_{size}.json`

### CoLA Models

Train with Chain of Low-rank Adapters:

```bash
cd CoLA

# 60M parameters
bash scripts/cola_scripts/cola60m.sh

# 130M parameters
bash scripts/cola_scripts/cola130m.sh

# 350M parameters
bash scripts/cola_scripts/cola350m.sh

# 1B parameters
bash scripts/cola_scripts/cola1b.sh

# 7B parameters
bash scripts/cola_scripts/cola7b.sh
```

**Configuration files**: `CoLA/cola_configs/cola_{size}.json`

### CoLA-M Models

Train with memory-efficient CoLA (includes gradient checkpointing):

```bash
cd CoLA

# 60M parameters
bash scripts/cola_m_scripts/colam60m.sh

# 130M parameters
bash scripts/cola_m_scripts/colam130m.sh

# 350M parameters
bash scripts/cola_m_scripts/colam350m.sh

# 1B parameters
bash scripts/cola_m_scripts/colam1b.sh

# 7B parameters
bash scripts/cola_m_scripts/colam7b.sh
```

**Configuration files**: `CoLA/cola_configs/colam_{size}.json`

## Offline Mode

For detailed offline mode setup and troubleshooting, see [scripts/OFFLINE_MODE_INSTRUCTIONS.md](scripts/OFFLINE_MODE_INSTRUCTIONS.md).

### Quick Reference

```bash
# 1. Download dataset
python scripts/download_c4_offline.py --target_tokens 10000000

# 2. Verify download
ls -la datasets/c4/tokenized/
cat datasets/c4/tokenized/dataset_info.json

# 3. Train with offline data
cd CoLA
torchrun --nproc_per_node=1 main.py \
    --model_config baseline_configs/llama_60m.json \
    --model_type llama \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --offline_mode \
    --offline_data_path ../datasets/c4/tokenized \
    --single_gpu
```

## Configuration

### Model Configurations

Model configurations are JSON files that specify architecture parameters:

```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 512,
  "intermediate_size": 1376,
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "vocab_size": 32000,
  ...
}
```

### Training Arguments

Key arguments for `main.py`:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_config` | Path to model config JSON | Required |
| `--model_type` | Model type: `llama`, `cola`, or `cola_m` | `cola` |
| `--batch_size` | Per-device batch size | Required |
| `--total_batch_size` | Global batch size (with gradient accumulation) | Required |
| `--lr` | Learning rate | `1e-4` |
| `--num_training_steps` | Number of update steps | `10000` |
| `--warmup_steps` | Warmup steps | `1000` |
| `--eval_every` | Evaluation frequency | `5000` |
| `--save_every` | Checkpoint saving frequency | `20000` |
| `--offline_mode` | Use pre-downloaded dataset | `False` |
| `--offline_data_path` | Path to offline dataset | `datasets/c4/tokenized` |
| `--single_gpu` | Disable DDP for single GPU | `False` |
| `--activation_checkpointing` | Enable gradient checkpointing | `False` |

View all arguments:
```bash
cd CoLA
python main.py --help
```

## Model Sizes

Available model configurations:

| Model | Parameters | Config File |
|-------|-----------|-------------|
| LLaMA-60M | 60M | `llama_60m.json` |
| LLaMA-130M | 130M | `llama_130m.json` |
| LLaMA-350M | 350M | `llama_350m.json` |
| LLaMA-1B | 1B | `llama_1b.json` |
| LLaMA-7B | 7B | `llama_7b.json` |

Configurations available in:
- Baseline: `CoLA/baseline_configs/`
- CoLA: `CoLA/cola_configs/cola_*.json`
- CoLA-M: `CoLA/cola_configs/colam_*.json`

## Multi-GPU Training

Use `torchrun` for distributed training:

```bash
# 4 GPUs
torchrun --nproc_per_node=4 main.py \
    --model_config baseline_configs/llama_130m.json \
    --model_type llama \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 10000

# 8 GPUs
torchrun --nproc_per_node=8 main.py \
    --model_config cola_configs/cola_1b.json \
    --model_type cola \
    --batch_size 16 \
    --total_batch_size 1024 \
    --num_training_steps 50000
```

## Checkpointing

Models are saved to `--save_dir` at intervals specified by `--save_every`.

### Resume Training

```bash
torchrun --nproc_per_node=1 main.py \
    --model_config baseline_configs/llama_60m.json \
    --model_type llama \
    --continue_from ./checkpoints/model_5000 \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 10000
```

## Monitoring

Training metrics are logged to Weights & Biases (wandb):

- Loss
- Learning rate
- Throughput (tokens/sec, examples/sec)
- Gradient norm
- Memory usage
- Evaluation perplexity

Configure wandb project with `--wandb_project` argument.

## Citation

If you use this code, please cite:

```bibtex
@article{cola2024,
  title={Chain of Low-rank Adapters for Efficient Language Model Training},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

This implementation is based on the GaLore optimizer and uses the C4 dataset from AllenAI.