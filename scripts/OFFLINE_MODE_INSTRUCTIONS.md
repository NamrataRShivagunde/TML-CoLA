# Offline Mode Instructions for C4 Dataset

This guide explains how to download a tokenized C4 dataset for offline training and use it with `main.py`.

## Step 1: Download and Tokenize C4 Data

Run the download script on your server. The script will:
- Stream the C4 dataset from HuggingFace
- Tokenize using the T5 tokenizer (same as training)
- Collect ~10M tokens for training and ~100k for validation
- Save to `datasets/c4/tokenized/`

### Quick Test (Small Dataset)
First, test with a small dataset to ensure everything works:

```bash
cd /Users/nammu/code/TML-CoLA
python scripts/download_c4_offline.py --target_tokens 50000 --val_tokens 5000
```

This will create `datasets/c4/tokenized/` with a small dataset (~50k tokens).

### Full Download (10M tokens - for actual training)
Once the test works, download the full dataset:

```bash
python scripts/download_c4_offline.py --target_tokens 10000000 --val_tokens 100000
```

**Expected time:** 10-30 minutes depending on network speed and server performance.

**Expected output structure:**
```
datasets/c4/tokenized/
├── train/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── validation/
│   ├── data-00000-of-00001.arrow
│   └── state.json
├── dataset_dict.json
└── dataset_info.json
```

### Custom Options

```bash
# Custom output directory
python scripts/download_c4_offline.py \
    --output_dir /path/to/custom/location \
    --target_tokens 10000000 \
    --val_tokens 100000

# Different max_length (must match training config)
python scripts/download_c4_offline.py \
    --target_tokens 10000000 \
    --max_length 512

# Overwrite existing dataset
python scripts/download_c4_offline.py --overwrite
```

## Step 2: Test the Dataset with main.py

Now test loading the offline dataset with a quick training run:

### Minimal Test Run

```bash
cd /Users/nammu/code/TML-CoLA/CoLA

# Test with small model and few steps
torchrun --nproc_per_node=1 main.py \
    --model_config baseline_configs/llama_60m.json \
    --model_type llama \
    --batch_size 4 \
    --total_batch_size 16 \
    --max_length 256 \
    --num_training_steps 10 \
    --eval_every 5 \
    --save_every 10000 \
    --save_dir ./test_offline_run \
    --offline_mode \
    --offline_data_path ../datasets/c4/tokenized \
    --single_gpu
```

**What to check:**
- Script should start without errors
- You should see: `Loading tokenized data from disk: ../datasets/c4/tokenized`
- Training should proceed normally
- Evaluation should work at step 5

### Full Training Run

Once the test works, run full training:

```bash
# Single GPU
torchrun --nproc_per_node=1 main.py \
    --model_config baseline_configs/llama_130m.json \
    --model_type llama \
    --batch_size 32 \
    --total_batch_size 512 \
    --max_length 256 \
    --num_training_steps 10000 \
    --eval_every 1000 \
    --save_every 2000 \
    --save_dir ./checkpoints/llama_130m_offline \
    --offline_mode \
    --offline_data_path ../datasets/c4/tokenized \
    --single_gpu

# Multi-GPU (4 GPUs example)
torchrun --nproc_per_node=4 main.py \
    --model_config baseline_configs/llama_130m.json \
    --model_type llama \
    --batch_size 32 \
    --total_batch_size 512 \
    --max_length 256 \
    --num_training_steps 10000 \
    --eval_every 1000 \
    --save_every 2000 \
    --save_dir ./checkpoints/llama_130m_offline \
    --offline_mode \
    --offline_data_path ../datasets/c4/tokenized
```

## Step 3: Verify Dataset Info

Check the metadata of your downloaded dataset:

```bash
cat datasets/c4/tokenized/dataset_info.json
```

You should see something like:
```json
{
  "tokenizer": "t5-base",
  "max_length": 256,
  "train_non_pad_tokens": 10000000,
  "train_examples": 50000,
  "val_non_pad_tokens": 100000,
  "val_examples": 500
}
```

## Troubleshooting

### Issue: "Module not found" errors
Make sure you have the required packages:
```bash
pip install datasets transformers torch tqdm loguru
```

### Issue: Permission denied for `/datasets/c4/`
The default output is `datasets/c4/tokenized` (relative path in your repo), not `/datasets/c4/tokenized` (absolute path).

### Issue: Out of memory during download
Reduce batch size:
```bash
python scripts/download_c4_offline.py --batch_texts_size 256
```

### Issue: Dataset loading fails in main.py
Verify the path is correct:
```bash
ls -la datasets/c4/tokenized/
```

Make sure to use relative or absolute path correctly in `--offline_data_path`.

## Notes

- **Token count**: The script counts non-padding tokens, so actual examples will vary based on text length
- **Reproducibility**: The dataset is shuffled with seed=42 for reproducibility
- **Disk space**: ~10M tokens = ~50k examples at max_length=256 ≈ 200-400 MB on disk
- **Network**: Initial download requires internet; subsequent training runs are fully offline
- **Reusability**: Once downloaded, the dataset can be reused for multiple training runs

## Summary Commands

```bash
# 1. Download dataset (run on server with internet)
python scripts/download_c4_offline.py --target_tokens 10000000

# 2. Quick test (single GPU, few steps)
cd CoLA
torchrun --nproc_per_node=1 main.py \
    --model_config baseline_configs/llama_60m.json \
    --model_type llama \
    --batch_size 4 \
    --total_batch_size 16 \
    --num_training_steps 10 \
    --offline_mode \
    --offline_data_path ../datasets/c4/tokenized \
    --single_gpu \
    --save_dir ./test_offline

# 3. Check everything worked
ls -la ../datasets/c4/tokenized/
cat ../datasets/c4/tokenized/dataset_info.json
```
