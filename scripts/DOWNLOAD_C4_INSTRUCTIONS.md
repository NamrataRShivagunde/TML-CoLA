# Offline Mode Instructions for C4 Dataset

This guide explains how to download a tokenized C4 dataset for offline training and use it with `main.py`.

## Step 1: Download and Tokenize C4 Data

Run the download script on your server. The script will:
- Stream the C4 dataset from HuggingFace
- Tokenize using the T5 tokenizer (same as training)
- Save to `datasets/c4/tokenized/`

### Quick Test (Small Dataset)
First, test with a small dataset to ensure everything works:

```bash
cd TML-CoLA
python scripts/download_c4_offline.py --target_tokens 50000 --val_tokens 5000
```

This will create `datasets/c4/tokenized/` with a small dataset (~50k tokens).

### Full Download (e.g. 10M tokens - for actual training)
Once the test works, download the full dataset:

```bash
python scripts/download_c4_offline.py --target_tokens 10000000 --val_tokens 100000

use --overwrite to overwrite data
```

python scripts/optimized_convert_raw_c4_to_tokenized_c4.py --target_tokens 3000000000

## Step 2 - Convert Raw C4 to Tokenized C4 (Large Run)

Use this command to convert locally downloaded raw C4 files into tokenized form:

```bash
python scripts/optimized_convert_raw_c4_to_tokenized_c4.py --target_tokens 3000000000
```

scripts/optimized_convert_raw_c4_to_tokenized_c4_only_val_set.py

This script reads raw local C4 JSON shards from `datasets/c4/raw/en/` and creates a tokenized validation-only dataset.

Example:

```bash
python scripts/optimized_convert_raw_c4_to_tokenized_c4_only_val_set.py \
	--input_dir datasets/c4/raw \
	--output_dir datasets/c4/tokenized \
	--val_tokens 100000
```

Other similar scripts

    TML-CoLA/scripts/optimized_download_c4_offline_withresume.py [Not great though!, need to revisit this script]

- It supports resume with progress tracking if the job stops.
- It writes tokenized shards and assembles a final dataset for offline training.


