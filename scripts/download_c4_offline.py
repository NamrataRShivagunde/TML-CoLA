#!/usr/bin/env python3
"""
Script to download and tokenise a portion of the allenai/c4 dataset for offline training.

This script streams C4, tokenizes with the T5 tokenizer, accumulates examples until
`target_tokens` non-padding tokens are collected for the training split, and similarly
creates a small validation split. The resulting dataset is saved with
`datasets.DatasetDict.save_to_disk()` and can be loaded by `main.py` using
`datasets.load_from_disk("/path/to/output")`.

Default output directory: ./datasets/c4/tokenized (change with --output_dir).

Example:
  python scripts/download_c4_offline.py --target_tokens 10000000 --output_dir ./datasets/c4/tokenized

Notes:
- The default tokenizer is `t5-base` to match `main.py`'s tokenizer.
- The script performs batch tokenisation for speed. It counts non-pad tokens
  (so padding doesn't inflate the token count) when checking the target.
- For testing, run with a small `--target_tokens` (e.g. 10000) to validate.
"""

import os
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
import shutil


def save_shard(examples, shard_path):
    """Save a shard of tokenized examples to disk using HF datasets."""
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
    })
    ds = Dataset.from_list(examples, features=features)
    ds.save_to_disk(shard_path)


def gather_examples(streaming_split, tokenizer, max_length, target_tokens, batch_texts_size, output_dir, shard_size):
    """Stream C4 split, tokenize in batches, save shards incrementally until target_tokens reached.

    Returns: (total_tokens, total_examples, next_shard_idx)
    """
    total_tokens = 0
    total_examples = 0
    next_shard = 0

    stream = load_dataset("allenai/c4", "en", split=streaming_split, streaming=True)
    print(f"Shuffling {streaming_split} split with seed 42")
    stream = stream.shuffle(seed=42)

    batch_texts = []
    shard_examples = []
    pad_id = tokenizer.pad_token_id

    pbar = tqdm(total=target_tokens, desc=f"{streaming_split} tokens", unit="tok")
    pbar.update(0)

    for ex in stream:
        if total_tokens >= target_tokens:
            break

        text = ex.get("text", "")
        if not text:
            continue
        batch_texts.append(text)

        if len(batch_texts) >= batch_texts_size:
            tok = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="np",
            )

            input_ids = tok["input_ids"]  # (B, max_length)
            attn_mask = tok["attention_mask"]  # (B, max_length)
            non_pad_counts = np.count_nonzero(input_ids != pad_id, axis=1)

            for ids, att, n_pad in zip(input_ids, attn_mask, non_pad_counts):
                if total_tokens >= target_tokens:
                    break
                shard_examples.append({
                    "input_ids": ids.tolist(),
                    "attention_mask": att.tolist(),
                })
                total_tokens += int(n_pad)
                total_examples += 1
                pbar.update(int(n_pad))

                if len(shard_examples) >= shard_size:
                    shard_path = os.path.join(output_dir, f"{streaming_split}_shard_{next_shard}")
                    save_shard(shard_examples, shard_path)
                    print(f"Saved {streaming_split} shard {next_shard} ({len(shard_examples)} examples)")
                    next_shard += 1
                    shard_examples = []

            batch_texts = []

    # Save final partial shard
    if shard_examples:
        shard_path = os.path.join(output_dir, f"{streaming_split}_shard_{next_shard}")
        save_shard(shard_examples, shard_path)
        print(f"Saved FINAL {streaming_split} shard {next_shard} ({len(shard_examples)} examples)")
        next_shard += 1

    pbar.close()
    return total_tokens, total_examples, next_shard


def main():
    parser = argparse.ArgumentParser(description="Download and tokenize C4 for offline mode")
    parser.add_argument("--output_dir", type=str, default="datasets/c4/tokenized", help="Directory to save tokenized dataset")
    parser.add_argument("--target_tokens", type=int, default=10_000_000, help="Target non-pad tokens for the training split (default: 10,000,000)")
    parser.add_argument("--val_tokens", type=int, default=100_000, help="Target non-pad tokens for validation split (default: 100,000)")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenization max length")
    parser.add_argument("--tokenizer", type=str, default="t5-base", help="Tokenizer model name")
    parser.add_argument("--batch_texts_size", type=int, default=2048, help="Number of raw texts to tokenise in a batch")
    parser.add_argument("--shard_size", type=int, default=5000, help="Number of examples per shard")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output_dir if present")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.output_dir, "dataset_info.json")) and not args.overwrite:
        print(f"Found existing dataset at {args.output_dir}. Use --overwrite to replace.")
        return

    print(f"Loading tokenizer {args.tokenizer} (this may download model files)...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        # Ensure there is a pad token (t5-base has it, but be safe)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    print(f"Collecting ~{args.target_tokens} tokens for training (max_length={args.max_length})")
    train_tokens, train_examples, train_shards = gather_examples(
        streaming_split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        target_tokens=args.target_tokens,
        batch_texts_size=args.batch_texts_size,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
    )
    print(f"Collected {train_tokens} non-pad tokens for training across {train_examples} examples and {train_shards} shards")

    print(f"Collecting ~{args.val_tokens} tokens for validation")
    val_tokens, val_examples, val_shards = gather_examples(
        streaming_split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        target_tokens=args.val_tokens,
        batch_texts_size=args.batch_texts_size,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
    )
    print(f"Collected {val_tokens} non-pad tokens for validation across {val_examples} examples and {val_shards} shards")

    print("Assembling shards into final datasets...")
    # Assemble train shards
    train_parts = []
    for i in range(train_shards):
        shard_path = os.path.join(args.output_dir, f"train_shard_{i}")
        train_parts.append(Dataset.load_from_disk(shard_path))
    train_ds = concatenate_datasets(train_parts) if train_parts else Dataset.from_dict({"input_ids": [], "attention_mask": []})

    # Assemble validation shards
    val_parts = []
    for i in range(val_shards):
        shard_path = os.path.join(args.output_dir, f"validation_shard_{i}")
        val_parts.append(Dataset.load_from_disk(shard_path))
    val_ds = concatenate_datasets(val_parts) if val_parts else Dataset.from_dict({"input_ids": [], "attention_mask": []})

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})
    ds_dict.save_to_disk(args.output_dir)

    # Save a small metadata file
    meta = {
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "train_non_pad_tokens": train_tokens,
        "train_examples": train_examples,
        "val_non_pad_tokens": val_tokens,
        "val_examples": val_examples,
    }
    import json

    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    import time
    time.sleep(5)

    print(f"Saved tokenized dataset to {args.output_dir}")
    import subprocess
    # Optional cleanup: remove shard folders now that final dataset is saved
    for name in os.listdir(args.output_dir):
        if name.startswith("train_shard_") or name.startswith("validation_shard_"):
            shard_path = os.path.join(args.output_dir, name)
            print(f"Force removing {shard_path}")
            # change permission for shard_path to do all operations
            chmod_command = ["chmod", "-R", "777", shard_path]
            subprocess.run(chmod_command, check=True)
            subprocess.run(["rm", "-rf", shard_path], check=True)
    
            

    print("Done")


if __name__ == "__main__":
    main()
