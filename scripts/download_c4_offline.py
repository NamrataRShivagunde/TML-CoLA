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
#from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


def gather_examples(streaming_split, tokenizer, max_length, target_tokens, batch_texts_size=512):
    """Stream a split from C4, tokenize in batches and collect tokenized examples until target_tokens reached.

    Returns a list of tokenized dicts: {"input_ids": [...], "attention_mask": [...]}
    and the number of non-pad tokens gathered.
    """
    input_ids_list = []
    attention_mask_list = []
    total_tokens = 0

    stream = load_dataset("allenai/c4", "en", split=streaming_split, streaming=True)

    batch_texts = []
    #pbar = tqdm(unit="examples", desc=f"Collecting from {streaming_split}")
    pad_id = tokenizer.pad_token_id

    for ex in stream:
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
            )

            for ids, att in zip(tok["input_ids"], tok["attention_mask"]):
                # count non-pad tokens
                non_pad = sum(1 for _id in ids if _id != pad_id)
                total_tokens += non_pad
                input_ids_list.append(list(ids))
                attention_mask_list.append(list(att))
                #pbar.update(1)
                if total_tokens >= target_tokens:
                    break
            batch_texts = []
            if total_tokens >= target_tokens:
                break

    # Process remaining batch_texts if still under target
    if total_tokens < target_tokens and len(batch_texts) > 0:
        tok = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
        )
        for ids, att in zip(tok["input_ids"], tok["attention_mask"]):
            non_pad = sum(1 for _id in ids if _id != pad_id)
            total_tokens += non_pad
            input_ids_list.append(list(ids))
            attention_mask_list.append(list(att))
            #pbar.update(1)
            if total_tokens >= target_tokens:
                break

    #pbar.close()
    tokenized_dict = {"input_ids": input_ids_list, "attention_mask": attention_mask_list}
    return tokenized_dict, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Download and tokenize C4 for offline mode")
    parser.add_argument("--output_dir", type=str, default="datasets/c4/tokenized", help="Directory to save tokenized dataset")
    parser.add_argument("--target_tokens", type=int, default=10_000_000, help="Target non-pad tokens for the training split (default: 10,000,000)")
    parser.add_argument("--val_tokens", type=int, default=100_000, help="Target non-pad tokens for validation split (default: 100,000)")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenization max length")
    parser.add_argument("--tokenizer", type=str, default="t5-base", help="Tokenizer model name")
    parser.add_argument("--batch_texts_size", type=int, default=512, help="Number of raw texts to tokenise in a batch")
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
    train_dict, train_tokens = gather_examples(
        streaming_split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        target_tokens=args.target_tokens,
        batch_texts_size=args.batch_texts_size,
    )
    print(f"Collected {train_tokens} non-pad tokens for training and {len(train_dict['input_ids'])} examples")

    print(f"Collecting ~{args.val_tokens} tokens for validation")
    val_dict, val_tokens = gather_examples(
        streaming_split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        target_tokens=args.val_tokens,
        batch_texts_size=args.batch_texts_size,
    )
    print(f"Collected {val_tokens} non-pad tokens for validation and {len(val_dict['input_ids'])} examples")

    print("Building datasets and saving to disk...")
    train_ds = Dataset.from_dict(train_dict)
    val_ds = Dataset.from_dict(val_dict)
    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    ds_dict.save_to_disk(args.output_dir)

    # Save a small metadata file
    meta = {
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "train_non_pad_tokens": train_tokens,
        "train_examples": len(train_dict["input_ids"]),
        "val_non_pad_tokens": val_tokens,
        "val_examples": len(val_dict["input_ids"]),
    }
    import json

    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved tokenized dataset to {args.output_dir}")
    print("Done")


if __name__ == "__main__":
    main()
