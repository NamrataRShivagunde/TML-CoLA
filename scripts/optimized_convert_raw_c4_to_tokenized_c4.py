#!/usr/bin/env python3
import os
import argparse
import json
import glob
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, Features, Sequence, Value
from transformers import AutoTokenizer
from datasets import concatenate_datasets


# =============================================================
# Utility: Load / Save Progress
# =============================================================

def load_progress(out_dir, split):
    """Load saved progress if exists, else create default state."""
    path = os.path.join(out_dir, f"progress_{split}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "total_tokens": 0,
        "total_examples": 0,
        "next_shard_idx": 0,
        "finished": False,
    }


def save_progress(out_dir, split, state):
    path = os.path.join(out_dir, f"progress_{split}.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# =============================================================
# Incremental Shard Saving
# =============================================================

def save_shard(examples, path):
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
    })
    ds = Dataset.from_list(examples, features=features)
    ds.save_to_disk(path)


# =============================================================
# Process Stream with Resume Support
# =============================================================

def process_stream(
    split, tokenizer, max_length, target_tokens,
    batch_size, shard_size, out_dir, input_dir
):

    pad_id = tokenizer.pad_token_id

    # ---- Load previous progress ----
    progress = load_progress(out_dir, split)

    if progress["finished"]:
        print(f"{split}: Already finished. Skipping.")
        return progress

    total_tokens = progress["total_tokens"]
    total_examples = progress["total_examples"]
    next_shard = progress["next_shard_idx"]

    print(f"\n=== {split.upper()} RESUME STATUS ===")
    print(f"Already processed tokens: {total_tokens}")
    print(f"Already processed examples: {total_examples}")
    print(f"Next shard to write: {next_shard}")
    print("=====================================\n")


    # Load from local files - FIXED
    #for debugging
    # if split == "train":
    #     data_files = [
    #         os.path.join(input_dir, "en", "c4-train.00001-of-01024.json.gz"),
    #         os.path.join(input_dir, "en", "c4-train.00002-of-01024.json.gz"),
    #     ]
    # else:  # validation
    #     data_files = [
    #         os.path.join(input_dir, "en", "c4-validation.00000-of-00008.json.gz"),
    #     ]

    # # Load from local files - FIXED to read all files
    if split == "train":
        data_files = sorted(glob.glob(os.path.join(input_dir, "en", "c4-train.*.json.gz")))
    else:  # validation
        data_files = sorted(glob.glob(os.path.join(input_dir, "en", "c4-validation.*.json.gz")))
 
    print(f"Loading from files: {data_files}")
    
    stream = load_dataset(
        "json",
        data_files=data_files,
        split="train",
        streaming=True
    )
    # load and shuffle stream
    # 
    stream = stream.shuffle(seed=42)

    batch_texts = []
    shard_examples = []
    examples_skipped = 0

    for ex in stream:
        # Skip already processed examples
        if examples_skipped < total_examples:
            examples_skipped += 1
            continue

        if total_tokens >= target_tokens:
            break

        text = ex.get("text", "")
        if not text:
            continue

        batch_texts.append(text)

        # Only process at batch size
        if len(batch_texts) < batch_size:
            continue

        # ---- Batch tokenize ----
        tok = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
        )

        batch_texts = []

        input_ids = np.array(tok["input_ids"])
        masks = np.array(tok["attention_mask"])

        non_pad_counts = np.count_nonzero(input_ids != pad_id, axis=1)

        for ids, att, n_pad in zip(input_ids, masks, non_pad_counts):
            # if total_tokens >= target_tokens:
            #     break

            shard_examples.append(
                {
                    "input_ids": ids.tolist(),
                    "attention_mask": att.tolist(),
                }
            )

            total_tokens += int(n_pad)
            total_examples += 1

            # ---- When shard is full: SAVE IMMEDIATELY ----
            if len(shard_examples) >= shard_size:
                shard_path = os.path.join(out_dir, f"{split}_shard_{next_shard}")
                save_shard(shard_examples, shard_path)
                print(f"Saved {split} shard {next_shard} ({len(shard_examples)} examples)")

                next_shard += 1
                shard_examples = []

                # ---- Save progress state ----
                save_progress(out_dir, split, {
                    "total_tokens": total_tokens,
                    "total_examples": total_examples,
                    "next_shard_idx": next_shard,
                    "finished": False,
                })

    # ---- Save final partial shard ----
    if shard_examples:
        shard_path = os.path.join(out_dir, f"{split}_shard_{next_shard}")
        save_shard(shard_examples, shard_path)
        print(f"Saved FINAL {split} shard {next_shard}")
        next_shard += 1

    # ---- Mark as finished ----
    save_progress(out_dir, split, {
        "total_tokens": total_tokens,
        "total_examples": total_examples,
        "next_shard_idx": next_shard,
        "finished": True,
    })

    return {
        "total_tokens": total_tokens,
        "total_examples": total_examples,
        "next_shard_idx": next_shard,
        "finished": True,
    }


# =============================================================
# Assemble All Shards
# =============================================================

def assemble_dataset(out_dir, split, num_shards):
    shards = []
    for i in range(num_shards):
        shard_path = os.path.join(out_dir, f"{split}_shard_{i}")
        shards.append(Dataset.load_from_disk(shard_path))
    return concatenate_datasets(shards)


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="datasets/c4/raw")
    parser.add_argument("--output_dir", type=str, default="datasets/c4/tokenized")
    parser.add_argument("--target_tokens", type=int, default=60_000_000)
    parser.add_argument("--val_tokens", type=int, default=100_000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="t5-base")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------
    # Tokenizer
    # -----------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # -----------------------------------------
    # TRAIN
    # -----------------------------------------
    train_state = process_stream(
        split="train",
        tokenizer=tokenizer,
        max_length=args.max_length,
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        out_dir=args.output_dir,
        input_dir=args.input_dir, 
    )

    # -----------------------------------------
    # VALIDATION
    # -----------------------------------------
    val_state = process_stream(
        split="validation",
        tokenizer=tokenizer,
        max_length=args.max_length,
        target_tokens=args.val_tokens,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        out_dir=args.output_dir,
        input_dir=args.input_dir, 
    )

    # -----------------------------------------
    # Assemble final Datasets
    # -----------------------------------------

    train_ds = assemble_dataset(args.output_dir, "train", train_state["next_shard_idx"])
    val_ds = assemble_dataset(args.output_dir, "validation", val_state["next_shard_idx"])

    final_path = args.output_dir

    # Save final dataset
    DatasetDict({"train": train_ds, "validation": val_ds}).save_to_disk(final_path)

    print("\nCleaning up shards...")
    # Delete all shard folders & progress files
    for filename in os.listdir(final_path):
        if filename.startswith("train_shard_") or filename.startswith("validation_shard_"):
            shard_path = os.path.join(final_path, filename)
            print(f"Removing {shard_path}")
            os.system(f"rm -rf '{shard_path}'")

    # Remove progress files
    for split in ["train", "validation"]:
        prog_path = os.path.join(final_path, f"progress_{split}.json")
        if os.path.exists(prog_path):
            print(f"Removing {prog_path}")
            os.remove(prog_path)


    # DatasetDict({"train": train_ds, "validation": val_ds}).save_to_disk(args.output_dir)

    # Metadata
    meta = {
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "train_non_pad_tokens": train_state["total_tokens"],
        "train_examples": train_state["total_examples"],
        "val_non_pad_tokens": val_state["total_tokens"],
        "val_examples": val_state["total_examples"],
    }

    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDONE — Dataset saved with resume support.")


if __name__ == "__main__":
    main()
