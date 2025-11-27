#!/usr/bin/env python3
import os
import argparse
import json
from datasets import Dataset, DatasetDict, concatenate_datasets


def assemble_dataset(shard_dir, split, start_idx, end_idx):
    """Assemble shards from start_idx to end_idx (inclusive)"""
    shards = []
    print(f"\nAssembling {split} shards {start_idx} to {end_idx}...")
    
    for i in range(start_idx, end_idx + 1):
        shard_path = os.path.join(shard_dir, f"{split}_shard_{i}")
        if not os.path.exists(shard_path):
            print(f"Warning: Shard {i} not found at {shard_path}, skipping...")
            continue
        print(f"Loading shard {i}...")
        shards.append(Dataset.load_from_disk(shard_path))
    
    if not shards:
        raise ValueError(f"No shards found for {split} in range {start_idx}-{end_idx}")
    
    print(f"Concatenating {len(shards)} shards...")
    return concatenate_datasets(shards)


def main():
    parser = argparse.ArgumentParser(description="Assemble tokenized shards into a final dataset")
    parser.add_argument("--shard_dir", type=str, required=True, help="Directory containing shards")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for final dataset")
    parser.add_argument("--train_start", type=int, default=0, help="First train shard index")
    parser.add_argument("--train_end", type=int, required=True, help="Last train shard index (inclusive)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Assemble train dataset
    train_ds = assemble_dataset(args.shard_dir, "train", args.train_start, args.train_end)
    print(f"Train dataset: {len(train_ds)} examples")

    # Save final dataset
    print(f"\nSaving final dataset to {args.output_dir}...")
    DatasetDict({"train": train_ds}).save_to_disk(args.output_dir)

    # Save metadata
    meta = {
        "train_shards": f"{args.train_start}-{args.train_end}",
        "train_examples": len(train_ds),
    }

    with open(os.path.join(args.output_dir, "train_set_assembly_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ DONE — Train Dataset assembled and saved.")


if __name__ == "__main__":
    main()
