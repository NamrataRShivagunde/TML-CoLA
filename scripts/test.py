#!/usr/bin/env python3

from datasets import load_dataset


def main():
	# Basic streaming from Hugging Face C4
	ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

	# Optional: shuffle the stream (seeded)
	ds = ds.shuffle(seed=42)

	# Iterate a few examples to verify loading
	for i, ex in enumerate(ds):
		text = ex.get("text", "").replace("\n", " ")
		print(f"[{i}] ", text[:120])
		if i >= 10:
			break


if __name__ == "__main__":
	main()
