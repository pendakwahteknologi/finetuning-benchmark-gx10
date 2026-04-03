"""Dataset preparation: load, normalize, split, save."""

import json
import os
import hashlib
from typing import Optional

from datasets import load_dataset

from .prompt_format import format_train_prompt


DOLLY_COLUMNS = {
    "instruction": "instruction",
    "context": "input",
    "response": "output",
    "category": "category",
}


def normalize_record(record: dict) -> dict:
    return {
        "instruction": record.get("instruction", "").strip(),
        "input": record.get("context", "").strip(),
        "output": record.get("response", "").strip(),
        "category": record.get("category", "").strip(),
    }


def prepare_dataset(
    data_dir: str = "./data/processed",
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    force: bool = False,
) -> dict:
    """Load dolly-15k, normalize, split 90/5/5, save to JSONL."""

    manifest_path = os.path.join(data_dir, "split_manifest.json")
    if os.path.exists(manifest_path) and not force:
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest

    os.makedirs(data_dir, exist_ok=True)

    # Load dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    records = [normalize_record(r) for r in ds]

    # Deterministic shuffle and split
    import random
    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_records = records[:n_train]
    val_records = records[n_train : n_train + n_val]
    test_records = records[n_train + n_val :]

    # Save splits
    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    for name, recs in splits.items():
        path = os.path.join(data_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    # Compute content hash for reproducibility verification
    all_text = "".join(r["instruction"] + r["output"] for r in records)
    content_hash = hashlib.sha256(all_text.encode()).hexdigest()[:16]

    # Category distribution
    from collections import Counter
    train_cats = Counter(r["category"] for r in train_records)
    test_cats = Counter(r["category"] for r in test_records)

    manifest = {
        "dataset": "databricks/databricks-dolly-15k",
        "seed": seed,
        "total_records": n,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "train_categories": dict(train_cats),
        "test_categories": dict(test_cats),
        "content_hash": content_hash,
        "data_dir": data_dir,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
