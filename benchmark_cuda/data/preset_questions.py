"""Create preset evaluation questions from test split."""

import json
import os
from collections import defaultdict
from typing import Optional


TARGET_PER_CATEGORY = 10
MIN_TOTAL = 50


def create_preset_questions(
    data_dir: str = "./data/processed",
    eval_dir: str = "./data/eval",
    seed: int = 42,
    force: bool = False,
) -> str:
    """Sample ~70 questions (10/category) from test split."""

    output_path = os.path.join(eval_dir, "preset_questions.jsonl")
    if os.path.exists(output_path) and not force:
        return output_path

    os.makedirs(eval_dir, exist_ok=True)

    # Load test split
    test_path = os.path.join(data_dir, "test.jsonl")
    records = []
    with open(test_path) as f:
        for line in f:
            records.append(json.loads(line))

    # Group by category
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)

    # Sample up to TARGET_PER_CATEGORY per category
    import random
    rng = random.Random(seed)

    selected = []
    category_counts = {}
    for cat in sorted(by_category.keys()):
        pool = by_category[cat]
        rng.shuffle(pool)
        n = min(TARGET_PER_CATEGORY, len(pool))
        selected.extend(pool[:n])
        category_counts[cat] = n

    # Add IDs
    questions = []
    for i, rec in enumerate(selected):
        questions.append({
            "id": i,
            "category": rec["category"],
            "instruction": rec["instruction"],
            "input": rec["input"],
            "reference_output": rec["output"],
        })

    # Save
    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    # Save distribution log
    dist_path = os.path.join(eval_dir, "preset_distribution.json")
    with open(dist_path, "w") as f:
        json.dump({
            "total_questions": len(questions),
            "category_counts": category_counts,
            "categories": sorted(by_category.keys()),
        }, f, indent=2)

    return output_path
