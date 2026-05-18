#!/usr/bin/env python3
"""Create derived BEEG train/eval splits with more frames examples.

The source Hugging Face dataset is left untouched. This script uses Arrow
select/concatenate operations instead of materializing every large context into
Python dictionaries, then writes local parquet files for the RLM RLVR
environment's `data_paths` mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset


DEFAULT_DATASET_ID = "lsteno/BEEG-agents"
DEFAULT_OUTPUT_DIR = Path("data/beeg_agents_balanced_35_40_25_frames40_v1")
FAMILIES = ("oolong", "frames", "longcodeu")
TARGET_FRACTIONS = {"oolong": 0.35, "frames": 0.40, "longcodeu": 0.25}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-destructive balanced BEEG train/eval parquet splits."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--sft-split", default="sft_traces")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def target_counts(total: int) -> dict[str, int]:
    frames = int(round(total * TARGET_FRACTIONS["frames"]))
    oolong = int(round(total * TARGET_FRACTIONS["oolong"]))
    longcodeu = total - frames - oolong
    return {"oolong": oolong, "frames": frames, "longcodeu": longcodeu}


def family_indices(dataset: Dataset) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {family: [] for family in FAMILIES}
    for index, family in enumerate(dataset["dataset"]):
        if family not in grouped:
            raise ValueError(f"Unexpected BEEG dataset family {family!r} at row {index}")
        grouped[str(family)].append(index)
    return grouped


def shuffled(indices: list[int], *, rng: random.Random) -> list[int]:
    copied = list(indices)
    rng.shuffle(copied)
    return copied


def select_or_empty(dataset: Dataset, indices: list[int]) -> Dataset | None:
    if not indices:
        return None
    return dataset.select(indices)


def concat_nonempty(parts: list[Dataset | None]) -> Dataset:
    nonempty = [part for part in parts if part is not None and len(part) > 0]
    if not nonempty:
        raise ValueError("Cannot concatenate an empty dataset list.")
    if len(nonempty) == 1:
        return nonempty[0]
    return concatenate_datasets(nonempty)


def rebalance_split(
    *,
    name: str,
    source: Dataset,
    sft: Dataset,
    sft_pool: dict[str, list[int]],
    rng: random.Random,
    shuffle_seed: int,
) -> tuple[Dataset, dict[str, list[int]], dict[str, Any]]:
    targets = target_counts(len(source))
    grouped = family_indices(source)
    kept_source_indices: list[int] = []
    displaced_source_indices: dict[str, list[int]] = {family: [] for family in FAMILIES}
    added_sft_indices: list[int] = []
    moves: dict[str, dict[str, int]] = {}

    for family in FAMILIES:
        indices = shuffled(grouped[family], rng=rng)
        target = targets[family]
        if len(indices) > target:
            kept_source_indices.extend(indices[:target])
            displaced_source_indices[family].extend(indices[target:])
            moves[family] = {"kept": target, "removed": len(indices) - target, "added": 0}
        elif len(indices) < target:
            needed = target - len(indices)
            if len(sft_pool[family]) < needed:
                raise ValueError(
                    f"Not enough {family} rows in sft pool for {name}: "
                    f"need {needed}, have {len(sft_pool[family])}"
                )
            take = sft_pool[family][:needed]
            del sft_pool[family][:needed]
            kept_source_indices.extend(indices)
            added_sft_indices.extend(take)
            moves[family] = {"kept": len(indices), "removed": 0, "added": needed}
        else:
            kept_source_indices.extend(indices)
            moves[family] = {"kept": len(indices), "removed": 0, "added": 0}

    balanced = concat_nonempty(
        [select_or_empty(source, kept_source_indices), select_or_empty(sft, added_sft_indices)]
    ).shuffle(seed=shuffle_seed)
    final_counts = dict(Counter(balanced["dataset"]))
    summary = {
        "split": name,
        "total": len(balanced),
        "target_counts": targets,
        "final_counts": final_counts,
        "moves": moves,
        "added_from_sft_ids": list(sft.select(added_sft_indices)["id"]) if added_sft_indices else [],
        "removed_to_sft_ids": {
            family: list(source.select(indices)["id"]) if indices else []
            for family, indices in displaced_source_indices.items()
        },
    }
    return balanced, displaced_source_indices, summary


def assert_no_overlap(splits: dict[str, Dataset]) -> None:
    owners: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for split, dataset in splits.items():
        for row_id in dataset["id"]:
            previous = owners.get(row_id)
            if previous is not None:
                duplicates.append((row_id, previous, split))
            owners[row_id] = split
    if duplicates:
        raise ValueError(f"Found duplicate ids across derived splits: {duplicates[:10]}")


def counts_for(dataset: Dataset) -> dict[str, int]:
    return {family: Counter(dataset["dataset"])[family] for family in FAMILIES}


def write_dataset(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path), batch_size=128)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_dataset(args.dataset_id, split=args.train_split)
    eval_ds = load_dataset(args.dataset_id, split=args.eval_split)
    sft = load_dataset(args.dataset_id, split=args.sft_split)

    source_counts = {
        "train": counts_for(train),
        "eval": counts_for(eval_ds),
        "sft_traces": counts_for(sft),
    }

    sft_pool = {family: shuffled(indices, rng=rng) for family, indices in family_indices(sft).items()}
    new_train, train_displaced, train_summary = rebalance_split(
        name="train_balanced",
        source=train,
        sft=sft,
        sft_pool=sft_pool,
        rng=rng,
        shuffle_seed=args.seed + 101,
    )
    new_eval, eval_displaced, eval_summary = rebalance_split(
        name="eval_balanced",
        source=eval_ds,
        sft=sft,
        sft_pool=sft_pool,
        rng=rng,
        shuffle_seed=args.seed + 202,
    )

    new_sft_parts: list[Dataset | None] = []
    for family in FAMILIES:
        new_sft_parts.append(select_or_empty(sft, sft_pool[family]))
        new_sft_parts.append(select_or_empty(train, train_displaced[family]))
        new_sft_parts.append(select_or_empty(eval_ds, eval_displaced[family]))
    new_sft = concat_nonempty(new_sft_parts).shuffle(seed=args.seed + 303)

    derived = {"train": new_train, "eval": new_eval, "sft_traces_remainder": new_sft}
    assert_no_overlap(derived)

    write_dataset(new_train, output_dir / "train.parquet")
    write_dataset(new_eval, output_dir / "eval.parquet")
    write_dataset(new_sft, output_dir / "sft_traces_remainder.parquet")

    split_summaries = {
        "train": train_summary,
        "eval": eval_summary,
        "sft_traces_remainder": {
            "split": "sft_traces_remainder",
            "total": len(new_sft),
            "final_counts": counts_for(new_sft),
        },
    }
    manifest = {
        "dataset_id": args.dataset_id,
        "source_splits": {
            "train": args.train_split,
            "eval": args.eval_split,
            "sft_traces": args.sft_split,
        },
        "output_dir": str(output_dir),
        "seed": args.seed,
        "target_fractions": TARGET_FRACTIONS,
        "source_counts": source_counts,
        "derived_summaries": split_summaries,
        "parquet_paths": {
            "train": str(output_dir / "train.parquet"),
            "eval": str(output_dir / "eval.parquet"),
            "sft_traces_remainder": str(output_dir / "sft_traces_remainder.parquet"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with (output_dir / "split_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "family", "count", "fraction"])
        writer.writeheader()
        for split, dataset in derived.items():
            counts = Counter(dataset["dataset"])
            total = len(dataset)
            for family in FAMILIES:
                writer.writerow(
                    {
                        "split": split,
                        "family": family,
                        "count": counts[family],
                        "fraction": counts[family] / total if total else 0.0,
                    }
                )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
