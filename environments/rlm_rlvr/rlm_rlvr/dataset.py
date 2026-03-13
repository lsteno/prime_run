from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import verifiers as vf
from datasets import Dataset

from .external_rlm import build_initial_messages
from .parsing import parse_answer_candidates

DEFAULT_TRAIN_DATASET_CANDIDATES = [
    Path("/home/coder/prime_run/data/train_with_context_tokens_split_20260313/train_curriculum.parquet"),
    Path("/home/coder/prime_run/data/train_with_context_tokens_split_20260313/train.parquet"),
    Path("/home/coder/rl_training/data/train.parquet"),
    Path("/home/coder/dataset-maker/data/oolong-like/train.parquet"),
]

DEFAULT_EVAL_DATASET_CANDIDATES = [
    Path("/home/coder/prime_run/data/train_with_context_tokens_split_20260313/eval.parquet"),
]


def _resolve_paths(paths: list[str] | None, *, default_candidates: list[Path], label: str) -> list[Path]:
    if paths:
        resolved = [Path(path) for path in paths]
    else:
        resolved = [path for path in default_candidates if path.exists()]
    if not resolved:
        raise FileNotFoundError(
            f"No parquet {label} dataset paths were found. Pass explicit paths to load_environment()."
        )
    return resolved


def _load_frame(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    required = {"prompt", "context", "answer", "dataset", "task"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Parquet dataset is missing required columns: {sorted(missing)}")
    return frame


def _to_dataset(frame: pd.DataFrame) -> Dataset:
    records: list[dict] = []
    for _, row in frame.iterrows():
        answers = parse_answer_candidates(row["answer"])
        if not answers:
            continue
        info = {
            "context": row["context"] if isinstance(row["context"], str) else "",
            "acceptable_answers": answers,
            "dataset_name": str(row["dataset"]),
            "source_task": str(row["task"]),
            "source_id": str(row["id"]) if "id" in row else "",
            "question": str(row["prompt"]),
        }
        records.append(
            {
                "prompt": build_initial_messages(
                    context_payload=info["context"],
                    root_prompt=str(row["prompt"]),
                ),
                "answer": answers[0],
                "task": str(row["prompt"]),
                "info": json.dumps(info, separators=(",", ":")),
            }
        )
    return Dataset.from_list(records)


def build_datasets(
    *,
    data_paths: list[str] | None,
    eval_data_paths: list[str] | None,
    seed: int,
    max_examples: int,
    max_eval_examples: int,
) -> tuple[vf.DatasetBuilder, vf.DatasetBuilder]:
    train_paths = _resolve_paths(
        data_paths,
        default_candidates=DEFAULT_TRAIN_DATASET_CANDIDATES,
        label="train",
    )
    eval_paths = _resolve_paths(
        eval_data_paths,
        default_candidates=DEFAULT_EVAL_DATASET_CANDIDATES,
        label="eval",
    )

    def build_train() -> Dataset:
        frame = _load_frame(train_paths)
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if max_examples > 0:
            frame = frame.iloc[:max_examples].reset_index(drop=True)
        return _to_dataset(frame)

    def build_eval() -> Dataset:
        frame = _load_frame(eval_paths)
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if max_eval_examples > 0:
            frame = frame.iloc[:max_eval_examples].reset_index(drop=True)
        return _to_dataset(frame)

    return build_train, build_eval