from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import verifiers as vf
from datasets import Dataset

from .parsing import parse_answer_candidates

DEFAULT_DATASET_CANDIDATES = [
    Path("/home/coder/rl_training/data/train.parquet"),
    Path("/home/coder/dataset-maker/data/oolong-like/train.parquet"),
]


def _resolve_paths(paths: list[str] | None) -> list[Path]:
    if paths:
        resolved = [Path(path) for path in paths]
    else:
        resolved = [path for path in DEFAULT_DATASET_CANDIDATES if path.exists()]
    if not resolved:
        raise FileNotFoundError(
            "No parquet dataset paths were found. Pass `data_paths` explicitly to load_environment()."
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
        }
        records.append(
            {
                "question": str(row["prompt"]),
                "answer": answers[0],
                "info": json.dumps(info, separators=(",", ":")),
            }
        )
    return Dataset.from_list(records)


def build_datasets(
    *,
    data_paths: list[str] | None,
    eval_data_paths: list[str] | None,
    seed: int,
    eval_fraction: float,
    eval_size: int | None,
    max_examples: int,
    max_eval_examples: int,
) -> tuple[vf.DatasetBuilder, vf.DatasetBuilder]:
    train_paths = _resolve_paths(data_paths)
    eval_paths = _resolve_paths(eval_data_paths) if eval_data_paths else None

    def build_train() -> Dataset:
        if eval_paths:
            frame = _load_frame(train_paths)
            frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        else:
            frame = _split_train_eval(_load_frame(train_paths), seed, eval_fraction, eval_size)[0]
        if max_examples > 0:
            frame = frame.iloc[:max_examples].reset_index(drop=True)
        return _to_dataset(frame)

    def build_eval() -> Dataset:
        if eval_paths:
            frame = _load_frame(eval_paths)
            frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        else:
            frame = _split_train_eval(_load_frame(train_paths), seed, eval_fraction, eval_size)[1]
        if max_eval_examples > 0:
            frame = frame.iloc[:max_eval_examples].reset_index(drop=True)
        return _to_dataset(frame)

    return build_train, build_eval


def _split_train_eval(
    frame: pd.DataFrame,
    seed: int,
    eval_fraction: float,
    eval_size: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(shuffled) < 2:
        raise ValueError("Need at least 2 examples to create train/eval splits.")

    if eval_size is not None:
        holdout = eval_size
    else:
        holdout = max(1, int(round(len(shuffled) * eval_fraction)))

    holdout = min(max(1, holdout), len(shuffled) - 1)
    eval_frame = shuffled.iloc[:holdout].reset_index(drop=True)
    train_frame = shuffled.iloc[holdout:].reset_index(drop=True)
    return train_frame, eval_frame