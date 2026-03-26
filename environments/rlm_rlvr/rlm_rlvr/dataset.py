from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import verifiers as vf
from datasets import Dataset, Features, Value, load_dataset

DEFAULT_HF_DATASET_ID = "lsteno/BEEG-agents"
DEFAULT_HF_TRAIN_SPLIT = "train"
DEFAULT_HF_EVAL_SPLIT = "eval"


def _resolve_paths(paths: list[str], *, label: str) -> list[Path]:
    resolved = [Path(path).expanduser() for path in paths]
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing parquet {label} dataset paths: {missing}."
        )
    return resolved


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_by_path(row: dict, path: str) -> object:
    current: object = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _pick_first(row: dict, keys: list[str]) -> object:
    for key in keys:
        value = _extract_by_path(row, key)
        if value is not None:
            return value
    return None


def _parse_answers(value: object) -> list[str]:
    if isinstance(value, list):
        parsed = [_stringify(item).strip() for item in value]
        return [item for item in parsed if item]

    if isinstance(value, dict):
        text = _stringify(value).strip()
        return [text] if text else []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return _parse_answers(parsed)

    text = _stringify(value).strip()
    return [text] if text else []


def _rows_to_dataset(rows: list[dict], *, dataset_name: str) -> Dataset:
    records: list[dict] = []
    skipped_missing_prompt = 0
    skipped_missing_answer = 0

    for row in rows:
        prompt_value = _pick_first(row, ["prompt", "question", "query", "instruction", "task"])
        prompt_text = _stringify(prompt_value).strip()
        if not prompt_text:
            skipped_missing_prompt += 1
            continue

        context_value = _pick_first(
            row,
            [
                "context",
                "context_payload",
                "input",
                "passage",
                "document",
                "metadata.context",
            ],
        )
        context_text = _stringify(context_value)

        answer_value = _pick_first(
            row,
            [
                "answer",
                "answers",
                "acceptable_answers",
                "final_answer",
                "target",
                "label",
                "gold",
                "expected_answer",
                "solution",
            ],
        )
        answers = _parse_answers(answer_value)
        if not answers:
            skipped_missing_answer += 1
            continue

        source_task = _stringify(_pick_first(row, ["task", "category", "dataset", "subset"])) or prompt_text
        source_id = _stringify(_pick_first(row, ["id", "uuid", "example_id", "row_id"]))
        row_dataset = _stringify(_pick_first(row, ["dataset", "dataset_name", "source"])) or dataset_name

        info = {
            "context": context_text,
            "acceptable_answers": answers,
            "dataset_name": row_dataset,
            "source_task": source_task,
            "source_id": source_id,
            "question": prompt_text,
        }
        records.append(
            {
                "question": prompt_text,
                "answer": answers[0],
                "task": source_task,
                "info": json.dumps(info, separators=(",", ":")),
            }
        )

    if not records:
        raise ValueError(
            "No valid dataset rows after normalization. "
            f"Skipped rows: missing prompt={skipped_missing_prompt}, missing answer={skipped_missing_answer}."
        )
    features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "task": Value("string"),
            "info": Value("large_string"),
        }
    )
    return Dataset.from_list(records, features=features)


def _load_frame(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _to_dataset(frame: pd.DataFrame) -> Dataset:
    return _rows_to_dataset(frame.to_dict(orient="records"), dataset_name="local-parquet")


def _load_hf_split(
    *,
    dataset_id: str,
    split: str,
    dataset_config: str | None,
    dataset_revision: str | None,
    seed: int,
) -> Dataset:
    kwargs: dict[str, str] = {"split": split}
    if dataset_revision:
        kwargs["revision"] = dataset_revision

    try:
        if dataset_config:
            return load_dataset(dataset_id, dataset_config, **kwargs)
        return load_dataset(dataset_id, **kwargs)
    except ValueError:
        if split != DEFAULT_HF_EVAL_SPLIT:
            raise
        train_kwargs = dict(kwargs)
        train_kwargs["split"] = DEFAULT_HF_TRAIN_SPLIT
        if dataset_config:
            train_dataset = load_dataset(dataset_id, dataset_config, **train_kwargs)
        else:
            train_dataset = load_dataset(dataset_id, **train_kwargs)
        split_map = train_dataset.train_test_split(test_size=0.1, seed=seed)
        return split_map["test"]


def build_datasets(
    *,
    data_paths: list[str] | None,
    eval_data_paths: list[str] | None,
    dataset_id: str | None,
    dataset_train_split: str,
    dataset_eval_split: str,
    dataset_config: str | None,
    dataset_revision: str | None,
    seed: int,
    max_examples: int,
    max_eval_examples: int,
) -> tuple[vf.DatasetBuilder, vf.DatasetBuilder]:
    use_local_paths = bool(data_paths) or bool(eval_data_paths)

    if use_local_paths:
        if not data_paths:
            raise ValueError("When using local parquet mode, provide non-empty data_paths.")

        train_paths = _resolve_paths(data_paths, label="train")
        eval_paths = _resolve_paths(eval_data_paths, label="eval") if eval_data_paths else []

        def _prepare(frame: pd.DataFrame, *, max_rows: int) -> Dataset:
            frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            if max_rows > 0:
                frame = frame.iloc[:max_rows].reset_index(drop=True)
            return _to_dataset(frame)

        def build_train() -> Dataset:
            frame = _load_frame(train_paths)
            return _prepare(frame, max_rows=max_examples)

        def build_eval() -> Dataset:
            if eval_paths:
                frame = _load_frame(eval_paths)
                return _prepare(frame, max_rows=max_eval_examples)

            frame = _load_frame(train_paths)
            split_map = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            eval_size = max(1, int(len(split_map) * 0.1)) if len(split_map) > 1 else len(split_map)
            eval_frame = split_map.iloc[:eval_size].reset_index(drop=True)
            if max_eval_examples > 0:
                eval_frame = eval_frame.iloc[:max_eval_examples].reset_index(drop=True)
            return _to_dataset(eval_frame)

        return build_train, build_eval

    resolved_dataset_id = dataset_id or DEFAULT_HF_DATASET_ID

    def build_train() -> Dataset:
        hf_dataset = _load_hf_split(
            dataset_id=resolved_dataset_id,
            split=dataset_train_split,
            dataset_config=dataset_config,
            dataset_revision=dataset_revision,
            seed=seed,
        )
        if seed >= 0:
            hf_dataset = hf_dataset.shuffle(seed=seed)
        if max_examples > 0:
            hf_dataset = hf_dataset.select(range(min(max_examples, len(hf_dataset))))
        return _rows_to_dataset(list(hf_dataset), dataset_name=resolved_dataset_id)

    def build_eval() -> Dataset:
        hf_dataset = _load_hf_split(
            dataset_id=resolved_dataset_id,
            split=dataset_eval_split,
            dataset_config=dataset_config,
            dataset_revision=dataset_revision,
            seed=seed,
        )
        if seed >= 0:
            hf_dataset = hf_dataset.shuffle(seed=seed)
        if max_eval_examples > 0:
            hf_dataset = hf_dataset.select(range(min(max_eval_examples, len(hf_dataset))))
        return _rows_to_dataset(list(hf_dataset), dataset_name=resolved_dataset_id)

    return build_train, build_eval
