from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


TRAINABLE_KINDS = {"root_turn", "root_finalize_turn", "recursive_turn", "finalize_turn"}
DEFAULT_DATASET_ID = "lsteno/rlm-rlvr-glm5-sft"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export accepted RLM traces to Prime SFT prompt/completion rows.")
    parser.add_argument("--input", type=Path, required=True, help="Path to successful_records.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Local directory for the exported DatasetDict.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="HF dataset id to push when --push is set.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Tokenizer used for validation.")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Held-out source-id ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--push", action=argparse.BooleanOptionalAction, default=True, help="Push private HF dataset.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_is_accepted(record: dict[str, Any]) -> bool:
    if record.get("error"):
        return False
    return bool(record.get("exact_match")) or record.get("judge_score") == 1.0


def sanitize_assistant_history(content: str) -> str:
    sanitized = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    return sanitized.strip()


def sanitize_prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "assistant":
            content = sanitize_assistant_history(content)
        sanitized.append({"role": role, "content": content.strip()})
    return sanitized


def segment_is_trainable_sft_turn(segment: dict[str, Any]) -> bool:
    if not bool(segment.get("is_trainable_rlm_turn")):
        return False
    if str(segment.get("kind")) not in TRAINABLE_KINDS:
        return False
    if str(segment.get("train_scope")) == "llm_subcall":
        return False
    return bool(str(segment.get("response_text", "")).strip()) and isinstance(segment.get("prompt_messages"), list)


def extract_sft_rows(records: list[dict[str, Any]], *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record_is_accepted(record):
            continue
        source_id = str(record.get("source_id", record.get("example_id", "")))
        for segment in record.get("segments", []):
            if not isinstance(segment, dict) or not segment_is_trainable_sft_turn(segment):
                continue
            rows.append(
                {
                    "source_id": source_id,
                    "example_id": record.get("example_id"),
                    "segment_order": int(segment.get("order", len(rows))),
                    "segment_kind": str(segment.get("kind")),
                    "prompt": sanitize_prompt_messages(segment["prompt_messages"]),
                    "completion": [{"role": "assistant", "content": str(segment["response_text"]).strip()}],
                    "final_answer": str(record.get("final_answer", "")),
                    "exact_match": bool(record.get("exact_match")),
                    "judge_score": record.get("judge_score"),
                    "prompt_token_count": int(segment.get("prompt_token_count", 0) or 0),
                    "completion_token_count": int(segment.get("completion_token_count", 0) or 0),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def split_rows_by_source(rows: list[dict[str, Any]], *, eval_ratio: float, seed: int) -> DatasetDict:
    source_ids = sorted({str(row["source_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    eval_count = max(1, int(round(len(source_ids) * eval_ratio))) if len(source_ids) > 1 else 0
    eval_ids = set(source_ids[:eval_count])
    train_rows = [row for row in rows if str(row["source_id"]) not in eval_ids]
    eval_rows = [row for row in rows if str(row["source_id"]) in eval_ids]
    return DatasetDict({"train": Dataset.from_list(train_rows), "eval": Dataset.from_list(eval_rows)})


def validate_tokenization(dataset: DatasetDict, *, tokenizer_name: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    stats = {"num_rows": 0, "max_tokens": 0}
    for split_name, split in dataset.items():
        for index, row in enumerate(split):
            messages = row["prompt"] + row["completion"]
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            if not token_ids:
                raise ValueError(f"Tokenization produced no tokens for {split_name}[{index}]")
            stats["num_rows"] += 1
            stats["max_tokens"] = max(stats["max_tokens"], len(token_ids))
    return stats


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    rows = extract_sft_rows(records, max_rows=args.max_rows)
    if not rows:
        raise ValueError("No accepted trainable SFT rows were found.")

    dataset = split_rows_by_source(rows, eval_ratio=args.eval_ratio, seed=args.seed)
    validation = validate_tokenization(dataset, tokenizer_name=args.tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    write_jsonl(args.output_dir / "train.jsonl", list(dataset["train"]))
    write_jsonl(args.output_dir / "eval.jsonl", list(dataset["eval"]))
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "input": str(args.input),
                "train_rows": len(dataset["train"]),
                "eval_rows": len(dataset["eval"]),
                "tokenization": validation,
            },
            indent=2,
        )
    )

    if args.push:
        dataset.push_to_hub(args.dataset_id, private=True)


if __name__ == "__main__":
    main()
