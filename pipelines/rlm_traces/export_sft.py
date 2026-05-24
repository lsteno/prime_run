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
DEFAULT_CONVERSATION_DATASET_ID = "lsteno/rlm-rlvr-sft-v2-conversations"
DEFAULT_PER_ROOT_TURN_DATASET_ID = "lsteno/rlm-rlvr-sft-v3-per-root-turn"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_CURATED_INPUT = Path("outputs/rlm_traces/combined-good-sft-traces-v2/curated-v2-reviewed/records.curated.jsonl")
DEFAULT_CONVERSATION_OUTPUT = Path("outputs/rlm_traces/combined-good-sft-traces-v2/sft_dataset_v2_conversations")
DEFAULT_PER_ROOT_TURN_OUTPUT = Path("outputs/rlm_traces/combined-good-sft-traces-v2/sft_dataset_v3_per_root_turn")

REPL_FENCE_RE = re.compile(r"```repl\s*\n.*?```", re.DOTALL)
FINAL_RE = re.compile(r"\bFINAL\s*\(", re.DOTALL)
FINAL_VAR_RE = re.compile(r"\bFINAL_VAR\s*\(", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export accepted RLM traces to Prime SFT prompt/completion rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_CURATED_INPUT, help="Path to records JSONL.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Local directory for the exported DatasetDict.")
    parser.add_argument("--dataset-id", default=None, help="HF dataset id to push when --push is set.")
    parser.add_argument(
        "--format",
        choices=["conversation", "per_turn", "per_root_turn"],
        default="conversation",
        help=(
            "conversation exports one multi-turn row per strict trace; per_turn preserves the older "
            "trainable-segment format; per_root_turn exports one strict root-decision row per RLM turn."
        ),
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Tokenizer used for validation.")
    parser.add_argument(
        "--chat-template-kwargs-json",
        default=None,
        help='Optional JSON object added to every row as chat_template_kwargs, e.g. {"enable_thinking": false}.',
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Held-out source-id ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument(
        "--validate-tokenization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run full tokenizer validation and fail rows longer than --max-seq-len.",
    )
    parser.add_argument("--push", action=argparse.BooleanOptionalAction, default=True, help="Push private HF dataset.")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_PER_ROOT_TURN_OUTPUT if args.format == "per_root_turn" else DEFAULT_CONVERSATION_OUTPUT
    if args.dataset_id is None:
        args.dataset_id = (
            DEFAULT_PER_ROOT_TURN_DATASET_ID if args.format == "per_root_turn" else DEFAULT_CONVERSATION_DATASET_ID
        )
    return args


def parse_json_object(value: str | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


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


def is_final_response(content: str) -> bool:
    return bool(FINAL_RE.search(content) or FINAL_VAR_RE.search(content))


def scrub_non_final_assistant_content(content: str) -> str:
    """Drop ignored plain text after a non-final REPL block.

    The RLM runtime executes fenced ```repl blocks and ignores trailing prose or
    bare answers in the same assistant message. Keeping that trailing text would
    teach a protocol artifact that the environment never actually consumes.
    """
    stripped = content.strip()
    if is_final_response(stripped):
        return stripped
    matches = list(REPL_FENCE_RE.finditer(stripped))
    if not matches:
        return stripped
    last = matches[-1]
    trailing = stripped[last.end() :].strip()
    if not trailing:
        return stripped
    return stripped[: last.end()].strip()


def segment_is_trainable_sft_turn(segment: dict[str, Any]) -> bool:
    if not bool(segment.get("is_trainable_rlm_turn")):
        return False
    if str(segment.get("kind")) not in TRAINABLE_KINDS:
        return False
    if str(segment.get("train_scope")) == "llm_subcall":
        return False
    return bool(str(segment.get("response_text", "")).strip()) and isinstance(segment.get("prompt_messages"), list)


def segment_is_strictly_exportable(segment: dict[str, Any]) -> bool:
    if not segment_is_trainable_sft_turn(segment):
        return False
    curation = segment.get("curation", {})
    if isinstance(curation, dict) and curation.get("exclude_from_sft"):
        return False
    return True


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


def record_is_strict_sft(record: dict[str, Any]) -> bool:
    return record.get("curation", {}).get("status") == "strict_sft"


def non_empty_plain_subcall_count(record: dict[str, Any]) -> int:
    count = 0
    for segment in record.get("segments", []):
        if not isinstance(segment, dict):
            continue
        if str(segment.get("train_scope")) != "llm_subcall":
            continue
        if str(segment.get("kind")) != "plain_query":
            continue
        if str(segment.get("response_text", "")).strip():
            count += 1
    return count


def message_key(message: dict[str, Any]) -> tuple[str, str]:
    return str(message.get("role", "")), str(message.get("content", "")).strip()


def initial_prompt_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    initial: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant":
            break
        initial.append(message)
    return sanitize_prompt_messages(initial)


def prompt_tail_after_last_assistant(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    last_assistant = -1
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            last_assistant = index
    if last_assistant < 0:
        return []
    return sanitize_prompt_messages(messages[last_assistant + 1 :])


def append_unique_messages(target: list[dict[str, str]], messages: list[dict[str, str]]) -> None:
    for message in messages:
        if not message.get("content", "").strip():
            continue
        if target and message_key(target[-1]) == message_key(message):
            continue
        target.append(message)


def iter_trainable_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for segment in sorted(record.get("segments", []), key=lambda item: int(item.get("order", 0) or 0)):
        if not isinstance(segment, dict) or not segment_is_strictly_exportable(segment):
            continue
        synthetic_rows = segment.get("_curation_sft_rows")
        if isinstance(synthetic_rows, list):
            for index, synthetic in enumerate(synthetic_rows):
                response_text = str(synthetic.get("response_text", "")).strip()
                prompt_messages = synthetic.get("prompt_messages", segment.get("prompt_messages", []))
                if response_text and isinstance(prompt_messages, list):
                    events.append(
                        {
                            "segment_order": int(segment.get("order", 0) or 0),
                            "segment_kind": str(segment.get("kind")),
                            "synthetic_index": index,
                            "row_role": str(synthetic.get("row_role", "curated")),
                            "prompt_messages": prompt_messages,
                            "response_text": response_text,
                        }
                    )
            continue
        events.append(
            {
                "segment_order": int(segment.get("order", 0) or 0),
                "segment_kind": str(segment.get("kind")),
                "synthetic_index": None,
                "row_role": "original_or_repaired",
                "prompt_messages": segment["prompt_messages"],
                "response_text": str(segment["response_text"]).strip(),
            }
        )
    return events


def build_conversation_row(record: dict[str, Any]) -> dict[str, Any] | None:
    if not record_is_strict_sft(record):
        return None
    if non_empty_plain_subcall_count(record) < 1:
        return None

    events = iter_trainable_events(record)
    if not events:
        return None

    prompt = initial_prompt_from_messages(events[0]["prompt_messages"])
    if not prompt:
        return None

    completion: list[dict[str, str]] = []
    assistant_turns = 0
    scrubbed_turns = 0
    for index, event in enumerate(events):
        if index > 0:
            append_unique_messages(completion, prompt_tail_after_last_assistant(event["prompt_messages"]))
        raw_content = event["response_text"]
        content = scrub_non_final_assistant_content(raw_content)
        if content != raw_content.strip():
            scrubbed_turns += 1
        if not content.strip():
            continue
        completion.append({"role": "assistant", "content": content})
        assistant_turns += 1

    if assistant_turns == 0 or not completion:
        return None

    curation = record.get("curation", {})
    source_id = str(record.get("source_id", record.get("example_id", "")))
    endpoint = record.get("endpoint")
    endpoint_model = endpoint.get("model") if isinstance(endpoint, dict) else None
    return {
        "row_id": f"{source_id}:conversation",
        "source_id": source_id,
        "example_id": record.get("example_id"),
        "prompt": prompt,
        "completion": completion,
        "final_answer": str(record.get("final_answer", "")),
        "exact_match": bool(record.get("exact_match")),
        "judge_score": record.get("judge_score"),
        "root_model": curation.get("root_model") or record.get("_root_model") or endpoint_model,
        "source_run_id": curation.get("source_run_id") or record.get("_source_run_id"),
        "source_input_path": curation.get("source_input_path") or record.get("_curation_input_path"),
        "curation_version": curation.get("version"),
        "curation_status": curation.get("status"),
        "curation_tags": curation.get("tags", []),
        "manual_decision_id": curation.get("manual_decision_id"),
        "prompt_variant": record.get("prompt_variant"),
        "num_assistant_turns": assistant_turns,
        "num_completion_messages": len(completion),
        "num_llm_subcalls": int(record.get("num_llm_subcalls", 0) or 0),
        "non_empty_llm_subcall_count": non_empty_plain_subcall_count(record),
        "scrubbed_non_final_repl_turns": scrubbed_turns,
        "total_prompt_tokens": int(record.get("total_prompt_tokens", 0) or 0),
        "total_completion_tokens": int(record.get("total_completion_tokens", 0) or 0),
        "total_rollout_tokens": int(record.get("total_rollout_tokens", 0) or 0),
    }


def extract_conversation_rows(records: list[dict[str, Any]], *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = build_conversation_row(record)
        if row is None:
            continue
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            return rows
    return rows


def build_per_root_turn_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    if not record_is_strict_sft(record):
        return []
    non_empty_subcalls = non_empty_plain_subcall_count(record)
    if non_empty_subcalls < 1:
        return []

    curation = record.get("curation", {})
    source_id = str(record.get("source_id", record.get("example_id", "")))
    endpoint = record.get("endpoint")
    endpoint_model = endpoint.get("model") if isinstance(endpoint, dict) else None
    rows: list[dict[str, Any]] = []
    for event in iter_trainable_events(record):
        raw_content = event["response_text"]
        content = scrub_non_final_assistant_content(raw_content)
        if not content.strip():
            continue
        synthetic_index = event.get("synthetic_index")
        suffix = f"{event['segment_order']:04d}"
        if synthetic_index is not None:
            suffix += f":synthetic-{synthetic_index}"
        rows.append(
            {
                "row_id": f"{source_id}:root_turn:{suffix}",
                "source_id": source_id,
                "example_id": record.get("example_id"),
                "segment_order": event["segment_order"],
                "segment_kind": event["segment_kind"],
                "synthetic_index": synthetic_index,
                "row_role": event["row_role"],
                "prompt": sanitize_prompt_messages(event["prompt_messages"]),
                "completion": [{"role": "assistant", "content": content}],
                "final_answer": str(record.get("final_answer", "")),
                "exact_match": bool(record.get("exact_match")),
                "judge_score": record.get("judge_score"),
                "root_model": curation.get("root_model") or record.get("_root_model") or endpoint_model,
                "source_run_id": curation.get("source_run_id") or record.get("_source_run_id"),
                "source_input_path": curation.get("source_input_path") or record.get("_curation_input_path"),
                "curation_version": curation.get("version"),
                "curation_status": curation.get("status"),
                "curation_tags": curation.get("tags", []),
                "manual_decision_id": curation.get("manual_decision_id"),
                "prompt_variant": record.get("prompt_variant"),
                "num_llm_subcalls": int(record.get("num_llm_subcalls", 0) or 0),
                "non_empty_llm_subcall_count": non_empty_subcalls,
                "scrubbed_non_final_repl_turn": content != raw_content.strip(),
                "total_prompt_tokens": int(record.get("total_prompt_tokens", 0) or 0),
                "total_completion_tokens": int(record.get("total_completion_tokens", 0) or 0),
                "total_rollout_tokens": int(record.get("total_rollout_tokens", 0) or 0),
            }
        )
    return rows


def extract_per_root_turn_rows(records: list[dict[str, Any]], *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for row in build_per_root_turn_rows(record):
            rows.append(row)
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


def add_chat_template_kwargs(rows: list[dict[str, Any]], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    if not kwargs:
        return rows
    updated: list[dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        existing = row_copy.get("chat_template_kwargs")
        if existing is None:
            existing = {}
        if not isinstance(existing, dict):
            raise ValueError(f"Row {row_copy.get('row_id')} has non-object chat_template_kwargs")
        row_copy["chat_template_kwargs"] = {**existing, **kwargs}
        updated.append(row_copy)
    return updated


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No SFT rows were exported.")
    seen_by_split: dict[str, set[str]] = {}
    for row in rows:
        if not row.get("prompt"):
            raise ValueError(f"Row {row.get('row_id')} has an empty prompt")
        if not row.get("completion"):
            raise ValueError(f"Row {row.get('row_id')} has an empty completion")
        if row.get("curation_status") == "audit_only":
            raise ValueError(f"Audit-only row exported: {row.get('row_id')}")
        if "non_empty_llm_subcall_count" in row and int(row.get("non_empty_llm_subcall_count", 0) or 0) < 1:
            raise ValueError(f"Row {row.get('row_id')} has no non-empty LLM subcall")
        assistant_messages = [m for m in row["completion"] if m.get("role") == "assistant" and str(m.get("content", "")).strip()]
        if not assistant_messages:
            raise ValueError(f"Row {row.get('row_id')} has no assistant completion")
        for message in row["prompt"] + row["completion"]:
            if message.get("role") == "assistant" and "<think>" in str(message.get("content", "")).lower():
                raise ValueError(f"Row {row.get('row_id')} contains assistant <think> history")
    return {"num_rows": len(rows), "num_source_ids": len({row["source_id"] for row in rows})}


def validate_tokenization(dataset: DatasetDict, *, tokenizer_name: str, max_seq_len: int | None = None) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    stats = {"num_rows": 0, "max_tokens": 0, "too_long_rows": []}
    for split_name, split in dataset.items():
        for index, row in enumerate(split):
            messages = row["prompt"] + row["completion"]
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                **row.get("chat_template_kwargs", {}),
                return_dict=False,
            )
            if not token_ids:
                raise ValueError(f"Tokenization produced no tokens for {split_name}[{index}]")
            if max_seq_len is not None and len(token_ids) > max_seq_len:
                stats["too_long_rows"].append(
                    {
                        "split": split_name,
                        "index": index,
                        "row_id": row.get("row_id"),
                        "source_id": row.get("source_id"),
                        "tokens": len(token_ids),
                    }
                )
            stats["num_rows"] += 1
            stats["max_tokens"] = max(stats["max_tokens"], len(token_ids))
    if stats["too_long_rows"]:
        first = stats["too_long_rows"][0]
        raise ValueError(
            f"{len(stats['too_long_rows'])} rows exceed max_seq_len={max_seq_len}; "
            f"first too-long row is {first}"
        )
    return stats


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    chat_template_kwargs = parse_json_object(args.chat_template_kwargs_json, label="--chat-template-kwargs-json")
    if args.format == "conversation":
        rows = extract_conversation_rows(records, max_rows=args.max_rows)
    elif args.format == "per_root_turn":
        rows = extract_per_root_turn_rows(records, max_rows=args.max_rows)
    else:
        rows = extract_sft_rows(records, max_rows=args.max_rows)
    rows = add_chat_template_kwargs(rows, chat_template_kwargs)
    if not rows:
        raise ValueError("No accepted trainable SFT rows were found.")
    schema_validation = validate_rows(rows) if args.format in {"conversation", "per_root_turn"} else {"num_rows": len(rows)}

    dataset = split_rows_by_source(rows, eval_ratio=args.eval_ratio, seed=args.seed)
    validation: dict[str, Any] | None = None
    if args.validate_tokenization:
        validation = validate_tokenization(dataset, tokenizer_name=args.tokenizer, max_seq_len=args.max_seq_len)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    write_jsonl(args.output_dir / "train.jsonl", list(dataset["train"]))
    write_jsonl(args.output_dir / "eval.jsonl", list(dataset["eval"]))
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "input": str(args.input),
                "format": args.format,
                "train_rows": len(dataset["train"]),
                "eval_rows": len(dataset["eval"]),
                "schema_validation": schema_validation,
                "tokenization": validation,
                "chat_template_kwargs": chat_template_kwargs,
            },
            indent=2,
        )
    )

    if args.push:
        dataset.push_to_hub(args.dataset_id, private=True)


if __name__ == "__main__":
    main()
