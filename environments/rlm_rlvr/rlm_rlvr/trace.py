from __future__ import annotations

import hashlib
import json
from typing import Any


def _normalize_prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        normalized.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
        )
    return normalized


def prompt_provenance(messages: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize_prompt_messages(messages)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "prompt_fingerprint": hashlib.sha1(serialized.encode("utf-8")).hexdigest(),
        "prompt_message_count": len(normalized),
        "prompt_char_count": sum(len(message["role"]) + len(message["content"]) for message in normalized),
    }


def make_segment(
    *,
    order: int,
    call_id: int,
    parent_call_id: int | None,
    depth: int,
    turn_index: int,
    kind: str,
    train_scope: str,
    is_trainable_rlm_turn: bool,
    response_source: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    completion_logprobs: list[float],
    completion_mask: list[bool] | None = None,
    prompt_token_count: int | None = None,
    completion_token_count: int | None = None,
    temperature: float,
    response_text: str,
    prompt_fingerprint: str | None = None,
    prompt_message_count: int | None = None,
    prompt_char_count: int | None = None,
) -> dict[str, Any]:
    if completion_mask is None:
        completion_mask = [True] * len(completion_ids)
    return {
        "order": order,
        "call_id": int(call_id),
        "parent_call_id": None if parent_call_id is None else int(parent_call_id),
        "depth": depth,
        "turn_index": int(turn_index),
        "kind": kind,
        "train_scope": train_scope,
        "is_trainable_rlm_turn": bool(is_trainable_rlm_turn),
        "response_source": response_source,
        "prompt_fingerprint": prompt_fingerprint,
        "prompt_message_count": prompt_message_count,
        "prompt_char_count": prompt_char_count,
        "prompt_ids": list(prompt_ids),
        "completion_ids": list(completion_ids),
        "prompt_token_count": len(prompt_ids) if prompt_token_count is None else int(prompt_token_count),
        "completion_token_count": len(completion_ids) if completion_token_count is None else int(completion_token_count),
        "completion_logprobs": [float(value) for value in completion_logprobs],
        "completion_mask": [bool(value) for value in completion_mask],
        "temperature": float(temperature),
        "response_text": response_text,
    }


def make_call_trace(call_id: int, depth: int, prompt: str) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "depth": depth,
        "prompt": prompt,
        "steps": [],
        "final_answer": None,
    }


def append_step_trace(
    trace: dict[str, Any],
    *,
    assistant: str,
    code_blocks: list[str],
    feedback: list[str],
    final_answer: str | None,
) -> None:
    trace["steps"].append(
        {
            "assistant": assistant,
            "code_blocks": list(code_blocks),
            "feedback": list(feedback),
            "final_answer": final_answer,
        }
    )
    if final_answer is not None:
        trace["final_answer"] = final_answer
