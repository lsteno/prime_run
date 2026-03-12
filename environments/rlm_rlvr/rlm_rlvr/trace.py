from __future__ import annotations

from typing import Any


def make_segment(
    *,
    order: int,
    depth: int,
    kind: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    completion_logprobs: list[float],
    completion_mask: list[bool] | None = None,
    temperature: float,
    response_text: str,
) -> dict[str, Any]:
    if completion_mask is None:
        completion_mask = [True] * len(completion_ids)
    return {
        "order": order,
        "depth": depth,
        "kind": kind,
        "prompt_ids": list(prompt_ids),
        "completion_ids": list(completion_ids),
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