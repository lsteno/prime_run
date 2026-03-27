from __future__ import annotations

from typing import Any


def _step_response_text(trajectory_step: dict[str, Any]) -> str:
    completion = trajectory_step.get("completion") or []
    if completion:
        last_message = completion[-1]
        if isinstance(last_message, dict):
            content = last_message.get("content")
            if isinstance(content, str):
                return content

    response = trajectory_step.get("response")
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
    return ""


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


def segment_from_trajectory_step(
    trajectory_step: dict[str, Any],
    *,
    order: int,
    depth: int,
    kind: str,
    default_temperature: float,
) -> dict[str, Any] | None:
    tokens = trajectory_step.get("tokens")
    if tokens is None:
        return None

    return make_segment(
        order=order,
        depth=depth,
        kind=kind,
        prompt_ids=list(tokens["prompt_ids"]),
        completion_ids=list(tokens["completion_ids"]),
        completion_logprobs=[float(value) for value in tokens["completion_logprobs"]],
        completion_mask=[bool(value) for value in tokens["completion_mask"]],
        temperature=float((trajectory_step.get("sampling_args") or {}).get("temperature", default_temperature)),
        response_text=_step_response_text(trajectory_step),
    )


def build_recursive_trace(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for call in calls:
        metadata = dict(call.get("metadata") or {})
        normalized.append(
            {
                "call_id": int(call.get("call_id", 0)),
                "depth": int(call.get("depth", 0)),
                "prompt": call.get("prompt", ""),
                "response": call.get("response", ""),
                "parent_turn": int(call.get("parent_turn", 0)),
                "remaining_depth": int(call.get("remaining_depth", 0)),
                "batch_id": call.get("batch_id"),
                "request_id": call.get("request_id"),
                "prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(metadata.get("completion_tokens", 0) or 0),
                "tool_call_count": int(metadata.get("tool_call_count", 0) or 0),
                "num_turns": int(metadata.get("num_turns", 0) or 0),
                "max_turns_reached": bool(metadata.get("max_turns_reached", False)),
            }
        )
    return normalized
