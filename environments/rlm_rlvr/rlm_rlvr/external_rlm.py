from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def _ensure_rlm_importable() -> None:
    if importlib.util.find_spec("rlm") is not None:
        return

    source_dir = Path(os.environ.get("RLM_SOURCE_DIR", "/home/coder/rlm"))
    if source_dir.exists():
        sys.path.insert(0, str(source_dir))

    if importlib.util.find_spec("rlm") is None:
        raise ModuleNotFoundError(
            "The 'rlm' package is required for rlm_rlvr. Install 'rlms' or set RLM_SOURCE_DIR to a local checkout."
        )


_ensure_rlm_importable()

from rlm.core.types import CodeBlock, QueryMetadata, RLMChatCompletion, RLMIteration, UsageSummary
from rlm.environments.local_repl import LocalREPL
from rlm.utils.parsing import find_code_blocks, find_final_answer, format_iteration
from rlm.utils.prompts import RLM_SYSTEM_PROMPT, build_rlm_system_prompt, build_user_prompt


def build_system_prompt(*, depth: int, max_depth: int, enable_rlm_query_batched_async: bool = True) -> str:
    messages = build_rlm_system_prompt(
        system_prompt=RLM_SYSTEM_PROMPT,
        query_metadata=QueryMetadata(""),
        recursion_budget=max(0, max_depth - depth),
        current_depth=depth,
        max_depth=max_depth,
        enable_rlm_query_batched_async=enable_rlm_query_batched_async,
    )
    return str(messages[0]["content"])


def build_initial_messages(
    *,
    context_payload: str | dict[str, Any] | list[Any],
    root_prompt: str,
    enable_rlm_query_batched_async: bool = True,
) -> list[dict[str, str]]:
    system_and_metadata = build_rlm_system_prompt(
        system_prompt=RLM_SYSTEM_PROMPT,
        query_metadata=QueryMetadata(context_payload),
        enable_rlm_query_batched_async=enable_rlm_query_batched_async,
    )
    return [system_and_metadata[1], build_user_prompt(root_prompt=root_prompt, iteration=0)]


def make_feedback_messages(iteration: RLMIteration, *, max_chars: int) -> list[dict[str, str]]:
    messages = format_iteration(iteration, max_character_length=max_chars)
    return [message for message in messages[1:] if message.get("role") == "user"]


def empty_usage_summary() -> UsageSummary:
    return UsageSummary(model_usage_summaries={})


__all__ = [
    "CodeBlock",
    "LocalREPL",
    "QueryMetadata",
    "RLMChatCompletion",
    "RLMIteration",
    "RLM_SYSTEM_PROMPT",
    "UsageSummary",
    "build_initial_messages",
    "build_system_prompt",
    "build_user_prompt",
    "empty_usage_summary",
    "find_code_blocks",
    "find_final_answer",
    "make_feedback_messages",
]