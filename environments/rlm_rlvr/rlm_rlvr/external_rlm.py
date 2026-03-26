from __future__ import annotations

from typing import Any

from rlm.core.types import CodeBlock, QueryMetadata, RLMChatCompletion, RLMIteration, UsageSummary
from rlm.environments import get_environment
from rlm.environments.base_env import BaseEnv
from rlm.environments.local_repl import LocalREPL
from rlm.utils.parsing import find_code_blocks, find_final_answer, format_iteration
from rlm.utils.prompts import RLM_SYSTEM_PROMPT, build_rlm_system_prompt, build_user_prompt


def build_system_prompt(*, depth: int, max_depth: int, enable_rlm_query_batched_async: bool = True) -> str:
    del enable_rlm_query_batched_async
    base_kwargs = {
        "system_prompt": RLM_SYSTEM_PROMPT,
        "query_metadata": QueryMetadata(""),
    }
    attempts = [
        {
            **base_kwargs,
            "recursion_budget": max(0, max_depth - depth),
            "current_depth": depth,
            "max_depth": max_depth,
        },
        {
            **base_kwargs,
            "current_depth": depth,
            "max_depth": max_depth,
        },
        base_kwargs,
    ]

    messages = None
    for kwargs in attempts:
        try:
            messages = build_rlm_system_prompt(**kwargs)
            break
        except TypeError:
            continue

    if messages is None:
        raise RuntimeError("Unable to call build_rlm_system_prompt with installed rlms version.")
    return str(messages[0]["content"])


def build_initial_messages(
    *,
    context_payload: str | dict[str, Any] | list[Any],
    root_prompt: str,
    enable_rlm_query_batched_async: bool = True,
) -> list[dict[str, str]]:
    del enable_rlm_query_batched_async
    try:
        system_and_metadata = build_rlm_system_prompt(
            system_prompt=RLM_SYSTEM_PROMPT,
            query_metadata=QueryMetadata(context_payload),
        )
    except TypeError:
        system_and_metadata = build_rlm_system_prompt(system_prompt=RLM_SYSTEM_PROMPT)
    return [system_and_metadata[1], build_user_prompt(root_prompt=root_prompt, iteration=0)]


def make_feedback_messages(iteration: RLMIteration, *, max_chars: int) -> list[dict[str, str]]:
    messages = format_iteration(iteration, max_character_length=max_chars)
    return [message for message in messages[1:] if message.get("role") == "user"]


def empty_usage_summary() -> UsageSummary:
    return UsageSummary(model_usage_summaries={})


__all__ = [
    "CodeBlock",
    "BaseEnv",
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
    "get_environment",
    "make_feedback_messages",
]