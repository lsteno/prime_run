from __future__ import annotations

import ast
import re
from typing import Any

from rlm.core.types import CodeBlock, QueryMetadata, RLMChatCompletion, RLMIteration, UsageSummary
from rlm.environments.local_repl import LocalREPL
from rlm.utils.parsing import find_code_blocks, format_iteration
from rlm.utils.prompts import build_rlm_system_prompt, build_user_prompt

from .prompt_variants import DEFAULT_PROMPT_VARIANT, get_system_prompt_template

_FINAL_CALL_RE = re.compile(r"FINAL\((.*?)\)", re.DOTALL)
_FINAL_VAR_CALL_RE = re.compile(r"FINAL_VAR\((.*?)\)", re.DOTALL)


def _literal_final_value(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw.strip().strip("\"'")
    return str(value)


def _lookup_repl_var(environment: Any, name: str) -> Any | None:
    for attr in ("locals", "globals", "vars", "variables", "state"):
        namespace = getattr(environment, attr, None)
        if isinstance(namespace, dict) and name in namespace:
            return namespace[name]
    repl = getattr(environment, "repl", None)
    if repl is not None and repl is not environment:
        return _lookup_repl_var(repl, name)
    return None


def find_final_answer(text: str, environment: Any | None = None) -> str | None:
    """Compatibility parser for rlms versions without find_final_answer."""
    if not text:
        return None
    var_matches = list(_FINAL_VAR_CALL_RE.finditer(text))
    if var_matches:
        raw_name = var_matches[-1].group(1).strip()
        try:
            name = ast.literal_eval(raw_name)
        except (SyntaxError, ValueError):
            name = raw_name.strip().strip("\"'")
        if isinstance(name, str) and name and environment is not None:
            value = _lookup_repl_var(environment, name)
            if value is not None:
                return str(value)
        return None
    matches = list(_FINAL_CALL_RE.finditer(text))
    if not matches:
        return None
    return _literal_final_value(matches[-1].group(1))


def append_budget_reminder(
    system_prompt_template: str,
    *,
    max_prompt_tokens: int | None = None,
    turn_max_tokens: int | None = None,
    subcall_max_tokens: int | None = None,
    subcall_budget_enabled: bool = False,
    max_total_subcalls: int | None = None,
    max_batched_subcalls: int | None = None,
    llm_only_subcalls: bool = False,
    include_budget_reminder: bool = True,
) -> str:
    if not include_budget_reminder:
        return system_prompt_template

    context_window = f"{max_prompt_tokens} prompt tokens" if max_prompt_tokens is not None else "the configured model context window"
    turn_budget = f"{turn_max_tokens} output tokens" if turn_max_tokens is not None else "the configured turn output budget"
    subcall_budget = f"{subcall_max_tokens} output tokens" if subcall_max_tokens is not None else "the configured subcall output budget"
    turn_label = "RLM turns" if llm_only_subcalls else "root and recursive RLM turns"
    reminder = (
        "Budget reminder: your prompt context window is "
        f"{context_window}; {turn_label} may output up to {turn_budget}; "
        f"one-shot LLM subcalls may output up to {subcall_budget}; make every subcall context-aware "
        "with only the relevant excerpts to avoid context rot."
    )
    if subcall_budget_enabled:
        total_calls = max_total_subcalls if max_total_subcalls is not None else "the configured number of"
        batched_calls = max_batched_subcalls if max_batched_subcalls is not None else "the configured number of"
        budget_scope = "LLM subcalls" if llm_only_subcalls else "calls across llm_query and rlm_query"
        reminder = (
            f"{reminder} The subcall budget is {total_calls} total {budget_scope}; "
            f"batched calls may schedule at most {batched_calls} prompts."
        )
    return f"{system_prompt_template.rstrip()}\n\n{reminder}\n"


def build_system_prompt(
    *,
    depth: int,
    max_depth: int,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    max_prompt_tokens: int | None = None,
    turn_max_tokens: int | None = None,
    subcall_max_tokens: int | None = None,
    subcall_budget_enabled: bool = False,
    max_total_subcalls: int | None = None,
    max_batched_subcalls: int | None = None,
    include_budget_reminder: bool = True,
) -> str:
    llm_only_subcalls = prompt_variant == "sanjaya_text_depth1_llm_only_v1"
    system_prompt_template = append_budget_reminder(
        get_system_prompt_template(prompt_variant),
        max_prompt_tokens=max_prompt_tokens,
        turn_max_tokens=turn_max_tokens,
        subcall_max_tokens=subcall_max_tokens,
        subcall_budget_enabled=subcall_budget_enabled,
        max_total_subcalls=max_total_subcalls,
        max_batched_subcalls=max_batched_subcalls,
        llm_only_subcalls=llm_only_subcalls,
        include_budget_reminder=include_budget_reminder,
    )
    base_kwargs = {
        "system_prompt": system_prompt_template,
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
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    max_prompt_tokens: int | None = None,
    turn_max_tokens: int | None = None,
    subcall_max_tokens: int | None = None,
    subcall_budget_enabled: bool = False,
    max_total_subcalls: int | None = None,
    max_batched_subcalls: int | None = None,
    include_budget_reminder: bool = True,
) -> list[dict[str, str]]:
    llm_only_subcalls = prompt_variant == "sanjaya_text_depth1_llm_only_v1"
    system_prompt_template = append_budget_reminder(
        get_system_prompt_template(prompt_variant),
        max_prompt_tokens=max_prompt_tokens,
        turn_max_tokens=turn_max_tokens,
        subcall_max_tokens=subcall_max_tokens,
        subcall_budget_enabled=subcall_budget_enabled,
        max_total_subcalls=max_total_subcalls,
        max_batched_subcalls=max_batched_subcalls,
        llm_only_subcalls=llm_only_subcalls,
        include_budget_reminder=include_budget_reminder,
    )
    try:
        system_and_metadata = build_rlm_system_prompt(
            system_prompt=system_prompt_template,
            query_metadata=QueryMetadata(context_payload),
        )
    except TypeError:
        system_and_metadata = build_rlm_system_prompt(system_prompt=system_prompt_template)
    return [*system_and_metadata, build_user_prompt(root_prompt=root_prompt, iteration=0)]


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
    "UsageSummary",
    "append_budget_reminder",
    "build_initial_messages",
    "build_system_prompt",
    "build_user_prompt",
    "empty_usage_summary",
    "find_code_blocks",
    "find_final_answer",
    "make_feedback_messages",
]
