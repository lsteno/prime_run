from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from rlm_rlvr.external_rlm import build_initial_messages, build_system_prompt
from rlm_rlvr.prompt_variants import DEFAULT_PROMPT_VARIANT, PROMPT_VARIANTS, get_system_prompt_template


def test_prompt_variants_validate_names() -> None:
    with pytest.raises(ValueError):
        get_system_prompt_template("missing")


@pytest.mark.parametrize("prompt_variant", sorted(PROMPT_VARIANTS))
def test_prompt_variants_keep_required_contracts(prompt_variant: str) -> None:
    prompt = get_system_prompt_template(prompt_variant)
    assert "sub-call" in prompt or "subcall" in prompt
    assert "`context`" in prompt
    assert "```repl" in prompt
    assert "FINAL(" in prompt
    assert "FINAL_VAR(" in prompt


def test_build_system_prompt_uses_requested_variant() -> None:
    prompt = build_system_prompt(
        depth=0,
        max_depth=2,
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        max_prompt_tokens=120000,
        turn_max_tokens=1024,
        subcall_max_tokens=1024,
    )
    assert "You are an RLM (Recursive Language Model) agent" in prompt
    assert "Budget reminder: your prompt context window is 120000 prompt tokens" in prompt
    assert "root and recursive RLM turns may output up to 1024 output tokens" in prompt
    assert "one-shot LLM subcalls may output up to 1024 output tokens" in prompt
    assert "avoid context rot" in prompt
    assert DEFAULT_PROMPT_VARIANT == "sanjaya_text_v1"
    assert sorted(PROMPT_VARIANTS) == ["default", "sanjaya_text_v1"]
    assert get_system_prompt_template("default") == get_system_prompt_template(DEFAULT_PROMPT_VARIANT)


def test_build_system_prompt_includes_subcall_budget_when_enabled() -> None:
    prompt = build_system_prompt(
        depth=0,
        max_depth=2,
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        max_prompt_tokens=120000,
        turn_max_tokens=1024,
        subcall_max_tokens=1024,
        subcall_budget_enabled=True,
        max_total_subcalls=40,
        max_batched_subcalls=12,
    )

    assert "subcall budget is 40 total calls" in prompt
    assert "batched calls may schedule at most 12 prompts" in prompt


def test_prompt_removes_mandatory_recursive_call_pressure() -> None:
    prompt = get_system_prompt_template(DEFAULT_PROMPT_VARIANT)

    assert "make at least one" not in prompt.lower()
    assert "When recursion budget remains" not in prompt


@pytest.mark.parametrize("prompt_variant", sorted(PROMPT_VARIANTS))
def test_all_built_system_prompt_variants_include_budget_reminder(prompt_variant: str) -> None:
    prompt = build_system_prompt(
        depth=0,
        max_depth=2,
        prompt_variant=prompt_variant,
        max_prompt_tokens=120000,
        turn_max_tokens=1024,
        subcall_max_tokens=1024,
    )

    assert prompt.count("Budget reminder:") == 1
    assert "context-aware" in prompt
    assert "avoid context rot" in prompt
    assert "subcall budget is" not in prompt


def test_initial_messages_include_system_context_metadata_and_question() -> None:
    messages = build_initial_messages(
        context_payload="alpha beta gamma",
        root_prompt="What is in the context?",
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        max_prompt_tokens=120000,
        turn_max_tokens=1024,
        subcall_max_tokens=1024,
    )

    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert "You are an RLM (Recursive Language Model) agent" in messages[0]["content"]
    assert "Budget reminder: your prompt context window is 120000 prompt tokens" in messages[0]["content"]
    assert "16 total characters" in messages[1]["content"]
    assert "What is in the context?" in messages[2]["content"]


def test_safer_8xa100_config_reduces_rollout_pressure_and_enables_budget() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "rlm_rlvr"
        / "qwen3_4b_instruct_sanjaya_medium_8xa100_40gb_budgeted.toml"
    )

    config = tomllib.loads(config_path.read_text())

    assert config["orchestrator"]["oversampling_factor"] == 1.0
    assert config["orchestrator"]["max_concurrent"] == 32
    assert config["orchestrator"]["env"][0]["args"]["subcall_budget_enabled"] is True
    assert config["orchestrator"]["env"][0]["args"]["max_total_subcalls"] == 40
    assert config["orchestrator"]["env"][0]["args"]["max_batched_subcalls"] == 40
    assert config["orchestrator"]["eval"]["env"][0]["args"]["subcall_budget_enabled"] is True
