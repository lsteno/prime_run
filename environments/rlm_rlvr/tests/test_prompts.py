from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from rlm_rlvr.external_rlm import build_initial_messages, build_system_prompt
from rlm_rlvr.prompt_variants import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
    get_system_prompt_template,
)


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
    assert "not a Python function" in prompt
    assert "the REPL will try to execute it and fail" in prompt
    assert "outside code blocks" in prompt
    assert "FINAL(json.dumps(...))" in prompt
    assert "`FINAL(answer)` returns the literal text" in prompt
    assert "not `FINAL(variable_name)`" in prompt


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
    assert sorted(PROMPT_VARIANTS) == ["default", "sanjaya_text_depth1_llm_only_v1", "sanjaya_text_v1"]
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


def test_prompt_frames_every_depth_rlms_as_orchestrators() -> None:
    prompt = get_system_prompt_template(DEFAULT_PROMPT_VARIANT)

    assert "At every depth" in prompt
    assert "Every RLM at every depth should act as an orchestrator" in prompt
    assert "Use `llm_query_batched` for independent lightweight analyses" in prompt
    assert "Make many small, evidence-carrying subcalls" in prompt


def test_depth1_prompt_variant_does_not_mention_sub_rlm_calls() -> None:
    prompt = get_system_prompt_template("sanjaya_text_depth1_llm_only_v1")

    forbidden = [
        "rlm_query",
        "rlm_query_batched",
        "recursive RLM",
        "recursive child",
        "child agent",
        "child agents",
        "subagent",
        "subagents",
        "sub-RLM",
        "sub RLM",
    ]
    lowered = prompt.lower()
    for phrase in forbidden:
        assert phrase.lower() not in lowered

    assert "`llm_query(prompt, model=None)`" in prompt
    assert "`llm_query_batched(prompts, model=None)`" in prompt
    assert "Act as an orchestrator" in prompt
    assert "Never solve the question entirely yourself" in prompt
    assert "your job is to DELEGATE analysis and semantic work" in prompt
    assert "make at least one focused `llm_query_batched` call before finalizing" in prompt
    assert "FINAL(" in prompt
    assert "FINAL_VAR(" in prompt


def test_depth1_built_system_prompt_does_not_mention_sub_rlm_calls() -> None:
    prompt = build_system_prompt(
        depth=0,
        max_depth=0,
        prompt_variant="sanjaya_text_depth1_llm_only_v1",
        max_prompt_tokens=65536,
        turn_max_tokens=1024,
        subcall_max_tokens=1024,
        subcall_budget_enabled=True,
        max_total_subcalls=60,
        max_batched_subcalls=60,
    )

    forbidden = [
        "rlm_query",
        "rlm_query_batched",
        "recursive RLM",
        "recursive child",
        "child agent",
        "child agents",
        "subagent",
        "subagents",
        "sub-RLM",
        "sub RLM",
    ]
    lowered = prompt.lower()
    for phrase in forbidden:
        assert phrase.lower() not in lowered

    assert "RLM turns may output up to 1024 output tokens" in prompt
    assert "The subcall budget is 60 total LLM subcalls" in prompt


def test_depth1_built_system_prompt_can_omit_budget_reminder_for_trace_generation() -> None:
    prompt = build_system_prompt(
        depth=0,
        max_depth=0,
        prompt_variant="sanjaya_text_depth1_llm_only_v1",
        max_prompt_tokens=200000,
        turn_max_tokens=1024,
        subcall_max_tokens=1024,
        subcall_budget_enabled=True,
        max_total_subcalls=60,
        max_batched_subcalls=60,
        include_budget_reminder=False,
    )

    assert "Budget reminder:" not in prompt
    assert "The subcall budget is" not in prompt
    assert "Never solve the question entirely yourself" in prompt


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


def test_safer_8xa100_config_reduces_rollout_pressure_and_uses_root_policy_subcalls() -> None:
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
    assert config["orchestrator"]["env"][0]["args"]["max_total_subcalls"] == 80
    assert config["orchestrator"]["env"][0]["args"]["max_batched_subcalls"] == 80
    assert config["orchestrator"]["env"][0]["args"]["subcall_batch_max_workers"] == 2
    assert "llm_subcall_provider" not in config["orchestrator"]["env"][0]["args"]
    assert "llm_subcall_model" not in config["orchestrator"]["env"][0]["args"]
    assert config["orchestrator"]["env"][0]["args"]["judge_provider"] == "vertex"
    assert config["orchestrator"]["env"][0]["args"]["judge_model"] == "gemini-3-flash-preview"
    assert config["orchestrator"]["env"][0]["args"]["judge_vertex_location"] == "global"
    assert config["orchestrator"]["env"][0]["args"]["judge_thinking_level"] == "medium"
    assert config["orchestrator"]["eval"]["env"][0]["args"]["subcall_budget_enabled"] is True
    assert config["orchestrator"]["eval"]["env"][0]["args"]["subcall_batch_max_workers"] == 2
    assert "llm_subcall_provider" not in config["orchestrator"]["eval"]["env"][0]["args"]
    assert "llm_subcall_model" not in config["orchestrator"]["eval"]["env"][0]["args"]


def test_budgeted_configs_route_plain_llm_subcalls_to_root_policy_and_judge_to_vertex() -> None:
    config_names = [
        "qwen3_4b_instruct_sanjaya_medium_4xa100_80gb_budgeted.toml",
        "qwen3_4b_instruct_sanjaya_medium_8xa100_40gb_budgeted.toml",
        "qwen3_4b_instruct_sanjaya_medium_8xrtx6000ada_48gb_budgeted.toml",
    ]
    for config_name in config_names:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "rlm_rlvr" / config_name
        config = tomllib.loads(config_path.read_text())
        train_args = config["orchestrator"]["env"][0]["args"]
        eval_args = config["orchestrator"]["eval"]["env"][0]["args"]

        assert train_args["efficiency_penalty_mode"] == "adaptive_group"
        assert train_args["adaptive_efficiency_beta_max"] == 0.05
        assert train_args["adaptive_efficiency_gamma"] == 2.0
        assert train_args["adaptive_efficiency_solve_rate_floor"] == 0.25
        assert train_args["adaptive_efficiency_cost_basis"] == "total_tokens"
        assert eval_args["efficiency_penalty_mode"] == "static_per_1k"
        assert eval_args["efficiency_penalty_coef"] == 0.0

        for args in (train_args, eval_args):
            assert args["inference_mode"] == "local"
            assert args["inference_base_url"] == "http://localhost:8000/v1"
            assert args["inference_api_key"] == "local-vllm"
            assert args["subcall_batch_max_workers"] == 2
            assert "llm_subcall_provider" not in args
            assert "llm_subcall_model" not in args
            assert "llm_subcall_vertex_project_env" not in args
            assert "llm_subcall_vertex_location" not in args
            assert "llm_subcall_thinking_level" not in args
            assert args["judge_provider"] == "vertex"
            assert args["judge_model"] == "gemini-3-flash-preview"
            assert args["judge_vertex_project_env"] == "GOOGLE_CLOUD_PROJECT"
            assert args["judge_vertex_location"] == "global"
            assert args["judge_thinking_level"] == "medium"
            assert "llm_subcall_base_url" not in args
            assert "llm_subcall_api_key_var" not in args
