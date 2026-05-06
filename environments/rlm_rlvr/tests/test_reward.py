from __future__ import annotations

import asyncio
from types import SimpleNamespace

import rlm_rlvr.reward as reward_module
from rlm_rlvr.reward import (
    _efficiency_penalty_from_state,
    _extract_message_text,
    _is_exact_match,
    _parse_binary_judge_score,
    _segment_rollout_token_totals,
    build_rubric,
)


def test_is_exact_match_handles_numeric_normalization() -> None:
    assert _is_exact_match("1,208", ["1208"])
    assert _is_exact_match('"1208"', ["1208"])


def test_is_exact_match_handles_json_normalization() -> None:
    assert _is_exact_match('{"b": 2, "a": 1}', ['{"a":1,"b":2}'])


def test_extract_message_text_falls_back_to_reasoning() -> None:
    message = SimpleNamespace(content=None, reasoning="1")
    assert _extract_message_text(message) == "1"


def test_parse_binary_judge_score_accepts_embedded_binary() -> None:
    assert _parse_binary_judge_score("score: 1") == 1.0


def test_segment_rollout_token_totals_include_non_trainable_subcalls() -> None:
    prompt_tokens, completion_tokens = _segment_rollout_token_totals(
        {
            "rlm_segments": [
                {
                    "kind": "root_turn",
                    "is_trainable_rlm_turn": True,
                    "prompt_ids": [1, 2, 3],
                    "completion_ids": [4, 5],
                },
                {
                    "kind": "plain_query",
                    "is_trainable_rlm_turn": False,
                    "prompt_ids": [10, 11],
                    "completion_ids": [12, 13, 14, 15],
                },
            ]
        }
    )

    assert prompt_tokens == 5
    assert completion_tokens == 6


def test_efficiency_penalty_uses_exact_segment_token_totals() -> None:
    penalty, prompt_tokens, completion_tokens, total_tokens = _efficiency_penalty_from_state(
        {
            "efficiency_penalty_coef": 0.02,
            "rlm_segments": [
                {
                    "prompt_ids": list(range(600)),
                    "completion_ids": list(range(400)),
                }
            ],
        }
    )

    assert prompt_tokens == 600
    assert completion_tokens == 400
    assert total_tokens == 1000
    assert penalty == 0.02


def test_build_rubric_applies_cost_penalty_on_exact_match(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "42",
        "efficiency_penalty_coef": 0.02,
        "rlm_segments": [
            {
                "kind": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_ids": list(range(300)),
                "completion_ids": list(range(200)),
            },
            {
                "kind": "plain_query",
                "is_trainable_rlm_turn": False,
                "prompt_ids": list(range(100)),
                "completion_ids": list(range(50)),
            },
        ],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 1.0 - 0.02 * 0.65
    assert state["reward_correctness"] == 1.0
    assert state["reward_efficiency_penalty"] == 0.02 * 0.65
    assert state["cost_prompt_tokens"] == 400.0
    assert state["cost_completion_tokens"] == 250.0
    assert state["cost_total_tokens"] == 650.0
