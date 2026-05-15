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


def test_build_rubric_clips_exact_match_reward_at_zero(monkeypatch) -> None:
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
        "efficiency_penalty_coef": 2.0,
        "rlm_segments": [
            {
                "prompt_ids": list(range(600)),
                "completion_ids": list(range(400)),
            }
        ],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 0.0
    assert state["reward_correctness"] == 1.0
    assert state["reward_efficiency_penalty"] == 2.0
    assert state["reward_total"] == 0.0


def test_build_rubric_no_answer_reward_is_zero_with_cost_penalty(monkeypatch) -> None:
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
        "final_answer": "",
        "efficiency_penalty_coef": 0.02,
        "rlm_segments": [
            {
                "prompt_ids": list(range(600)),
                "completion_ids": list(range(400)),
            }
        ],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 0.0
    assert state["reward_correctness"] == 0.0
    assert state["reward_efficiency_penalty"] == 0.02
    assert state["reward_total"] == 0.0


def test_build_rubric_incorrect_judge_reward_is_zero_with_cost_penalty(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    async def _judge_zero(*args, **kwargs):
        del args, kwargs
        return 0.0, "0", None

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(reward_module, "_call_binary_judge", _judge_zero)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "wrong",
        "efficiency_penalty_coef": 0.02,
        "rlm_segments": [
            {
                "prompt_ids": list(range(600)),
                "completion_ids": list(range(400)),
            }
        ],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 0.0
    assert state["reward_correctness"] == 0.0
    assert state["reward_efficiency_penalty"] == 0.02
    assert state["reward_total"] == 0.0


def test_vertex_judge_calls_generate_content_and_parses_binary(monkeypatch) -> None:
    class _FakeThinkingConfig:
        def __init__(self, *, thinking_level: str) -> None:
            self.thinking_level = thinking_level

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeAioModels:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(text="1")

    class _FakeClient:
        last_client = None

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.aio = SimpleNamespace(models=_FakeAioModels())
            _FakeClient.last_client = self

    fake_types = SimpleNamespace(ThinkingConfig=_FakeThinkingConfig, GenerateContentConfig=_FakeGenerateContentConfig)
    monkeypatch.setattr(reward_module, "_load_google_genai", lambda: (SimpleNamespace(Client=_FakeClient), fake_types))

    rubric = build_rubric(
        judge_provider="vertex",
        judge_model="gemini-3-flash-preview",
        judge_base_url="",
        judge_api_key=None,
        judge_vertex_project="test-project",
        judge_vertex_location="global",
        judge_thinking_level="medium",
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "forty two",
        "efficiency_penalty_coef": 0.0,
        "rlm_segments": [],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 1.0
    client = _FakeClient.last_client
    assert client.kwargs == {"vertexai": True, "project": "test-project", "location": "global"}
    call = client.aio.models.calls[0]
    assert call["model"] == "gemini-3-flash-preview"
    assert call["config"].kwargs["thinking_config"].thinking_level == "medium"
    assert call["config"].kwargs["max_output_tokens"] == 1024
    assert state["judge_raw_response"] == "1"


def test_openai_compatible_judge_retries_rate_limits(monkeypatch) -> None:
    async def _no_sleep(attempt: int) -> None:
        del attempt

    class _FakeRateLimitError(Exception):
        status_code = 429

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls < 3:
                raise _FakeRateLimitError("rate limited")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="1"))])

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(reward_module, "_sleep_before_judge_retry", _no_sleep)

    score, raw_response, parse_error = asyncio.run(
        reward_module._call_binary_judge(
            fake_client,
            judge_model="judge-model",
            judge_prompt="prompt",
        )
    )

    assert score == 1.0
    assert raw_response == "1"
    assert parse_error is None
    assert completions.calls == 3


def test_vertex_judge_retries_resource_exhausted(monkeypatch) -> None:
    async def _no_sleep(attempt: int) -> None:
        del attempt

    class _FakeThinkingConfig:
        def __init__(self, *, thinking_level: str) -> None:
            self.thinking_level = thinking_level

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeAioModels:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_content(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return SimpleNamespace(text="1")

    models = _FakeAioModels()
    fake_client = SimpleNamespace(aio=SimpleNamespace(models=models))
    fake_types = SimpleNamespace(ThinkingConfig=_FakeThinkingConfig, GenerateContentConfig=_FakeGenerateContentConfig)
    monkeypatch.setattr(reward_module, "_load_google_genai", lambda: (SimpleNamespace(), fake_types))
    monkeypatch.setattr(reward_module, "_sleep_before_judge_retry", _no_sleep)

    score, raw_response, parse_error = asyncio.run(
        reward_module._call_vertex_binary_judge(
            fake_client,
            judge_model="gemini-3-flash-preview",
            judge_prompt="prompt",
            thinking_level="medium",
        )
    )

    assert score == 1.0
    assert raw_response == "1"
    assert parse_error is None
    assert models.calls == 3


def test_vertex_judge_retries_access_token_type_unsupported(monkeypatch) -> None:
    async def _no_sleep(attempt: int) -> None:
        del attempt

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeVertexJudgeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_content(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("401 UNAUTHENTICATED ACCESS_TOKEN_TYPE_UNSUPPORTED")
            return SimpleNamespace(text="1")

    fake_client = _FakeVertexJudgeClient()
    fake_types = SimpleNamespace(GenerateContentConfig=_FakeGenerateContentConfig)
    monkeypatch.setattr(reward_module, "_load_google_genai", lambda: (SimpleNamespace(), fake_types))
    monkeypatch.setattr(reward_module, "_sleep_before_judge_retry", _no_sleep)

    score, raw_response, parse_error = asyncio.run(
        reward_module._call_vertex_binary_judge(
            fake_client,
            judge_model="gemini-3-flash-preview",
            judge_prompt="prompt",
            thinking_level=None,
        )
    )

    assert score == 1.0
    assert raw_response == "1"
    assert parse_error is None
    assert fake_client.calls == 2
