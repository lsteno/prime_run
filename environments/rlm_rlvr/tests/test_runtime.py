from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI

import rlm_rlvr.runtime as runtime_module
from rlm_rlvr.env import RLMRLVREnv, load_environment
from rlm_rlvr.prompt_variants import DEFAULT_PROMPT_VARIANT
from rlm_rlvr.repl import RecursiveLocalRepl, code_uses_subcalls
from rlm_rlvr.runtime import RecursiveRuntime, RuntimeConfig, SubcallPromptTooLargeError, SyncInferenceSession, TokenPayload, VertexGeminiSession
from rlm_rlvr.trace import make_call_trace


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool, return_dict: bool):
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_dict is True
        return {"input_ids": [101, 102, 103]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(text), len(text) + 1] if text else []


class _BudgetTokenizer:
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool, return_dict: bool):
        assert tokenize is True
        assert return_dict is True
        input_ids: list[int] = []
        for index, message in enumerate(messages):
            input_ids.append(1000 + index)
            input_ids.extend(self.encode(message["content"]))
        if add_generation_prompt:
            input_ids.append(2000)
        return {"input_ids": input_ids}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(part) for part in text.split() if part]


class _FakeClient:
    def __init__(self, *, expect_logprobs: bool = True, usage=None) -> None:
        self.bodies: list[dict] = []
        self.expect_logprobs = expect_logprobs
        self.usage = usage

    def post(self, path: str, *, body, cast_to):
        del cast_to
        self.bodies.append(body)
        assert path == "chat/completions"
        if self.expect_logprobs:
            assert body["logprobs"] is True
        else:
            assert "logprobs" not in body
        choice = SimpleNamespace(
            message=SimpleNamespace(content="ok"),
            token_ids=None,
            logprobs=None,
        )
        return SimpleNamespace(choices=[choice], prompt_token_ids=None, usage=self.usage)


class _FakeSyncInferenceSession:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def count_text_tokens(self, text: str) -> int:
        return len(text.split())


class _FakeVertexGeminiSession:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def count_text_tokens(self, text: str) -> int:
        return len(text.split())


def _runtime_state(**overrides):
    state = {
        "current_call_depth": 0,
        "current_call_id": 0,
        "current_parent_call_id": None,
        "current_branch_max_depth": 2,
        "rlm_segment_counter": 0,
        "rlm_call_counter": 1,
        "rlm_segments": [],
        "rlm_trace": [],
        "total_model_tokens": 0.0,
        "max_depth_reached": 0,
        "used_repl": False,
        "used_recursion": False,
        "used_llm_subcalls": False,
        "used_rlm_subcalls": False,
        "num_subcalls": 0,
        "num_llm_subcalls": 0,
        "num_rlm_subcalls": 0,
        "subcall_budget_enabled": False,
        "subcall_budget_total": 40,
        "subcall_budget_remaining": 40,
        "subcall_budget_exhausted": False,
        "sampling_temperature": 0.0,
    }
    state.update(overrides)
    return state


def test_generate_falls_back_when_token_metadata_is_missing() -> None:
    session = object.__new__(SyncInferenceSession)
    session.model_name = "fake-model"
    session.client = _FakeClient()
    session.tokenizer = _FakeTokenizer()
    session.max_prompt_tokens = None
    session.enable_vllm_extra_body = True

    text, payload = SyncInferenceSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert text == "ok"
    assert payload.prompt_ids == [101, 102, 103]
    assert payload.completion_ids == [2, 3]
    assert payload.completion_logprobs == [0.0, 0.0]
    assert payload.completion_mask == [True, True]
    assert session.client.bodies[0]["return_token_ids"] is True
    assert session.client.bodies[0]["top_k"] == -1
    assert session.client.bodies[0]["min_p"] == 0.0
    assert "extra_body" not in session.client.bodies[0]


def test_generate_does_not_send_vllm_extras_for_hosted_mode() -> None:
    session = object.__new__(SyncInferenceSession)
    session.model_name = "fake-model"
    session.client = _FakeClient()
    session.tokenizer = _FakeTokenizer()
    session.max_prompt_tokens = None
    session.enable_vllm_extra_body = False

    SyncInferenceSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert "return_token_ids" not in session.client.bodies[0]
    assert "top_k" not in session.client.bodies[0]
    assert "min_p" not in session.client.bodies[0]
    assert "extra_body" not in session.client.bodies[0]


def test_openai_gpt5_session_uses_max_completion_tokens_field() -> None:
    session = SyncInferenceSession(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        default_headers=None,
        model_name="gpt-5.4",
        tokenizer_name=None,
        max_prompt_tokens=None,
        request_logprobs=False,
        enable_token_accounting=False,
    )
    session.client = _FakeClient(expect_logprobs=False)

    SyncInferenceSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert session.client.bodies[0]["max_completion_tokens"] == 8
    assert "max_tokens" not in session.client.bodies[0]


def test_generate_uses_api_usage_token_counts_for_plain_subcalls() -> None:
    session = object.__new__(SyncInferenceSession)
    session.model_name = "openai/gpt-5.4-mini"
    session.client = _FakeClient(expect_logprobs=False, usage=SimpleNamespace(prompt_tokens=17, completion_tokens=9))
    session.tokenizer = _FakeTokenizer()
    session.max_prompt_tokens = None
    session.enable_vllm_extra_body = False
    session.request_logprobs = False

    text, payload = SyncInferenceSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert text == "ok"
    assert payload.prompt_ids == [101, 102, 103]
    assert payload.completion_ids == [2, 3]
    assert payload.prompt_token_count == 17
    assert payload.completion_token_count == 9


def test_openai_compatible_plain_subcall_retries_transient_errors(monkeypatch) -> None:
    class _FakeRateLimitError(RuntimeError):
        status_code = 429

    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, path: str, *, body, cast_to):
            del path, body, cast_to
            self.calls += 1
            if self.calls < 3:
                raise _FakeRateLimitError("rate limited")
            choice = SimpleNamespace(message=SimpleNamespace(content="ok"), token_ids=None, logprobs=None)
            usage = SimpleNamespace(prompt_tokens=17, completion_tokens=9)
            return SimpleNamespace(choices=[choice], prompt_token_ids=None, usage=usage)

    monkeypatch.setattr(runtime_module, "_sleep_before_subcall_retry", lambda attempt: None)
    session = object.__new__(SyncInferenceSession)
    session.model_name = "openai/gpt-5.4-mini"
    session.client = _FlakyClient()
    session.tokenizer = _FakeTokenizer()
    session.max_prompt_tokens = None
    session.enable_vllm_extra_body = False
    session.request_logprobs = False
    session.retry_transient_errors = True

    text, payload = SyncInferenceSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert text == "ok"
    assert session.client.calls == 3
    assert payload.prompt_token_count == 17
    assert payload.completion_token_count == 9


def test_local_sync_generation_does_not_retry_by_default(monkeypatch) -> None:
    class _FakeUnavailableError(RuntimeError):
        status_code = 503

    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, path: str, *, body, cast_to):
            del path, body, cast_to
            self.calls += 1
            raise _FakeUnavailableError("unavailable")

    monkeypatch.setattr(runtime_module, "_sleep_before_subcall_retry", lambda attempt: None)
    session = object.__new__(SyncInferenceSession)
    session.model_name = "local-training-model"
    session.client = _FlakyClient()
    session.tokenizer = _FakeTokenizer()
    session.max_prompt_tokens = None
    session.enable_vllm_extra_body = True
    session.request_logprobs = True

    with pytest.raises(_FakeUnavailableError):
        SyncInferenceSession.generate(
            session,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )

    assert session.client.calls == 1


def test_generate_falls_back_to_tokenizer_counts_when_usage_is_missing() -> None:
    session = object.__new__(SyncInferenceSession)
    session.model_name = "openai/gpt-5.4-mini"
    session.client = _FakeClient(expect_logprobs=False, usage=None)
    session.tokenizer = _FakeTokenizer()
    session.max_prompt_tokens = None
    session.enable_vllm_extra_body = False
    session.request_logprobs = False

    _, payload = SyncInferenceSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert payload.prompt_token_count == 3
    assert payload.completion_token_count == 2


def test_generate_can_skip_token_accounting_for_external_traces() -> None:
    session = object.__new__(SyncInferenceSession)
    session.model_name = "z-ai/glm-5"
    session.client = _FakeClient(expect_logprobs=False, usage=None)
    session.tokenizer = None
    session.max_prompt_tokens = None
    session.enable_vllm_extra_body = False
    session.request_logprobs = False
    session.retry_transient_errors = False
    session.openai_extra_body = {}
    session.enable_token_accounting = False

    text, payload = SyncInferenceSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert text == "ok"
    assert payload.prompt_ids == []
    assert payload.completion_ids == []
    assert payload.prompt_token_count == 0
    assert payload.completion_token_count == 0


def test_vertex_generate_uses_usage_metadata_and_includes_thinking_tokens(monkeypatch) -> None:
    class _FakePart:
        @staticmethod
        def from_text(*, text: str):
            return {"text": text}

    class _FakeContent:
        def __init__(self, *, role: str, parts: list[dict[str, str]]) -> None:
            self.role = role
            self.parts = parts

    class _FakeThinkingConfig:
        def __init__(self, *, thinking_level: str) -> None:
            self.thinking_level = thinking_level

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        Part=_FakePart,
        Content=_FakeContent,
        ThinkingConfig=_FakeThinkingConfig,
        GenerateContentConfig=_FakeGenerateContentConfig,
    )
    monkeypatch.setattr(runtime_module, "_load_google_genai", lambda: (SimpleNamespace(), fake_types))

    class _FakeModels:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                text="ok",
                usage_metadata=SimpleNamespace(prompt_token_count=17, total_token_count=31, candidates_token_count=3),
            )

    session = object.__new__(VertexGeminiSession)
    session.model_name = "gemini-3.1-flash-lite"
    session.client = SimpleNamespace(models=_FakeModels())
    session.tokenizer = _FakeTokenizer()
    session.thinking_level = "medium"

    text, payload = VertexGeminiSession.generate(
        session,
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert text == "ok"
    assert payload.prompt_ids == [101, 102, 103]
    assert payload.completion_ids == [2, 3]
    assert payload.prompt_token_count == 17
    assert payload.completion_token_count == 14
    call = session.client.models.calls[0]
    assert call["model"] == "gemini-3.1-flash-lite"
    assert call["config"].kwargs["thinking_config"].thinking_level == "medium"


def test_vertex_plain_subcall_retries_transient_errors(monkeypatch) -> None:
    class _FakePart:
        @staticmethod
        def from_text(*, text: str):
            return {"text": text}

    class _FakeContent:
        def __init__(self, *, role: str, parts: list[dict[str, str]]) -> None:
            self.role = role
            self.parts = parts

    class _FakeThinkingConfig:
        def __init__(self, *, thinking_level: str) -> None:
            self.thinking_level = thinking_level

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        Part=_FakePart,
        Content=_FakeContent,
        ThinkingConfig=_FakeThinkingConfig,
        GenerateContentConfig=_FakeGenerateContentConfig,
    )
    monkeypatch.setattr(runtime_module, "_load_google_genai", lambda: (SimpleNamespace(), fake_types))
    monkeypatch.setattr(runtime_module, "_sleep_before_subcall_retry", lambda attempt: None)

    class _FakeModels:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return SimpleNamespace(
                text="ok",
                usage_metadata=SimpleNamespace(prompt_token_count=17, total_token_count=31),
            )

    session = object.__new__(VertexGeminiSession)
    session.model_name = "gemini-3.1-flash-lite"
    session.client = SimpleNamespace(models=_FakeModels())
    session.tokenizer = _FakeTokenizer()
    session.thinking_level = "medium"

    text, payload = VertexGeminiSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert text == "ok"
    assert session.client.models.calls == 3
    assert payload.prompt_token_count == 17
    assert payload.completion_token_count == 14


def test_vertex_generate_falls_back_to_tokenizer_counts_when_usage_is_missing(monkeypatch) -> None:
    class _FakePart:
        @staticmethod
        def from_text(*, text: str):
            return {"text": text}

    class _FakeContent:
        def __init__(self, *, role: str, parts: list[dict[str, str]]) -> None:
            self.role = role
            self.parts = parts

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        Part=_FakePart,
        Content=_FakeContent,
        ThinkingConfig=lambda **kwargs: kwargs,
        GenerateContentConfig=_FakeGenerateContentConfig,
    )
    monkeypatch.setattr(runtime_module, "_load_google_genai", lambda: (SimpleNamespace(), fake_types))

    class _FakeModels:
        def generate_content(self, **kwargs):
            del kwargs
            return SimpleNamespace(text="ok")

    session = object.__new__(VertexGeminiSession)
    session.model_name = "gemini-3.1-flash-lite"
    session.client = SimpleNamespace(models=_FakeModels())
    session.tokenizer = _FakeTokenizer()
    session.thinking_level = "medium"

    _, payload = VertexGeminiSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert payload.prompt_token_count == 3
    assert payload.completion_token_count == 2


def test_vertex_generate_retries_empty_responses(monkeypatch) -> None:
    class _FakePart:
        @staticmethod
        def from_text(*, text: str):
            return {"text": text}

    class _FakeContent:
        def __init__(self, *, role: str, parts: list[dict[str, str]]) -> None:
            self.role = role
            self.parts = parts

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        Part=_FakePart,
        Content=_FakeContent,
        ThinkingConfig=lambda **kwargs: kwargs,
        GenerateContentConfig=_FakeGenerateContentConfig,
    )
    monkeypatch.setattr(runtime_module, "_load_google_genai", lambda: (SimpleNamespace(), fake_types))
    monkeypatch.setattr(VertexGeminiSession, "_sleep_before_empty_response_retry", lambda self, attempt: None)

    class _FakeModels:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(text="", usage_metadata=SimpleNamespace(prompt_token_count=5, total_token_count=7))
            return SimpleNamespace(text="ok", usage_metadata=SimpleNamespace(prompt_token_count=6, total_token_count=10))

    session = object.__new__(VertexGeminiSession)
    session.model_name = "gemini-3.1-flash-lite"
    session.client = SimpleNamespace(models=_FakeModels())
    session.tokenizer = _FakeTokenizer()
    session.thinking_level = "medium"
    session.empty_response_max_attempts = 3
    session.empty_response_base_retry_seconds = 1.0
    session.empty_response_max_retry_seconds = 30.0

    text, payload = VertexGeminiSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert text == "ok"
    assert session.client.models.calls == 2
    assert payload.prompt_token_count == 11
    assert payload.completion_token_count == 6
    assert payload.metadata == {
        "generation_attempt_count": 2,
        "empty_response_retry_count": 1,
        "empty_response_exhausted": False,
    }


def test_vertex_generate_marks_empty_response_exhaustion(monkeypatch) -> None:
    class _FakePart:
        @staticmethod
        def from_text(*, text: str):
            return {"text": text}

    class _FakeContent:
        def __init__(self, *, role: str, parts: list[dict[str, str]]) -> None:
            self.role = role
            self.parts = parts

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_types = SimpleNamespace(
        Part=_FakePart,
        Content=_FakeContent,
        ThinkingConfig=lambda **kwargs: kwargs,
        GenerateContentConfig=_FakeGenerateContentConfig,
    )
    monkeypatch.setattr(runtime_module, "_load_google_genai", lambda: (SimpleNamespace(), fake_types))
    monkeypatch.setattr(VertexGeminiSession, "_sleep_before_empty_response_retry", lambda self, attempt: None)

    class _FakeModels:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, **kwargs):
            del kwargs
            self.calls += 1
            return SimpleNamespace(text="")

    session = object.__new__(VertexGeminiSession)
    session.model_name = "gemini-3.1-flash-lite"
    session.client = SimpleNamespace(models=_FakeModels())
    session.tokenizer = _FakeTokenizer()
    session.thinking_level = "medium"
    session.empty_response_max_attempts = 2
    session.empty_response_base_retry_seconds = 1.0
    session.empty_response_max_retry_seconds = 30.0

    text, payload = VertexGeminiSession.generate(
        session,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert text == ""
    assert session.client.models.calls == 2
    assert payload.completion_ids == []
    assert payload.metadata == {
        "generation_attempt_count": 2,
        "empty_response_retry_count": 1,
        "empty_response_exhausted": True,
    }


def test_setup_state_rebuilds_root_prompt_with_real_context_metadata(monkeypatch) -> None:
    import rlm_rlvr.env as env_module

    monkeypatch.setattr(env_module, "SyncInferenceSession", _FakeSyncInferenceSession)
    environment = object.__new__(RLMRLVREnv)
    environment.runtime_config = RuntimeConfig(prompt_variant=DEFAULT_PROMPT_VARIANT)
    environment.efficiency_penalty_coef = 0.02
    state = {
        "client": AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY"),
        "model": "fake-model",
        "info": {
            "context": "alpha beta gamma",
            "question": "What is in the context?",
        },
        "sampling_args": {},
    }

    updated = asyncio.run(environment.setup_state(state))

    assert [message["role"] for message in updated["prompt"]] == ["system", "user", "user"]
    assert "16 total characters" in updated["prompt"][1]["content"]
    assert "What is in the context?" in updated["prompt"][2]["content"]
    assert updated["rlm_call_counter"] == 1
    assert updated["rlm_trace"][0]["call_id"] == 0
    assert updated["rlm_trace"][0]["depth"] == 0
    assert updated["rlm_trace"][0]["prompt"] == "What is in the context?"
    assert updated["used_llm_subcalls"] is False
    assert updated["used_rlm_subcalls"] is False


def test_setup_state_creates_separate_vertex_plain_llm_session(monkeypatch) -> None:
    import rlm_rlvr.env as env_module

    monkeypatch.setattr(env_module, "SyncInferenceSession", _FakeSyncInferenceSession)
    monkeypatch.setattr(env_module, "VertexGeminiSession", _FakeVertexGeminiSession)
    environment = object.__new__(RLMRLVREnv)
    environment.runtime_config = RuntimeConfig(
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        llm_subcall_provider="vertex",
        llm_subcall_model="gemini-3.1-flash-lite",
        llm_subcall_vertex_project="test-project",
        llm_subcall_vertex_location="global",
        llm_subcall_thinking_level="medium",
        llm_subcall_empty_response_max_attempts=3,
    )
    environment.efficiency_penalty_coef = 0.02
    state = {
        "client": AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY"),
        "model": "local-training-model",
        "info": {
            "context": "alpha beta gamma",
            "question": "What is in the context?",
        },
        "sampling_args": {},
    }

    updated = asyncio.run(environment.setup_state(state))

    local_session = updated["_sync_session"]
    plain_session = updated["_plain_llm_session"]
    assert plain_session is not local_session
    assert local_session.kwargs["model_name"] == "local-training-model"
    assert local_session.kwargs["enable_vllm_extra_body"] is False
    assert plain_session.kwargs["model_name"] == "gemini-3.1-flash-lite"
    assert plain_session.kwargs["project"] == "test-project"
    assert plain_session.kwargs["location"] == "global"
    assert plain_session.kwargs["thinking_level"] == "medium"
    assert plain_session.kwargs["empty_response_max_attempts"] == 3


def test_setup_state_enables_retries_for_openai_compatible_plain_llm_session(monkeypatch) -> None:
    import rlm_rlvr.env as env_module

    monkeypatch.setattr(env_module, "SyncInferenceSession", _FakeSyncInferenceSession)
    environment = object.__new__(RLMRLVREnv)
    environment.runtime_config = RuntimeConfig(
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        llm_subcall_provider="openai_compatible",
        llm_subcall_model="openai/gpt-5.4-mini",
        llm_subcall_base_url="https://openrouter.ai/api/v1",
        llm_subcall_api_key="test-key",
    )
    environment.efficiency_penalty_coef = 0.02
    state = {
        "client": AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY"),
        "model": "local-training-model",
        "info": {
            "context": "alpha beta gamma",
            "question": "What is in the context?",
        },
        "sampling_args": {},
    }

    updated = asyncio.run(environment.setup_state(state))

    local_session = updated["_sync_session"]
    plain_session = updated["_plain_llm_session"]
    assert plain_session is not local_session
    assert "retry_transient_errors" not in local_session.kwargs
    assert plain_session.kwargs["model_name"] == "openai/gpt-5.4-mini"
    assert plain_session.kwargs["request_logprobs"] is False
    assert plain_session.kwargs["retry_transient_errors"] is True


def test_env_response_records_root_repl_feedback(tmp_path) -> None:
    environment = object.__new__(RLMRLVREnv)
    environment.runtime_config = RuntimeConfig(
        max_iterations=4,
        execution_output_char_limit=4000,
        live_trace_dir=str(tmp_path),
    )

    root_trace = make_call_trace(call_id=0, depth=0, prompt="What happened?")
    state = {
        "_runtime": SimpleNamespace(build_finalize_message=lambda: "finalize"),
        "_root_repl": RecursiveLocalRepl(
            context_payload="root context",
            llm_query_fn=lambda prompt, model: {"prompt": prompt, "model": model or "test-model", "response": prompt},
            rlm_query_fn=lambda prompt, model, max_depth: {"prompt": prompt, "model": model or "test-model", "response": prompt},
        ),
        "_sync_session": _FakeSyncInferenceSession(),
        "_root_trace": root_trace,
        "rlm_trace": [root_trace],
        "trajectory": [],
        "info": {"question": "What happened?", "source_id": "sample-1"},
        "prompt_variant": "default",
        "live_trace_dir": str(tmp_path),
        "used_repl": False,
        "used_recursion": False,
        "used_llm_subcalls": False,
        "used_rlm_subcalls": False,
        "final_answer": None,
        "total_env_tokens": 0.0,
        "max_depth_reached": 0,
        "num_subcalls": 0,
        "num_llm_subcalls": 0,
        "num_rlm_subcalls": 0,
        "rlm_segments": [],
    }

    response = asyncio.run(
        environment.env_response(
            [{"role": "assistant", "content": "```repl\nprint('hello from root')\n```"}],
            state,
        )
    )

    assert response
    assert len(root_trace["steps"]) == 1
    step = root_trace["steps"][0]
    assert step["code_blocks"] == ["print('hello from root')"]
    assert any("hello from root" in feedback for feedback in step["feedback"])
    assert step["final_answer"] is None

    live_trace_path = tmp_path / "default" / "sample-1.json"
    assert live_trace_path.exists()
    live_trace = json.loads(live_trace_path.read_text())
    assert live_trace["event"] == "root_feedback"
    assert live_trace["traces"][0]["steps"][0]["code_blocks"] == ["print('hello from root')"]
    assert any("hello from root" in feedback for feedback in live_trace["traces"][0]["steps"][0]["feedback"])


def _root_env_response_state(tmp_path):
    root_trace = make_call_trace(call_id=0, depth=0, prompt="What happened?")
    return {
        "_runtime": SimpleNamespace(build_finalize_message=lambda: "finalize"),
        "_root_repl": RecursiveLocalRepl(
            context_payload="root context",
            llm_query_fn=lambda prompt, model: {"prompt": prompt, "model": model or "test-model", "response": prompt},
            rlm_query_fn=lambda prompt, model, max_depth: {
                "prompt": prompt,
                "model": model or "test-model",
                "response": prompt,
            },
        ),
        "_sync_session": _FakeSyncInferenceSession(),
        "_root_trace": root_trace,
        "rlm_trace": [root_trace],
        "trajectory": [],
        "info": {"question": "What happened?", "source_id": "sample-1"},
        "prompt_variant": "default",
        "live_trace_dir": str(tmp_path),
        "used_repl": False,
        "used_recursion": False,
        "used_llm_subcalls": False,
        "used_rlm_subcalls": False,
        "final_answer": None,
        "used_forced_finalize_prompt": False,
        "hit_max_turn_without_final": False,
        "missing_final": False,
        "finalized_before_forced_prompt": False,
        "finalized_on_forced_prompt": False,
        "total_env_tokens": 0.0,
        "max_depth_reached": 0,
        "num_subcalls": 0,
        "num_llm_subcalls": 0,
        "num_rlm_subcalls": 0,
        "rlm_segments": [],
    }


def test_env_response_processes_forced_finalize_final_var_repl(tmp_path) -> None:
    environment = object.__new__(RLMRLVREnv)
    environment.runtime_config = RuntimeConfig(
        max_iterations=1,
        execution_output_char_limit=4000,
        live_trace_dir=str(tmp_path),
    )
    state = _root_env_response_state(tmp_path)
    state["used_forced_finalize_prompt"] = True
    state["_root_repl"].execute_code('answer = "42"')

    response = asyncio.run(
        environment.env_response(
            [{"role": "assistant", "content": '```repl\nFINAL_VAR("answer")\n```'}],
            state,
        )
    )

    assert response == []
    assert state["final_answer"] == "42"
    assert state["finalized_on_forced_prompt"] is True
    assert state["finalized_before_forced_prompt"] is False
    assert state["hit_max_turn_without_final"] is False


def test_env_response_marks_missing_final_after_forced_finalize(tmp_path) -> None:
    environment = object.__new__(RLMRLVREnv)
    environment.runtime_config = RuntimeConfig(
        max_iterations=1,
        execution_output_char_limit=4000,
        live_trace_dir=str(tmp_path),
    )
    state = _root_env_response_state(tmp_path)
    state["used_forced_finalize_prompt"] = True

    response = asyncio.run(
        environment.env_response(
            [{"role": "assistant", "content": "42"}],
            state,
        )
    )

    assert response == []
    assert state["final_answer"] is None
    assert state["hit_max_turn_without_final"] is True
    assert state["missing_final"] is True
    assert state["finalized_on_forced_prompt"] is False


def test_add_trajectory_step_marks_root_turn_trainable_with_prompt_provenance(monkeypatch) -> None:
    import rlm_rlvr.env as env_module

    async def _fake_add_trajectory_step(self, state, trajectory_step):
        del self
        state.setdefault("trajectory", []).append(trajectory_step)

    monkeypatch.setattr(env_module.vf.MultiTurnEnv, "add_trajectory_step", _fake_add_trajectory_step)
    environment = object.__new__(RLMRLVREnv)
    environment.runtime_config = RuntimeConfig(temperature=0.7, live_trace_dir=None)
    state = {
        "trajectory": [],
        "rlm_segment_counter": 0,
        "rlm_segments": [],
        "total_model_tokens": 0.0,
        "sampling_args": {"temperature": 0.7},
    }
    trajectory_step = {
        "prompt": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
        ],
        "completion": [{"role": "assistant", "content": "answer"}],
        "tokens": {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [4, 5],
            "completion_logprobs": [-0.1, -0.2],
            "completion_mask": [True, True],
        },
        "extras": {},
    }

    asyncio.run(environment.add_trajectory_step(state, trajectory_step))

    segment = state["rlm_segments"][0]
    assert trajectory_step["extras"]["rlm_segment_order"] == 0
    assert segment["call_id"] == 0
    assert segment["parent_call_id"] is None
    assert segment["turn_index"] == 0
    assert segment["train_scope"] == "root_turn"
    assert segment["is_trainable_rlm_turn"] is True
    assert segment["response_source"] == "root"
    assert segment["prompt_message_count"] == 2
    assert segment["prompt_char_count"] == len("system") + len("sys") + len("user") + len("question")


def test_generate_preserves_messages_without_prompt_fitting() -> None:
    client = _FakeClient()
    session = object.__new__(SyncInferenceSession)
    session.model_name = "fake-model"
    session.client = client
    session.tokenizer = _BudgetTokenizer()
    session.max_prompt_tokens = 10
    session.enable_vllm_extra_body = True

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "drop these tokens first please"},
        {"role": "user", "content": "keep this question available"},
    ]

    SyncInferenceSession.generate(
        session,
        messages=messages,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert client.bodies, "expected a completion request"
    assert client.bodies[0]["messages"] == messages


def test_recursive_local_repl_routes_llm_and_rlm_queries() -> None:
    repl = RecursiveLocalRepl(
        context_payload="context",
        llm_query_fn=lambda prompt, model: {
            "prompt": prompt,
            "model": model or "test-model",
            "response": "plain-response",
            "kind": "plain_query",
            "depth": 1,
            "execution_time": 0.01,
        },
        rlm_query_fn=lambda prompt, model, max_depth: {
            "prompt": prompt,
            "model": model or "test-model",
            "response": f"recursive-response:{max_depth}",
            "kind": "recursive_query",
            "depth": 1,
            "trace": [{"call_id": 1, "steps": []}],
            "execution_time": 0.02,
        },
    )

    result = repl.execute_code(
        "print(llm_query('alpha'))\n"
        "print(rlm_query('beta', max_depth=3))\n"
    )

    assert "plain-response" in result.stdout
    assert "recursive-response:3" in result.stdout
    assert result.stderr == ""
    assert [call.metadata["kind"] for call in result.rlm_calls] == ["plain_query", "recursive_query"]


def test_code_uses_subcalls_detects_query_helpers() -> None:
    assert code_uses_subcalls("answer = llm_query('alpha')")
    assert code_uses_subcalls("answers = rlm_query_batched(['alpha'])")
    assert not code_uses_subcalls("answer = sum([1, 2, 3])")


def test_recursive_local_repl_fast_timeout_interrupts_local_code() -> None:
    repl = RecursiveLocalRepl(
        context_payload="context",
        llm_query_fn=lambda prompt, model: {"prompt": prompt, "model": model, "response": "ok"},
        rlm_query_fn=lambda prompt, model, max_depth: {"prompt": prompt, "model": model, "response": "ok"},
        repl_timeout_seconds=1.0,
        repl_fast_timeout_seconds=0.01,
    )

    result = repl.execute_code("while True:\n    pass")

    assert "REPL execution timed out after 0.01s" in result.stderr
    assert result.final_answer is None
    assert result.rlm_calls == []


def test_recursive_local_repl_subcall_code_uses_long_timeout() -> None:
    def slow_llm_query(prompt: str, model: str | None) -> dict[str, object]:
        del model
        time.sleep(0.05)
        return {
            "prompt": prompt,
            "model": "test-model",
            "response": "plain-response",
            "kind": "plain_query",
            "depth": 1,
            "execution_time": 0.05,
        }

    repl = RecursiveLocalRepl(
        context_payload="context",
        llm_query_fn=slow_llm_query,
        rlm_query_fn=lambda prompt, model, max_depth: {"prompt": prompt, "model": model, "response": "ok"},
        repl_timeout_seconds=1.0,
        repl_fast_timeout_seconds=0.01,
    )

    result = repl.execute_code("answer = llm_query('alpha')")

    assert result.stderr == ""
    assert result.locals["answer"] == "plain-response"
    assert [call.metadata["kind"] for call in result.rlm_calls] == ["plain_query"]


def test_plain_query_batch_runs_in_parallel() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(_sync_session=SimpleNamespace(model_name="fake-model")),
        RuntimeConfig(max_prompt_tokens=4096, live_trace_dir=None),
    )

    def fake_plain_query(prompt: str, model: str | None = None, consume_budget: bool = True) -> dict[str, object]:
        del model, consume_budget
        time.sleep(0.05)
        return {
            "prompt": prompt,
            "model": "fake-model",
            "response": f"response:{prompt}",
            "kind": "plain_query",
            "depth": 1,
            "execution_time": 0.01,
        }

    runtime._plain_query = fake_plain_query  # type: ignore[method-assign]

    start = time.perf_counter()
    payloads = runtime.run_plain_query_batch(["alpha", "beta", "gamma"], max_workers=3)
    elapsed = time.perf_counter() - start

    assert [payload["response"] for payload in payloads] == [
        "response:alpha",
        "response:beta",
        "response:gamma",
    ]
    assert elapsed < 0.13


def test_recursive_query_batch_runs_serially_by_default() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(_sync_session=SimpleNamespace(model_name="fake-model")),
        RuntimeConfig(max_prompt_tokens=4096, live_trace_dir=None),
    )
    call_order: list[str] = []
    thread_ids: list[int] = []

    def fake_recursive_query(
        prompt: str,
        model: str | None = None,
        max_depth: int | None = None,
        consume_budget: bool = True,
    ) -> dict[str, object]:
        del model, max_depth, consume_budget
        call_order.append(prompt)
        thread_ids.append(threading.get_ident())
        time.sleep(0.01)
        return {
            "prompt": prompt,
            "model": "fake-model",
            "response": f"response:{prompt}",
            "final_answer": f"answer:{prompt}",
            "kind": "recursive_query",
            "depth": 1,
            "trace": [{"call_id": 1, "steps": []}],
            "execution_time": 0.02,
        }

    runtime._recursive_query = fake_recursive_query  # type: ignore[method-assign]

    payloads = runtime.run_recursive_query_batch(["alpha", "beta", "gamma"], max_workers=3)

    assert [payload["response"] for payload in payloads] == [
        "response:alpha",
        "response:beta",
        "response:gamma",
    ]
    assert call_order == ["alpha", "beta", "gamma"]
    assert len(set(thread_ids)) == 1


def test_recursive_query_batch_thread_mode_runs_in_parallel() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(_sync_session=SimpleNamespace(model_name="fake-model")),
        RuntimeConfig(max_prompt_tokens=4096, live_trace_dir=None, recursive_rlm_batch_mode="thread"),
    )
    lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    def fake_recursive_query(
        prompt: str,
        model: str | None = None,
        max_depth: int | None = None,
        consume_budget: bool = True,
    ) -> dict[str, object]:
        nonlocal active_calls, max_active_calls
        del model, max_depth, consume_budget
        with lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        try:
            time.sleep(0.05)
        finally:
            with lock:
                active_calls -= 1
        return {
            "prompt": prompt,
            "model": "fake-model",
            "response": f"response:{prompt}",
            "final_answer": f"answer:{prompt}",
            "kind": "recursive_query",
            "depth": 1,
            "trace": [{"call_id": 1, "steps": []}],
            "execution_time": 0.02,
        }

    runtime._recursive_query = fake_recursive_query  # type: ignore[method-assign]

    payloads = runtime.run_recursive_query_batch(["alpha", "beta", "gamma"], max_workers=3)

    assert [payload["response"] for payload in payloads] == [
        "response:alpha",
        "response:beta",
        "response:gamma",
    ]
    assert max_active_calls > 1


def test_recursive_query_batch_mode_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="recursive_rlm_batch_mode"):
        RuntimeConfig(recursive_rlm_batch_mode="process")


def test_recursive_local_repl_batched_calls_preserve_pending_call_order() -> None:
    def llm_query_batch_fn(
        prompts: list[str],
        model: str | None,
        max_workers: int | None,
    ) -> list[dict[str, object]]:
        del model, max_workers
        return [
            {
                "prompt": prompt,
                "model": "test-model",
                "response": f"plain:{prompt}",
                "kind": "plain_query",
                "depth": 1,
                "execution_time": 0.01,
            }
            for prompt in prompts
        ]

    def rlm_query_batch_fn(
        prompts: list[str],
        model: str | None,
        max_depth: int | None,
        max_workers: int | None,
    ) -> list[dict[str, object]]:
        del model, max_depth, max_workers
        return [
            {
                "prompt": prompt,
                "model": "test-model",
                "response": f"recursive:{prompt}",
                "kind": "recursive_query",
                "depth": 1,
                "trace": [{"call_id": index + 1, "steps": []}],
                "execution_time": 0.02,
            }
            for index, prompt in enumerate(prompts)
        ]

    repl = RecursiveLocalRepl(
        context_payload="context",
        llm_query_fn=lambda prompt, model: {"prompt": prompt, "model": model or "test-model", "response": prompt},
        rlm_query_fn=lambda prompt, model, max_depth: {"prompt": prompt, "model": model or "test-model", "response": prompt},
        llm_query_batch_fn=llm_query_batch_fn,
        rlm_query_batch_fn=rlm_query_batch_fn,
    )

    llm_responses = repl._llm_query_batched(["alpha", "beta"])
    rlm_responses = repl._rlm_query_batched_async(["gamma", "delta"], max_workers=4)

    assert llm_responses == ["plain:alpha", "plain:beta"]
    assert rlm_responses == ["recursive:gamma", "recursive:delta"]
    assert [call.prompt for call in repl._pending_llm_calls] == ["alpha", "beta", "gamma", "delta"]


def test_plain_query_budget_exhaustion_returns_visible_error_without_generation() -> None:
    class _BudgetSession:
        model_name = "fake-model"

        def generate(self, **kwargs):
            del kwargs
            raise AssertionError("budget-exhausted subcall should not generate")

    state = _runtime_state(
        _sync_session=_BudgetSession(),
        subcall_budget_enabled=True,
        subcall_budget_total=1,
        subcall_budget_remaining=0,
        subcall_budget_exhausted=True,
    )
    runtime = RecursiveRuntime(
        state,
        RuntimeConfig(subcall_budget_enabled=True, max_total_subcalls=1, live_trace_dir=None),
    )

    payload = runtime._plain_query("blocked")

    assert payload["budget_error"] is True
    assert payload["response"] == "Error: subcall budget exhausted (0/1 calls remaining)."
    assert state["num_subcalls"] == 0


def test_enabled_budget_decrements_single_llm_and_rlm_subcalls() -> None:
    class _StubSession:
        model_name = "fake-model"

        def generate(self, *, messages, max_tokens: int, temperature: float, top_p: float):
            del messages, max_tokens, temperature, top_p
            return (
                "FINAL(42)",
                TokenPayload(
                    prompt_ids=[11],
                    completion_ids=[21],
                    completion_logprobs=[0.0],
                    completion_mask=[True],
                ),
            )

    state = _runtime_state(
        _sync_session=_StubSession(),
        subcall_budget_enabled=True,
        subcall_budget_total=2,
        subcall_budget_remaining=2,
    )
    runtime = RecursiveRuntime(
        state,
        RuntimeConfig(
            subcall_budget_enabled=True,
            max_total_subcalls=2,
            max_depth=2,
            max_iterations=1,
            turn_max_tokens=8,
            subcall_max_tokens=4,
            live_trace_dir=None,
        ),
    )

    runtime._plain_query("plain")
    runtime._recursive_query("recursive")

    assert state["subcall_budget_remaining"] == 0
    assert state["subcall_budget_exhausted"] is True
    assert state["num_subcalls"] == 2
    assert state["num_llm_subcalls"] == 1
    assert state["num_rlm_subcalls"] == 1


def test_batched_budget_clips_prompts_and_returns_budget_errors() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(
            _sync_session=SimpleNamespace(model_name="fake-model"),
            subcall_budget_enabled=True,
            subcall_budget_total=2,
            subcall_budget_remaining=2,
        ),
        RuntimeConfig(
            subcall_budget_enabled=True,
            max_total_subcalls=2,
            max_batched_subcalls=2,
            max_prompt_tokens=4096,
            live_trace_dir=None,
        ),
    )

    def fake_plain_query(prompt: str, model: str | None = None, consume_budget: bool = True) -> dict[str, object]:
        del model, consume_budget
        return {
            "prompt": prompt,
            "model": "fake-model",
            "response": f"response:{prompt}",
            "kind": "plain_query",
            "depth": 1,
            "execution_time": 0.01,
        }

    runtime._plain_query = fake_plain_query  # type: ignore[method-assign]

    payloads = runtime.run_plain_query_batch(["alpha", "beta", "gamma"])

    assert [payload["response"] for payload in payloads] == [
        "response:alpha",
        "response:beta",
        "Error: subcall budget exhausted (0/2 calls remaining).",
    ]
    assert payloads[2]["budget_error"] is True
    assert runtime.state["subcall_budget_remaining"] == 0


def test_batch_fanout_limit_does_not_mark_budget_exhausted_when_calls_remain() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(
            _sync_session=SimpleNamespace(model_name="fake-model"),
            subcall_budget_enabled=True,
            subcall_budget_total=10,
            subcall_budget_remaining=10,
        ),
        RuntimeConfig(
            subcall_budget_enabled=True,
            max_total_subcalls=10,
            max_batched_subcalls=2,
            max_prompt_tokens=4096,
            live_trace_dir=None,
        ),
    )

    def fake_plain_query(prompt: str, model: str | None = None, consume_budget: bool = True) -> dict[str, object]:
        del model, consume_budget
        return {
            "prompt": prompt,
            "model": "fake-model",
            "response": f"response:{prompt}",
            "kind": "plain_query",
            "depth": 1,
            "execution_time": 0.01,
        }

    runtime._plain_query = fake_plain_query  # type: ignore[method-assign]

    payloads = runtime.run_plain_query_batch(["alpha", "beta", "gamma"])

    assert payloads[2]["response"] == "Error: subcall batch fanout limit reached (2 prompts maximum; 8/10 calls remaining)."
    assert runtime.state["subcall_budget_remaining"] == 8
    assert runtime.state["subcall_budget_exhausted"] is False


def test_budget_feedback_message_is_numeric_only() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(
            _sync_session=SimpleNamespace(model_name="fake-model"),
            subcall_budget_enabled=True,
            subcall_budget_total=40,
            subcall_budget_remaining=17,
        ),
        RuntimeConfig(subcall_budget_enabled=True, max_total_subcalls=40, live_trace_dir=None),
    )

    assert runtime.budget_feedback_message() == {
        "role": "user",
        "content": "Subcall budget remaining: 17/40.",
    }


def test_disabled_budget_feedback_is_absent() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(_sync_session=SimpleNamespace(model_name="fake-model")),
        RuntimeConfig(live_trace_dir=None),
    )

    assert runtime.budget_feedback_message() is None


def test_oversized_llm_subcall_is_blocked_and_visible_in_repl() -> None:
    class _BudgetSession:
        model_name = "fake-model"

        def generate(self, **kwargs):
            del kwargs
            raise AssertionError("oversized subcall should be blocked before model generation")

    state = _runtime_state(_sync_session=_BudgetSession())
    runtime = RecursiveRuntime(
        state,
        RuntimeConfig(
            max_prompt_tokens=100,
            subcall_prompt_limit_ratio=0.85,
            live_trace_dir=None,
        ),
    )
    repl = RecursiveLocalRepl(
        context_payload="context",
        llm_query_fn=runtime._plain_query,
        rlm_query_fn=lambda prompt, model, max_depth: {"response": prompt},
    )

    result = repl.execute_code("answer = llm_query('word ' * 90)")

    assert "Error: LM query failed" in result.stdout
    assert "llm_query prompt is too large" in result.stdout
    assert "approximate 85% subcall prompt budget" in result.stdout
    assert "answer" in result.locals
    assert "llm_query prompt is too large" in result.locals["answer"]
    assert repl._pending_llm_calls == []


def test_oversized_batched_llm_subcalls_raise_one_error() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(_sync_session=SimpleNamespace(model_name="fake-model")),
        RuntimeConfig(
            max_prompt_tokens=100,
            subcall_prompt_limit_ratio=0.85,
            live_trace_dir=None,
        ),
    )

    with pytest.raises(SubcallPromptTooLargeError, match=r"indices 0 .* 2"):
        runtime.run_plain_query_batch(["word " * 90, "small prompt", "word " * 95])


def test_oversized_recursive_subcall_is_blocked_before_child_state_changes() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(
            _sync_session=SimpleNamespace(model_name="fake-model"),
            rlm_call_counter=1,
        ),
        RuntimeConfig(
            max_depth=2,
            max_prompt_tokens=1000,
            subcall_prompt_limit_ratio=0.85,
            live_trace_dir=None,
        ),
    )

    with pytest.raises(SubcallPromptTooLargeError, match="rlm_query prompt is too large"):
        runtime._recursive_query("word " * 900)

    assert runtime.state["used_recursion"] is False
    assert runtime.state["num_subcalls"] == 0
    assert runtime.state["num_rlm_subcalls"] == 0
    assert runtime.state["rlm_trace"] == []


def test_create_repl_rejects_non_local_backend() -> None:
    from rlm_rlvr.repl import create_repl

    with pytest.raises(ValueError, match="only local REPL"):
        create_repl(
            backend="prime",
            backend_kwargs=None,
            context_payload="context",
            llm_query_fn=lambda prompt, model: {"response": prompt},
            rlm_query_fn=lambda prompt, model, max_depth: {"response": prompt},
        )


def test_load_environment_rejects_non_local_backend_before_key_checks() -> None:
    with pytest.raises(ValueError, match="only local REPL"):
        load_environment(repl_backend="docker")


def test_extract_final_answer_accepts_markdown_wrapped_final() -> None:
    from rlm_rlvr.parsing import extract_final_answer

    assert extract_final_answer("`FINAL(Label: negative)`") == "Label: negative"
    assert extract_final_answer("Final answer:\n`FINAL(User: 30836)`") == "User: 30836"
    assert extract_final_answer("```repl\nprint(1)\n```\nFINAL(1849)") == "1849"


def test_recursive_query_updates_depth_and_trace() -> None:
    class _StubSession:
        model_name = "fake-model"

        def generate(self, *, messages, max_tokens: int, temperature: float, top_p: float):
            del messages, max_tokens, temperature, top_p
            return (
                "FINAL(42)",
                TokenPayload(
                    prompt_ids=[11, 12],
                    completion_ids=[21, 22],
                    completion_logprobs=[0.0, 0.0],
                    completion_mask=[True, True],
                ),
            )

    state = _runtime_state(_sync_session=_StubSession())
    runtime = RecursiveRuntime(
        state,
        RuntimeConfig(
            max_depth=2,
            max_iterations=1,
            turn_max_tokens=8,
            subcall_max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            repl_backend="local",
            live_trace_dir=None,
        ),
    )

    result = runtime._recursive_query("solve it")

    assert result["final_answer"] == "42"
    assert result["response"] == "42"
    assert result["depth"] == 1
    assert state["used_recursion"] is True
    assert state["used_llm_subcalls"] is False
    assert state["used_rlm_subcalls"] is True
    assert state["num_subcalls"] == 1
    assert state["num_llm_subcalls"] == 0
    assert state["num_rlm_subcalls"] == 1
    assert state["max_depth_reached"] == 1
    assert len(state["rlm_trace"]) == 1


def test_recursive_query_segments_capture_exact_prompt_context() -> None:
    class _CapturingSession:
        model_name = "fake-model"

        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        def generate(self, *, messages, max_tokens: int, temperature: float, top_p: float):
            del max_tokens, temperature, top_p
            captured = [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]
            self.calls.append(captured)
            return (
                "FINAL(42)",
                TokenPayload(
                    prompt_ids=[11, 12],
                    completion_ids=[21, 22],
                    completion_logprobs=[0.0, 0.0],
                    completion_mask=[True, True],
                ),
            )

    session = _CapturingSession()
    state = _runtime_state(_sync_session=session)
    runtime = RecursiveRuntime(
        state,
        RuntimeConfig(
            max_depth=2,
            max_iterations=1,
            turn_max_tokens=8,
            subcall_max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            repl_backend="local",
            live_trace_dir=None,
        ),
    )

    result = runtime._recursive_query("solve it")

    assert result["call_id"] == 1
    segment = state["rlm_segments"][0]
    expected_messages = session.calls[0]
    expected_chars = sum(len(message["role"]) + len(message["content"]) for message in expected_messages)
    assert segment["call_id"] == 1
    assert segment["parent_call_id"] == 0
    assert segment["turn_index"] == 0
    assert segment["train_scope"] == "recursive_turn"
    assert segment["is_trainable_rlm_turn"] is True
    assert segment["prompt_message_count"] == len(expected_messages)
    assert segment["prompt_char_count"] == expected_chars
    assert isinstance(segment["prompt_fingerprint"], str)
    assert len(segment["prompt_fingerprint"]) == 40


def test_plain_query_counts_as_depth_one_llm_subcall() -> None:
    class _StubSession:
        model_name = "fake-model"

        def generate(self, *, messages, max_tokens: int, temperature: float, top_p: float):
            del messages, max_tokens, temperature, top_p
            return (
                "plain answer",
                TokenPayload(
                    prompt_ids=[11, 12],
                    completion_ids=[21, 22],
                    completion_logprobs=[0.0, 0.0],
                    completion_mask=[True, True],
                ),
            )

    state = _runtime_state(_sync_session=_StubSession())
    runtime = RecursiveRuntime(
        state,
        RuntimeConfig(
            subcall_max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            capture_prompt_messages=True,
            live_trace_dir=None,
        ),
    )

    result = runtime._plain_query("solve directly")

    assert result["response"] == "plain answer"
    assert result["kind"] == "plain_query"
    assert state["used_recursion"] is True
    assert state["used_llm_subcalls"] is True
    assert state["used_rlm_subcalls"] is False
    assert state["num_subcalls"] == 1
    assert state["num_llm_subcalls"] == 1
    assert state["num_rlm_subcalls"] == 0
    assert state["max_depth_reached"] == 1
    assert len(state["rlm_segments"]) == 1


def test_plain_query_uses_vertex_session_and_recursive_query_uses_local_session() -> None:
    class _LabeledSession:
        def __init__(self, model_name: str, response: str) -> None:
            self.model_name = model_name
            self.response = response
            self.calls = 0

        def generate(self, *, messages, max_tokens: int, temperature: float, top_p: float):
            del messages, max_tokens, temperature, top_p
            self.calls += 1
            return (
                self.response,
                TokenPayload(
                    prompt_ids=[11, 12],
                    completion_ids=[21, 22],
                    completion_logprobs=[0.0, 0.0],
                    completion_mask=[True, True],
                    prompt_token_count=7,
                    completion_token_count=5,
                ),
            )

    local_session = _LabeledSession("local-training-model", "FINAL(local)")
    vertex_session = _LabeledSession("gemini-3.1-flash-lite", "plain answer")
    state = _runtime_state(
        _sync_session=local_session,
        _plain_llm_session=vertex_session,
    )
    runtime = RecursiveRuntime(
        state,
        RuntimeConfig(
            max_depth=2,
            max_iterations=1,
            turn_max_tokens=8,
            subcall_max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            live_trace_dir=None,
        ),
    )

    plain_result = runtime._plain_query("solve directly")
    recursive_result = runtime._recursive_query("solve recursively")

    assert plain_result["model"] == "gemini-3.1-flash-lite"
    assert recursive_result["model"] == "local-training-model"
    assert vertex_session.calls == 1
    assert local_session.calls == 1
    plain_segment, recursive_segment = state["rlm_segments"]
    assert plain_segment["train_scope"] == "llm_subcall"
    assert plain_segment["is_trainable_rlm_turn"] is False
    assert plain_segment["prompt_token_count"] == 7
    assert plain_segment["completion_token_count"] == 5
    assert recursive_segment["train_scope"] == "recursive_turn"
    assert recursive_segment["is_trainable_rlm_turn"] is True


def test_plain_query_segments_are_non_trainable_llm_subcalls() -> None:
    class _StubSession:
        model_name = "fake-model"

        def generate(self, *, messages, max_tokens: int, temperature: float, top_p: float):
            del messages, max_tokens, temperature, top_p
            return (
                "plain answer",
                TokenPayload(
                    prompt_ids=[11, 12],
                    completion_ids=[21, 22],
                    completion_logprobs=[0.0, 0.0],
                    completion_mask=[True, True],
                ),
            )

    state = _runtime_state(_sync_session=_StubSession())
    runtime = RecursiveRuntime(
        state,
        RuntimeConfig(
            subcall_max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            capture_prompt_messages=True,
            live_trace_dir=None,
        ),
    )

    runtime._plain_query("solve directly")

    segment = state["rlm_segments"][0]
    assert segment["call_id"] == 0
    assert segment["parent_call_id"] is None
    assert segment["train_scope"] == "llm_subcall"
    assert segment["is_trainable_rlm_turn"] is False
    assert segment["response_source"] == "llm_subcall"
    assert segment["turn_index"] == -1
    assert segment["prompt_messages"] == [{"role": "user", "content": "solve directly"}]
    assert segment["request"] == {
        "model": "fake-model",
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def test_live_trace_compacts_segments_without_token_arrays(tmp_path) -> None:
    from rlm_rlvr.live_trace import write_live_trace

    root_trace = make_call_trace(call_id=0, depth=0, prompt="Question?")
    state = {
        "prompt_variant": DEFAULT_PROMPT_VARIANT,
        "live_trace_dir": str(tmp_path),
        "info": {"source_id": "source/with spaces", "question": "Question?"},
        "rlm_trace": [root_trace],
        "rlm_segments": [
            {
                "order": 0,
                "call_id": 0,
                "parent_call_id": None,
                "depth": 0,
                "turn_index": 0,
                "kind": "root_turn",
                "train_scope": "root_turn",
                "is_trainable_rlm_turn": True,
                "response_source": "root",
                "prompt_fingerprint": "abc123",
                "prompt_message_count": 3,
                "prompt_char_count": 42,
                "prompt_ids": [1, 2, 3],
                "completion_ids": [4, 5],
                "prompt_token_count": 17,
                "completion_token_count": 9,
                "completion_logprobs": [-0.1, -0.2],
                "completion_mask": [True, True],
                "temperature": 0.7,
                "response_text": "```repl\nprint(1)\n```",
            }
        ],
        "used_repl": True,
        "used_recursion": True,
        "used_llm_subcalls": True,
        "used_rlm_subcalls": False,
        "max_depth_reached": 1,
        "num_subcalls": 1,
        "num_llm_subcalls": 1,
        "num_rlm_subcalls": 0,
        "total_model_tokens": 2.0,
        "total_env_tokens": 0.0,
        "final_answer": None,
    }

    write_live_trace(state, event="test")

    path = tmp_path / DEFAULT_PROMPT_VARIANT / "source_with_spaces.json"
    live_trace = json.loads(path.read_text())
    segment = live_trace["segments"][0]
    assert live_trace["status"]["used_llm_subcalls"] is True
    assert live_trace["status"]["num_llm_subcalls"] == 1
    assert segment["call_id"] == 0
    assert segment["parent_call_id"] is None
    assert segment["turn_index"] == 0
    assert segment["train_scope"] == "root_turn"
    assert segment["is_trainable_rlm_turn"] is True
    assert segment["prompt_tokens"] == 17
    assert segment["completion_tokens"] == 9
    assert segment["response_text"] == "```repl\nprint(1)\n```"
    assert "prompt_ids" not in segment
    assert "completion_ids" not in segment


def test_live_trace_suffix_allows_parallel_rollouts_for_same_source(tmp_path) -> None:
    from rlm_rlvr.live_trace import assign_live_trace_suffix, write_live_trace

    def make_state() -> dict:
        root_trace = make_call_trace(call_id=0, depth=0, prompt="Question?")
        return {
            "prompt_variant": "default",
            "live_trace_dir": str(tmp_path),
            "info": {"source_id": "same-source", "question": "Question?"},
            "rlm_trace": [root_trace],
            "rlm_segments": [],
        }

    state_a = make_state()
    state_b = make_state()
    assign_live_trace_suffix(state_a)
    assign_live_trace_suffix(state_b)
    write_live_trace(state_a, event="a")
    write_live_trace(state_b, event="b")

    paths = sorted((tmp_path / "default").glob("same-source-*.json"))
    assert len(paths) == 2
    assert json.loads(paths[0].read_text())["event"] in {"a", "b"}
    assert json.loads(paths[1].read_text())["event"] in {"a", "b"}
