from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI

from rlm_rlvr.env import RLMRLVREnv, load_environment
from rlm_rlvr.repl import RecursiveLocalRepl
from rlm_rlvr.runtime import RecursiveRuntime, RuntimeConfig, SyncInferenceSession, TokenPayload


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool, return_dict: bool):
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_dict is True
        return {"input_ids": [101, 102, 103]}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(text), len(text) + 1] if text else []

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(f"tok{token_id}" for token_id in token_ids)


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

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join("x" * max(token_id, 1) for token_id in token_ids)


class _FakeClient:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def post(self, path: str, *, body, cast_to):
        del cast_to
        self.bodies.append(body)
        assert path == "chat/completions"
        assert body["logprobs"] is True
        assert body["extra_body"]["return_token_ids"] is True
        choice = SimpleNamespace(
            message=SimpleNamespace(content="ok"),
            token_ids=None,
            logprobs=None,
        )
        return SimpleNamespace(choices=[choice], prompt_token_ids=None)


class _FakeSyncInferenceSession:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def count_text_tokens(self, text: str) -> int:
        return len(text.split())


def test_generate_falls_back_when_token_metadata_is_missing() -> None:
    session = object.__new__(SyncInferenceSession)
    session.model_name = "fake-model"
    session.client = _FakeClient()
    session.tokenizer = _FakeTokenizer()
    session.max_prompt_tokens = None

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


def test_setup_state_rebuilds_root_prompt_with_real_context_metadata(monkeypatch) -> None:
    import rlm_rlvr.env as env_module

    monkeypatch.setattr(env_module, "SyncInferenceSession", _FakeSyncInferenceSession)
    environment = object.__new__(RLMRLVREnv)
    environment.runtime_config = RuntimeConfig(prompt_variant="balanced_v1")
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


def test_generate_trims_prompt_history_to_fit_budget() -> None:
    client = _FakeClient()
    session = object.__new__(SyncInferenceSession)
    session.model_name = "fake-model"
    session.client = client
    session.tokenizer = _BudgetTokenizer()
    session.max_prompt_tokens = 10

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
    request_body = client.bodies[0]
    prompt_ids = session.render_prompt_ids(request_body["messages"], add_generation_prompt=True)
    assert len(prompt_ids) <= 10
    assert len(request_body["messages"]) < len(messages)


def test_recursive_local_repl_routes_llm_and_rlm_queries() -> None:
    repl = RecursiveLocalRepl(
        context_payload="context",
        llm_query_fn=lambda prompt, model: {
            "prompt": prompt,
            "model": model or "test-model",
            "response": "plain-response",
            "kind": "plain_query",
            "depth": 0,
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


def test_recursive_local_repl_async_batch_runs_serially_in_order() -> None:
    calls: list[str] = []

    def rlm_query_fn(prompt: str, model: str | None, max_depth: int | None):
        del model, max_depth
        calls.append(prompt)
        return {
            "prompt": prompt,
            "model": "test-model",
            "response": f"response:{prompt}",
            "kind": "recursive_query",
            "depth": 1,
            "trace": [{"call_id": len(calls), "steps": []}],
            "execution_time": 0.02,
        }

    repl = RecursiveLocalRepl(
        context_payload="context",
        llm_query_fn=lambda prompt, model: {"prompt": prompt, "model": model or "test-model", "response": prompt},
        rlm_query_fn=rlm_query_fn,
    )

    responses = repl._rlm_query_batched_async(["alpha", "beta", "gamma"], max_workers=4)

    assert responses == ["response:alpha", "response:beta", "response:gamma"]
    assert calls == ["alpha", "beta", "gamma"]
    assert [call.prompt for call in repl._pending_llm_calls] == ["alpha", "beta", "gamma"]


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

    state = {
        "_sync_session": _StubSession(),
        "rlm_segment_counter": 0,
        "rlm_call_counter": 0,
        "rlm_segments": [],
        "rlm_trace": [],
        "total_model_tokens": 0.0,
        "max_depth_reached": 0,
        "current_call_depth": 0,
        "current_branch_max_depth": 2,
        "used_recursion": False,
        "num_subcalls": 0,
        "sampling_temperature": 0.0,
    }
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
        ),
    )

    result = runtime._recursive_query("solve it")

    assert result["final_answer"] == "42"
    assert result["response"] == "42"
    assert result["depth"] == 1
    assert state["used_recursion"] is True
    assert state["num_subcalls"] == 1
    assert state["max_depth_reached"] == 1
    assert len(state["rlm_trace"]) == 1
