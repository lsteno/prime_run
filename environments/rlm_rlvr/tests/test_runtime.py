from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI

from rlm_rlvr.env import RLMRLVREnv, load_environment
from rlm_rlvr.prompt_variants import DEFAULT_PROMPT_VARIANT
from rlm_rlvr.repl import RecursiveLocalRepl
from rlm_rlvr.runtime import RecursiveRuntime, RuntimeConfig, SubcallPromptTooLargeError, SyncInferenceSession, TokenPayload
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
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def post(self, path: str, *, body, cast_to):
        del cast_to
        self.bodies.append(body)
        assert path == "chat/completions"
        assert body["logprobs"] is True
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


def test_recursive_query_batch_runs_in_parallel() -> None:
    runtime = RecursiveRuntime(
        _runtime_state(_sync_session=SimpleNamespace(model_name="fake-model")),
        RuntimeConfig(max_prompt_tokens=4096, live_trace_dir=None),
    )

    def fake_recursive_query(
        prompt: str,
        model: str | None = None,
        max_depth: int | None = None,
        consume_budget: bool = True,
    ) -> dict[str, object]:
        del model, max_depth, consume_budget
        time.sleep(0.05)
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

    start = time.perf_counter()
    payloads = runtime.run_recursive_query_batch(["alpha", "beta", "gamma"], max_workers=3)
    elapsed = time.perf_counter() - start

    assert [payload["response"] for payload in payloads] == [
        "response:alpha",
        "response:beta",
        "response:gamma",
    ]
    assert elapsed < 0.13


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
    assert segment["prompt_tokens"] == 3
    assert segment["completion_tokens"] == 2
    assert segment["response_text"] == "```repl\nprint(1)\n```"
    assert "prompt_ids" not in segment
    assert "completion_ids" not in segment
