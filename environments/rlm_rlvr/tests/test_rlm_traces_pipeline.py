from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.rlm_traces import export_sft
from pipelines.rlm_traces import curate_sft_traces
from pipelines.rlm_traces import run as trace_run
from pipelines.rlm_traces import run_missing_with_retry
from scripts.rlm_sft import upload_latest_weights


def test_glm5_endpoint_resolution_uses_prime_api(monkeypatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "prime-test-key")

    endpoint = trace_run.resolve_endpoint(
        {"endpoint_id": "glm-5", "extra_body": {"reasoning": {"enabled": False}}},
        endpoints_path=REPO_ROOT / "configs" / "endpoints.toml",
    )

    assert endpoint.model == "z-ai/glm-5"
    assert endpoint.url == "https://api.pinference.ai/api/v1"
    assert endpoint.api_key_env == "PRIME_API_KEY"
    assert endpoint.api_key == "prime-test-key"
    assert endpoint.endpoint_type == "openai_chat_completions"
    assert endpoint.extra_body == {"reasoning": {"enabled": False}}


def test_trace_config_serialization_redacts_api_keys() -> None:
    endpoint = trace_run.EndpointConfig(
        endpoint_id=None,
        model="openai/gpt-5.4",
        url="https://api.pinference.ai/api/v1",
        api_key_env="PRIME_API_KEY",
        api_key="prime-secret",
    )
    judge = trace_run.JudgeConfig(
        provider="openai_compatible",
        model="judge-model",
        endpoint=endpoint,
    )

    assert trace_run.redacted_dataclass_dict(endpoint)["api_key"] == "<redacted>"
    payload = trace_run.redacted_dataclass_dict(judge)
    assert payload["endpoint"]["api_key"] == "<redacted>"
    assert "prime-secret" not in repr(payload)


def test_external_teacher_session_does_not_request_logprobs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeSession:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.model_name = kwargs["model_name"]

    class _FakeRuntime:
        def __init__(self, state, config) -> None:
            self.state = state
            self.config = config

        def _plain_query(self, *args, **kwargs):
            return {}

        def _recursive_query(self, *args, **kwargs):
            return {}

        def run_plain_query_batch(self, prompts, **kwargs):
            return []

        def run_recursive_query_batch(self, prompts, **kwargs):
            return []

    fake_modules = {
        "SyncInferenceSession": _FakeSession,
        "RecursiveRuntime": _FakeRuntime,
        "create_repl": lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    }
    monkeypatch.setattr(trace_run, "load_rlm_modules", lambda: fake_modules)

    runtime_config = SimpleNamespace(
        max_depth=0,
        temperature=0.7,
        tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
        max_prompt_tokens=65536,
        repl_backend="local",
        repl_backend_kwargs=None,
        subcall_budget_enabled=True,
        max_total_subcalls=60,
        max_batched_subcalls=60,
    )
    endpoint = trace_run.EndpointConfig(
        endpoint_id="glm-5",
        model="z-ai/glm-5",
        url="https://api.pinference.ai/api/v1",
        api_key_env="PRIME_API_KEY",
        api_key="prime-test-key",
    )

    trace_run.init_state(endpoint=endpoint, runtime_config=runtime_config, disable_recursive_subcalls=True)

    assert captured["request_logprobs"] is False
    assert captured["retry_transient_errors"] is True
    assert captured["openai_extra_body"] is None
    assert captured["enable_token_accounting"] is False
    assert captured["tokenizer_name"] == "Qwen/Qwen3-4B-Instruct-2507"


def test_trace_pipeline_creates_separate_vertex_plain_subcall_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeSession:
        def __init__(self, **kwargs) -> None:
            self.model_name = kwargs["model_name"]

    class _FakeVertexSession:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.model_name = kwargs["model_name"]

    class _FakeRuntime:
        def __init__(self, state, config) -> None:
            self.state = state
            self.config = config

        def _plain_query(self, *args, **kwargs):
            return {}

        def _recursive_query(self, *args, **kwargs):
            return {}

        def run_plain_query_batch(self, prompts, **kwargs):
            return []

        def run_recursive_query_batch(self, prompts, **kwargs):
            return []

    fake_modules = {
        "SyncInferenceSession": _FakeSession,
        "VertexGeminiSession": _FakeVertexSession,
        "RecursiveRuntime": _FakeRuntime,
        "create_repl": lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    }
    monkeypatch.setattr(trace_run, "load_rlm_modules", lambda: fake_modules)

    runtime_config = SimpleNamespace(
        max_depth=0,
        temperature=0.7,
        tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
        max_prompt_tokens=200000,
        repl_backend="local",
        repl_backend_kwargs=None,
        subcall_budget_enabled=True,
        max_total_subcalls=60,
        max_batched_subcalls=60,
    )
    endpoint = trace_run.EndpointConfig(
        endpoint_id=None,
        model="openai/gpt-5.4",
        url="https://api.pinference.ai/api/v1",
        api_key_env="PRIME_API_KEY",
        api_key="prime-test-key",
    )
    subcall_config = trace_run.PlainSubcallConfig(
        provider="vertex",
        model="gemini-3-flash-preview",
        vertex_project="test-project",
        vertex_location="global",
        thinking_level="medium",
    )

    state, *_ = trace_run.init_state(
        endpoint=endpoint,
        runtime_config=runtime_config,
        plain_subcall_config=subcall_config,
        disable_recursive_subcalls=True,
    )

    assert captured == {
        "model_name": "gemini-3-flash-preview",
        "project": "test-project",
        "location": "global",
        "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
        "max_prompt_tokens": 200000,
        "thinking_level": "medium",
        "empty_response_max_attempts": 1,
        "empty_response_base_retry_seconds": 1.0,
        "empty_response_max_retry_seconds": 30.0,
    }
    assert state["_plain_llm_session"].model_name == "gemini-3-flash-preview"
    assert state["_sync_session"].model_name == "openai/gpt-5.4"


def test_trace_pipeline_reuses_root_policy_when_plain_subcall_config_is_absent(monkeypatch) -> None:
    vertex_calls: list[dict[str, object]] = []

    class _FakeSession:
        def __init__(self, **kwargs) -> None:
            self.model_name = kwargs["model_name"]

    class _FakeVertexSession:
        def __init__(self, **kwargs) -> None:
            vertex_calls.append(kwargs)
            self.model_name = kwargs["model_name"]

    class _FakeRuntime:
        def __init__(self, state, config) -> None:
            self.state = state
            self.config = config

        def _plain_query(self, *args, **kwargs):
            return {}

        def _recursive_query(self, *args, **kwargs):
            return {}

        def run_plain_query_batch(self, prompts, **kwargs):
            return []

        def run_recursive_query_batch(self, prompts, **kwargs):
            return []

    fake_modules = {
        "SyncInferenceSession": _FakeSession,
        "VertexGeminiSession": _FakeVertexSession,
        "RecursiveRuntime": _FakeRuntime,
        "create_repl": lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    }
    monkeypatch.setattr(trace_run, "load_rlm_modules", lambda: fake_modules)

    runtime_config = SimpleNamespace(
        max_depth=0,
        temperature=0.7,
        tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
        max_prompt_tokens=200000,
        repl_backend="local",
        repl_backend_kwargs=None,
        subcall_budget_enabled=True,
        max_total_subcalls=60,
        max_batched_subcalls=60,
    )
    endpoint = trace_run.EndpointConfig(
        endpoint_id=None,
        model="openai/gpt-5.4",
        url="https://api.pinference.ai/api/v1",
        api_key_env="PRIME_API_KEY",
        api_key="prime-test-key",
    )

    state, *_ = trace_run.init_state(
        endpoint=endpoint,
        runtime_config=runtime_config,
        plain_subcall_config=None,
        disable_recursive_subcalls=True,
    )

    assert vertex_calls == []
    assert state["_plain_llm_session"] is state["_sync_session"]
    assert state["_plain_llm_session"].model_name == "openai/gpt-5.4"
    assert state["llm_subcall_session_source"] == "root_policy"


def test_load_examples_can_pin_source_ids(monkeypatch) -> None:
    rows = [
        {"id": "a", "question": "qa", "answer": "aa", "context": ""},
        {"id": "b", "question": "qb", "answer": "ab", "context": ""},
        {"id": "c", "question": "qc", "answer": "ac", "context": ""},
    ]

    class _FakeDataset:
        def __init__(self, items) -> None:
            self.items = list(items)

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            return self.items[index]

        def select(self, indices):
            return _FakeDataset([self.items[index] for index in indices])

        def __iter__(self):
            return iter(self.items)

    monkeypatch.setattr(trace_run, "load_dataset_fn", lambda: lambda *args, **kwargs: _FakeDataset(rows))
    modules = {"parse_answer_candidates": lambda value: [str(value)]}
    monkeypatch.setattr(trace_run, "load_rlm_modules", lambda: modules)

    examples = trace_run.load_examples(
        {
            "dataset_id": "test",
            "split": "sft_traces",
            "source_ids": ["c", "a"],
            "seed": 0,
            "max_examples": 1,
        }
    )

    assert [example.source_id for example in examples] == ["c", "a"]


def test_trace_generation_context_metadata_message_omits_context_and_budget() -> None:
    message = trace_run.build_context_metadata_message("alpha beta gamma")

    assert message["role"] == "user"
    assert "Your context is a str with 16 total characters" in message["content"]
    assert "char lengths" in message["content"]
    assert "model turns remaining" not in message["content"]
    assert "alpha beta gamma" not in message["content"]
    assert "context window" not in message["content"]
    assert "prompt tokens" not in message["content"]


def test_vertex_judge_retries_and_parses_binary(monkeypatch) -> None:
    calls = {"count": 0}

    class _RateLimitError(Exception):
        status_code = 429

    class _FakeThinkingConfig:
        def __init__(self, *, thinking_level: str) -> None:
            self.thinking_level = thinking_level

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeModels:
        def generate_content(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise _RateLimitError("rate limit")
            return SimpleNamespace(text="1")

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.models = _FakeModels()

    monkeypatch.setattr(
        trace_run,
        "load_google_genai",
        lambda: (
            SimpleNamespace(Client=_FakeClient),
            SimpleNamespace(ThinkingConfig=_FakeThinkingConfig, GenerateContentConfig=_FakeGenerateContentConfig),
        ),
    )
    monkeypatch.setattr(trace_run, "_sleep_before_retry", lambda **kwargs: None)

    score, raw = trace_run.judge_answer(
        judge_config=trace_run.JudgeConfig(
            provider="vertex",
            model="gemini-3-flash-preview",
            vertex_project="test-project",
            vertex_location="global",
            thinking_level="medium",
        ),
        question="Question?",
        expected_answers=["answer"],
        predicted_answer="answer",
    )

    assert score == 1.0
    assert raw == "1"
    assert calls["count"] == 2


def test_sft_export_filters_plain_subcalls_and_sanitizes_history() -> None:
    records = [
        {
            "source_id": "source-a",
            "example_id": 0,
            "exact_match": False,
            "judge_score": 1.0,
            "final_answer": "42",
            "segments": [
                {
                    "order": 0,
                    "kind": "root_turn",
                    "train_scope": "root_turn",
                    "is_trainable_rlm_turn": True,
                    "prompt_messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "assistant", "content": "<think>hidden</think> visible"},
                        {"role": "user", "content": "question"},
                    ],
                    "response_text": "FINAL(42)",
                    "prompt_token_count": 10,
                    "completion_token_count": 3,
                },
                {
                    "order": 1,
                    "kind": "plain_query",
                    "train_scope": "llm_subcall",
                    "is_trainable_rlm_turn": False,
                    "prompt_messages": [{"role": "user", "content": "sub"}],
                    "response_text": "trace only",
                },
            ],
        }
    ]

    rows = export_sft.extract_sft_rows(records)

    assert len(rows) == 1
    assert rows[0]["prompt"][1]["content"] == "visible"
    assert rows[0]["completion"] == [{"role": "assistant", "content": "FINAL(42)"}]


def _strict_conversation_record() -> dict:
    return {
        "source_id": "source-conv",
        "example_id": 1,
        "exact_match": True,
        "judge_score": None,
        "final_answer": "42",
        "num_llm_subcalls": 1,
        "total_prompt_tokens": 100,
        "total_completion_tokens": 20,
        "total_rollout_tokens": 120,
        "curation": {
            "version": "curated-v2",
            "status": "strict_sft",
            "tags": ["strict_sft"],
            "source_run_id": "test-run",
            "source_input_path": "records.curated.jsonl",
            "root_model": "openai/gpt-5.4",
            "manual_decision_id": "source-conv:record",
        },
        "segments": [
            {
                "order": 0,
                "kind": "root_turn",
                "train_scope": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "metadata"},
                    {"role": "user", "content": "question"},
                ],
                "response_text": "```repl\nprint('inspect')\n```\n\n{\"answer\": 42}",
            },
            {
                "order": 1,
                "kind": "plain_query",
                "train_scope": "llm_subcall",
                "is_trainable_rlm_turn": False,
                "prompt_messages": [{"role": "user", "content": "subcall prompt"}],
                "response_text": "subcall evidence",
            },
            {
                "order": 2,
                "kind": "root_turn",
                "train_scope": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "metadata"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "```repl\nprint('inspect')\n```\n\n{\"answer\": 42}"},
                    {"role": "user", "content": "Code executed:\n```python\nprint('inspect')\n```\n\nREPL output:\ninspect"},
                    {"role": "user", "content": "Continue using the REPL."},
                ],
                "response_text": "FINAL(42)",
            },
        ],
    }


def test_conversation_export_uses_one_row_per_strict_trace_and_masks_feedback() -> None:
    record = _strict_conversation_record()

    rows = export_sft.extract_conversation_rows([record])

    assert len(rows) == 1
    row = rows[0]
    assert row["row_id"] == "source-conv:conversation"
    assert [message["role"] for message in row["prompt"]] == ["system", "user", "user"]
    assert [message["role"] for message in row["completion"]] == ["assistant", "user", "user", "assistant"]
    assert row["completion"][1]["content"].startswith("Code executed:")
    assert row["completion"][2]["content"] == "Continue using the REPL."
    assert row["completion"][0]["content"] == "```repl\nprint('inspect')\n```"
    assert row["completion"][-1]["content"] == "FINAL(42)"
    assert row["non_empty_llm_subcall_count"] == 1
    assert row["scrubbed_non_final_repl_turns"] == 1


def test_conversation_export_excludes_audit_only_and_subcall_free_records() -> None:
    strict = _strict_conversation_record()
    audit_only = copy.deepcopy(strict)
    audit_only["source_id"] = "audit"
    audit_only["curation"]["status"] = "audit_only"
    no_subcall = copy.deepcopy(strict)
    no_subcall["source_id"] = "no-subcall"
    no_subcall["segments"][1]["response_text"] = ""

    rows = export_sft.extract_conversation_rows([strict, audit_only, no_subcall])

    assert [row["source_id"] for row in rows] == ["source-conv"]


def test_conversation_export_preserves_curated_synthetic_final_turn() -> None:
    record = _strict_conversation_record()
    record["segments"][2]["_curation_sft_rows"] = [
        {
            "row_role": "analysis_before_final",
            "prompt_messages": record["segments"][2]["prompt_messages"],
            "response_text": "```repl\nanswer = '42'\n```",
        },
        {
            "row_role": "synthetic_final_after_feedback",
            "prompt_messages": [
                *record["segments"][2]["prompt_messages"],
                {"role": "assistant", "content": "```repl\nanswer = '42'\n```"},
                {"role": "user", "content": "REPL variables: ['answer']"},
            ],
            "response_text": '```repl\nFINAL_VAR("answer")\n```',
        },
    ]

    rows = export_sft.extract_conversation_rows([record])

    contents = [message["content"] for message in rows[0]["completion"] if message["role"] == "assistant"]
    assert "```repl\nanswer = '42'\n```" in contents
    assert '```repl\nFINAL_VAR("answer")\n```' in contents


def test_per_root_turn_export_uses_one_row_per_root_decision_with_full_history() -> None:
    record = _strict_conversation_record()

    rows = export_sft.extract_per_root_turn_rows([record])

    assert len(rows) == 2
    assert rows[0]["row_id"] == "source-conv:root_turn:0000"
    assert rows[0]["prompt"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "metadata"},
        {"role": "user", "content": "question"},
    ]
    assert rows[0]["completion"] == [{"role": "assistant", "content": "```repl\nprint('inspect')\n```"}]
    assert rows[1]["row_id"] == "source-conv:root_turn:0002"
    assert [message["role"] for message in rows[1]["prompt"]] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert rows[1]["prompt"][3]["content"] == "visible" or "print('inspect')" in rows[1]["prompt"][3]["content"]
    assert rows[1]["prompt"][4]["content"].startswith("Code executed:")
    assert rows[1]["completion"] == [{"role": "assistant", "content": "FINAL(42)"}]
    assert rows[1]["non_empty_llm_subcall_count"] == 1


def test_per_root_turn_export_preserves_curated_synthetic_final_turns() -> None:
    record = _strict_conversation_record()
    record["segments"][2]["_curation_sft_rows"] = [
        {
            "row_role": "analysis_before_final",
            "prompt_messages": record["segments"][2]["prompt_messages"],
            "response_text": "```repl\nanswer = '42'\n```",
        },
        {
            "row_role": "synthetic_final_after_feedback",
            "prompt_messages": [
                *record["segments"][2]["prompt_messages"],
                {"role": "assistant", "content": "```repl\nanswer = '42'\n```"},
                {"role": "user", "content": "REPL variables: ['answer']"},
            ],
            "response_text": '```repl\nFINAL_VAR("answer")\n```',
        },
    ]

    rows = export_sft.extract_per_root_turn_rows([record])

    assert [row["row_role"] for row in rows] == [
        "original_or_repaired",
        "analysis_before_final",
        "synthetic_final_after_feedback",
    ]
    assert rows[-1]["prompt"][-1] == {"role": "user", "content": "REPL variables: ['answer']"}
    assert rows[-1]["completion"] == [{"role": "assistant", "content": '```repl\nFINAL_VAR("answer")\n```'}]


def test_sft_export_can_attach_chat_template_kwargs() -> None:
    record = _strict_conversation_record()

    rows = export_sft.extract_per_root_turn_rows([record])
    rows = export_sft.add_chat_template_kwargs(rows, {"enable_thinking": False})

    assert rows
    assert all(row["chat_template_kwargs"] == {"enable_thinking": False} for row in rows)


def _curation_record(*, response_text: str = "FINAL(42)", plain_response: str = "evidence", num_subcalls: int = 1):
    return {
        "source_id": "source-a",
        "example_id": 0,
        "exact_match": True,
        "judge_score": None,
        "answer": "42",
        "final_answer": "42",
        "num_llm_subcalls": num_subcalls,
        "root_steps": [{"assistant": response_text, "feedback_messages": [{"role": "user", "content": "observed"}]}],
        "segments": [
            {
                "order": 0,
                "kind": "plain_query",
                "train_scope": "llm_subcall",
                "is_trainable_rlm_turn": False,
                "prompt_fingerprint": "plain-a",
                "response_text": plain_response,
            },
            {
                "order": 1,
                "kind": "root_turn",
                "train_scope": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_messages": [{"role": "user", "content": "question"}],
                "response_text": response_text,
                "prompt_token_count": 10,
                "completion_token_count": 3,
            },
        ],
    }


def test_curator_repairs_missing_final_wrapper_and_exports_strict_row() -> None:
    record = _curation_record(response_text="42")

    curated, manifest, patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["curation"]["status"] == "strict_sft"
    assert "missing_final_wrapper_repaired" in curated[0]["curation"]["tags"]
    assert patches[0]["repair_type"] == "missing_final_wrapper"
    assert rows[0]["completion"] == [{"role": "assistant", "content": "FINAL(42)"}]
    assert manifest[0]["status"] == "strict_sft"


def test_curator_tags_empty_subcalls_without_modifying_them_when_recovered() -> None:
    record = _curation_record(response_text="```repl\nprint('verify')\n```", plain_response="")
    record["segments"].append(
        {
            "order": 1,
            "kind": "plain_query",
            "train_scope": "llm_subcall",
            "is_trainable_rlm_turn": False,
            "prompt_fingerprint": "plain-b",
            "response_text": "later evidence",
        }
    )
    record["segments"].append(
        {
            "order": 2,
            "kind": "root_turn",
            "train_scope": "root_turn",
            "is_trainable_rlm_turn": True,
            "prompt_messages": [{"role": "user", "content": "question after evidence"}],
            "response_text": "FINAL(42)",
        }
    )

    curated, *_ = curate_sft_traces.curate_records([record])

    assert curated[0]["segments"][0]["response_text"] == ""
    assert curated[0]["curation"]["status"] == "strict_sft"
    assert "empty_subcall" in curated[0]["curation"]["tags"]
    assert "empty_subcall_recovered" in curated[0]["curation"]["tags"]


def test_curator_excludes_unrecovered_empty_subcalls() -> None:
    record = _curation_record(response_text="The empty response proves it is 42.\nFINAL(42)", plain_response="")

    curated, _manifest, _patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["segments"][0]["response_text"] == ""
    assert curated[0]["curation"]["status"] == "audit_only"
    assert "empty_subcall_unrecovered" in curated[0]["curation"]["tags"]
    assert rows == []


def test_curator_keeps_high_subcall_traces_eligible() -> None:
    record = _curation_record(num_subcalls=25)

    curated, _manifest, _patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["curation"]["status"] == "strict_sft"
    assert "overdelegated" in curated[0]["curation"]["tags"]
    assert "high_subcall_count" in curated[0]["curation"]["tags"]
    assert len(rows) == 1


def test_curator_repairs_final_inside_repl() -> None:
    record = _curation_record(response_text="```repl\n# done\nFINAL(42)\n```")

    curated, _manifest, patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["curation"]["status"] == "strict_sft"
    assert patches[0]["repair_type"] == "literal_final_in_repl"
    assert rows[0]["completion"] == [{"role": "assistant", "content": "FINAL(42)"}]


def test_curator_preserves_quoted_literal_final_inside_repl_for_reviewed_sources() -> None:
    record = _curation_record(response_text='```repl\nFINAL("Jenna Ortega")\n```')
    record["source_id"] = "frames-0230"
    record["answer"] = "jenna ortega"
    record["final_answer"] = '"Jenna Ortega"'

    curated, _manifest, patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["curation"]["status"] == "strict_sft"
    assert patches == []
    assert "literal_final_in_repl_preserved_quoted" in curated[0]["curation"]["tags"]
    assert rows[0]["completion"] == [{"role": "assistant", "content": '```repl\nFINAL("Jenna Ortega")\n```'}]


def test_curator_accepts_final_var_inside_repl() -> None:
    record = _curation_record(response_text='```repl\nanswer = "42"\nFINAL_VAR("answer")\n```')

    curated, _manifest, patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["curation"]["status"] == "strict_sft"
    assert patches == []
    assert rows[0]["completion"] == [{"role": "assistant", "content": '```repl\nanswer = "42"\nFINAL_VAR("answer")\n```'}]


def test_curator_moves_outside_final_var_into_repl() -> None:
    record = _curation_record(response_text="FINAL_VAR(answer)")

    curated, _manifest, patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["curation"]["status"] == "strict_sft"
    assert patches[0]["repair_type"] == "final_var_outside_repl"
    assert rows[0]["completion"] == [{"role": "assistant", "content": '```repl\nFINAL_VAR("answer")\n```'}]


def test_curator_excludes_reviewed_weak_subcall_traces() -> None:
    record = _curation_record(response_text="FINAL(Harold Wilson)", plain_response="NOT_FOUND")
    record["source_id"] = "frames-0029"
    record["answer"] = "Harold Wilson"
    record["final_answer"] = "Harold Wilson"

    curated, _manifest, _patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["curation"]["status"] == "audit_only"
    assert "weak_subcall_signal" in curated[0]["curation"]["tags"]
    assert rows == []


def test_curator_rewrites_sft_system_prompt_to_current_variant() -> None:
    record = _curation_record(response_text="FINAL(42)")
    record["prompt_variant"] = "sanjaya_text_depth1_llm_only_v1"
    record["segments"][1]["prompt_messages"] = [
        {"role": "system", "content": "old prompt with FINAL_VAR(variable_name)"},
        {"role": "user", "content": "question"},
    ]

    _curated, _manifest, _patches, rows = curate_sft_traces.curate_records([record])

    assert "FINAL_VAR(\"variable_name\")" in rows[0]["prompt"][0]["content"]
    assert rows[0]["prompt_rewritten_to_current_variant"] is True


def test_curator_requires_non_empty_subcall_for_strict_sft() -> None:
    record = _curation_record(response_text="FINAL(42)", plain_response="")

    curated, _manifest, _patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["curation"]["status"] == "audit_only"
    assert "all_subcalls_empty" in curated[0]["curation"]["tags"]
    assert rows == []


def test_curator_splits_code_plus_final_with_existing_feedback() -> None:
    response = "```repl\nprint(42)\n```\nFINAL(42)"
    record = _curation_record(response_text=response)

    curated, _manifest, patches, rows = curate_sft_traces.curate_records([record])

    assert curated[0]["curation"]["status"] == "strict_sft"
    assert patches[0]["repair_type"] == "code_plus_final_split"
    assert [row["row_role"] for row in rows] == ["analysis_before_final", "synthetic_final_after_feedback"]
    assert rows[0]["completion"][0]["content"] == "```repl\nprint(42)\n```"
    assert rows[1]["completion"][0]["content"] == "FINAL(42)"


def test_curator_does_not_mutate_input_records() -> None:
    record = _curation_record(response_text="42", plain_response="")
    before = copy.deepcopy(record)

    curate_sft_traces.curate_records([record])

    assert record == before


def test_trace_audit_repl_execution_preserves_raw_outputs_and_subcalls() -> None:
    execution = SimpleNamespace(
        stdout="x" * 5000,
        stderr="err",
        final_answer="42",
        execution_time=0.5,
        locals={"answer": 42, "context": "full context"},
        rlm_calls=[
            SimpleNamespace(
                root_model="z-ai/glm-5",
                prompt="sub prompt",
                response="sub response",
                execution_time=1.25,
                metadata={"kind": "plain_query"},
            )
        ],
    )

    audit = trace_run.audit_repl_execution("print('hi')", execution)

    assert audit["code"] == "print('hi')"
    assert audit["stdout"] == "x" * 5000
    assert audit["stderr"] == "err"
    assert audit["locals"]["answer"] == {"type": "int", "value": 42}
    assert audit["locals"]["context"] == {"__omitted__": "REPL context variable", "type": "str", "char_count": 12}
    assert audit["llm_calls"][0]["prompt"] == "sub prompt"
    assert audit["llm_calls"][0]["response"] == "sub response"


def test_sft_split_groups_by_source_id() -> None:
    rows = [
        {"source_id": "a", "prompt": [], "completion": []},
        {"source_id": "a", "prompt": [], "completion": []},
        {"source_id": "b", "prompt": [], "completion": []},
        {"source_id": "b", "prompt": [], "completion": []},
    ]

    dataset = export_sft.split_rows_by_source(rows, eval_ratio=0.5, seed=0)
    train_ids = set(dataset["train"]["source_id"])
    eval_ids = set(dataset["eval"]["source_id"])

    assert train_ids
    assert eval_ids
    assert train_ids.isdisjoint(eval_ids)


def test_curated_v2_sft_config_is_full_sft_for_qwen_instruct() -> None:
    config_path = REPO_ROOT / "configs" / "rlm_sft" / "local_8xrtx6000ada_48gb_qwen3_4b_instruct_curated_v2.toml"
    config = tomllib.loads(config_path.read_text())

    assert config["model"]["name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert config["tokenizer"]["name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert "lora" not in config["model"]
    assert config["data"]["name"] == "lsteno/rlm-rlvr-sft-v2-conversations"
    assert config["val"]["data"]["name"] == "lsteno/rlm-rlvr-sft-v2-conversations"
    assert config["data"]["loss_mask"] == {"system": False, "user": False, "assistant": True, "tool": False}
    assert config["data"]["seq_len"] == 32768
    assert config["data"]["pack_function"] == "cat"


def test_curated_v3_per_root_turn_sft_config_masks_prompt_history() -> None:
    config_path = (
        REPO_ROOT
        / "configs"
        / "rlm_sft"
        / "local_8xa100_80gb_qwen3_4b_instruct_curated_v3_per_root_turn.toml"
    )
    config = tomllib.loads(config_path.read_text())

    assert config["model"]["name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert config["tokenizer"]["name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert "lora" not in config["model"]
    assert config["model"]["seq_len"] == 32768
    assert config["model"]["cp"] == 4
    assert config["deployment"]["num_gpus"] == 8
    assert config["data"]["name"] == "lsteno/rlm-rlvr-sft-v3-per-root-turn"
    assert config["val"]["data"]["name"] == "lsteno/rlm-rlvr-sft-v3-per-root-turn"
    assert config["data"]["loss_mask"] == {
        "system": False,
        "user": False,
        "assistant": True,
        "tool": False,
        "train_on_prompt": False,
    }
    assert config["val"]["data"]["loss_mask"]["train_on_prompt"] is False
    assert config["data"]["seq_len"] == 32768
    assert config["data"]["pack_function"] == "cat"


def test_curated_v3_qwen3_8b_sft_config_uses_non_thinking_dataset_and_paper_style_batch() -> None:
    config_path = (
        REPO_ROOT
        / "configs"
        / "rlm_sft"
        / "local_8xa100_80gb_qwen3_8b_nonthinking_curated_v3_per_root_turn.toml"
    )
    config = tomllib.loads(config_path.read_text())

    assert config["model"]["name"] == "Qwen/Qwen3-8B"
    assert config["tokenizer"]["name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert "lora" not in config["model"]
    assert config["model"]["seq_len"] == 32768
    assert config["model"]["cp"] == 4
    assert config["deployment"]["num_gpus"] == 8
    assert config["max_steps"] == 180
    assert config["data"]["batch_size"] == 64
    assert config["data"]["name"] == "lsteno/rlm-rlvr-sft-v3-per-root-turn-qwen3-8b-nonthinking"
    assert config["val"]["data"]["name"] == "lsteno/rlm-rlvr-sft-v3-per-root-turn-qwen3-8b-nonthinking"
    assert config["data"]["loss_mask"] == {
        "system": False,
        "user": False,
        "assistant": True,
        "tool": False,
        "train_on_prompt": False,
    }
    assert config["val"]["data"]["loss_mask"]["train_on_prompt"] is False
    assert config["data"]["seq_len"] == 32768
    assert config["data"]["pack_function"] == "cat"


def test_upload_latest_weights_selects_largest_step_and_builds_hf_commands(tmp_path, monkeypatch) -> None:
    (tmp_path / "weights" / "step_2").mkdir(parents=True)
    (tmp_path / "weights" / "step_10").mkdir(parents=True)
    (tmp_path / "weights" / "step_10" / "model.safetensors").write_text("weights")
    (tmp_path / "weights" / "not_a_step").mkdir()
    monkeypatch.setenv("HF_TOKEN", "test-token")
    calls: list[tuple[str, str]] = []

    class _FakeHfApi:
        def __init__(self, token: str) -> None:
            calls.append(("init", token))

        def create_repo(self, **kwargs) -> None:
            calls.append(("create_repo", kwargs["repo_id"]))

        def upload_folder(self, **kwargs) -> None:
            calls.append(("upload_folder", str(kwargs["folder_path"])))

        def list_repo_files(self, **kwargs) -> list[str]:
            calls.append(("list_repo_files", kwargs["repo_id"]))
            return ["model.safetensors"]

    monkeypatch.setattr(upload_latest_weights, "HfApi", _FakeHfApi)

    selected = upload_latest_weights.upload_latest_weights(tmp_path, "lsteno/test-model")

    assert selected == tmp_path / "weights" / "step_10"
    assert calls == [
        ("init", "test-token"),
        ("create_repo", "lsteno/test-model"),
        ("upload_folder", str(tmp_path / "weights" / "step_10")),
        ("list_repo_files", "lsteno/test-model"),
    ]


def test_glm5_trace_configs_are_depth_one_llm_only() -> None:
    for name, workers in [
        ("generate_glm5_sft_10.toml", 1),
        ("generate_glm5_sft_smoke.toml", 1),
        ("generate_glm5_sft_full.toml", 4),
    ]:
        config = trace_run.load_toml(REPO_ROOT / "pipelines" / "rlm_traces" / "examples" / name)
        assert config["model"]["endpoint_id"] == "glm-5"
        assert config["model"]["extra_body"]["reasoning"]["enabled"] is False
        assert config["dataset"]["split"] == "sft_traces"
        assert config["rollout"]["max_depth"] == 1
        assert config["rollout"]["disable_recursive_subcalls"] is True
        assert config["rollout"]["max_iterations"] == 12
        assert config["rollout"]["max_total_subcalls"] == 60
        assert config["rollout"]["include_budget_reminder"] is False
        assert config["max_workers"] == workers
        if name == "generate_glm5_sft_10.toml":
            assert config["run_name"] == "glm5-sft-10-delegate"
            assert config["judge"]["provider"] == "openai_compatible"
            assert config["judge"]["model"] == "google/gemini-3-flash-preview"
        else:
            assert config["judge"]["provider"] == "vertex"
            assert config["judge"]["model"] == "gemini-3-flash-preview"


def test_gpt54_full_trace_config_uses_process_workers() -> None:
    config = trace_run.load_toml(
        REPO_ROOT
        / "pipelines"
        / "rlm_traces"
        / "examples"
        / "generate_prime_gpt54_vertex_flash_lite_sft_full.toml"
    )

    assert config["max_workers"] == 6
    assert config["worker_backend"] == "process"
    assert config["rollout"]["repl_backend"] == "local"
    assert config["llm_subcall"]["provider"] == "vertex"


def test_missing_trace_driver_failure_predicate_requires_correctness_and_subcalls() -> None:
    good = {"error": None, "final_answer": "42", "exact_match": True, "judge_score": None, "num_llm_subcalls": 1}
    judge_good = {"error": None, "final_answer": "forty two", "exact_match": False, "judge_score": 1.0, "num_llm_subcalls": 2}
    no_subcalls = {"error": None, "final_answer": "42", "exact_match": True, "judge_score": None, "num_llm_subcalls": 0}
    judge_bad = {"error": None, "final_answer": "13", "exact_match": False, "judge_score": 0.0, "num_llm_subcalls": 2}
    errored = {"error": "boom", "final_answer": "42", "exact_match": True, "judge_score": None, "num_llm_subcalls": 1}

    assert run_missing_with_retry.record_is_good_trace(good) is True
    assert run_missing_with_retry.record_is_good_trace(judge_good) is True
    assert run_missing_with_retry.record_is_good_trace(no_subcalls) is False
    assert run_missing_with_retry.record_is_good_trace(judge_bad) is False
    assert run_missing_with_retry.record_is_good_trace(errored) is False


def test_missing_trace_driver_computes_failed_ids_from_latest_records() -> None:
    records = [
        {"source_id": "a", "error": None, "final_answer": "old", "exact_match": False, "judge_score": 0.0, "num_llm_subcalls": 1},
        {"source_id": "a", "error": None, "final_answer": "42", "exact_match": True, "judge_score": None, "num_llm_subcalls": 1},
        {"source_id": "b", "error": None, "final_answer": "42", "exact_match": True, "judge_score": None, "num_llm_subcalls": 0},
    ]

    assert run_missing_with_retry.failed_source_ids(records, ["a", "b", "c"]) == ["b", "c"]


def test_missing_trace_driver_collects_attempted_source_ids(tmp_path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            [
                '{"source_id": "source-a"}',
                '{"source_id": "source-b"}',
                "",
            ]
        )
    )

    assert run_missing_with_retry.collect_attempted_source_ids([records_path]) == {"source-a", "source-b"}


def test_trace_run_progress_bar_renders_counts() -> None:
    assert trace_run.render_progress_bar(5, 10, width=10) == "[#####-----]"
    assert trace_run.fmt_duration(65) == "1m05s"


def test_missing_trace_driver_renders_gpt54_vertex_config(tmp_path) -> None:
    rendered = run_missing_with_retry.render_trace_config(
        run_name="run-a",
        output_dir=tmp_path,
        endpoints_path="configs/endpoints.toml",
        model_cfg={
            "model": "openai/gpt-5.4",
            "url": "https://api.pinference.ai/api/v1",
            "api_key_env": "PRIME_API_KEY",
            "extra_body": {"reasoning": {"enabled": False}},
        },
        llm_subcall_cfg={
            "provider": "vertex",
            "model": "gemini-3-flash-preview",
            "vertex_project_env": "GOOGLE_CLOUD_PROJECT",
            "vertex_location": "global",
            "thinking_level": "medium",
            "empty_response_max_attempts": 3,
        },
        dataset_cfg={"dataset_id": "lsteno/BEEG-agents", "split": "sft_traces", "seed": 42},
        rollout_cfg={
            "prompt_variants": ["sanjaya_text_depth1_llm_only_v1"],
            "max_iterations": 15,
            "max_depth": 1,
            "disable_recursive_subcalls": True,
            "turn_max_tokens": 4096,
            "subcall_max_tokens": 4096,
            "max_prompt_tokens": 2000000,
            "max_total_subcalls": 80,
            "max_batched_subcalls": 80,
            "include_budget_reminder": False,
            "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
        },
        judge_cfg={
            "enabled": True,
            "provider": "vertex",
            "model": "gemini-3-flash-preview",
            "vertex_project_env": "GOOGLE_CLOUD_PROJECT",
            "vertex_location": "global",
            "thinking_level": "medium",
        },
        source_ids=["source-b", "source-a"],
        max_workers=6,
        worker_backend="process",
        resume=True,
    )
    parsed_path = tmp_path / "config.toml"
    parsed_path.write_text(rendered)
    parsed = trace_run.load_toml(parsed_path)

    assert parsed["model"]["model"] == "openai/gpt-5.4"
    assert parsed["model"]["extra_body"]["reasoning"]["enabled"] is False
    assert parsed["llm_subcall"]["provider"] == "vertex"
    assert parsed["llm_subcall"]["empty_response_max_attempts"] == 3
    assert parsed["dataset"]["source_ids"] == ["source-b", "source-a"]
    assert parsed["judge"]["model"] == "gemini-3-flash-preview"
    assert parsed["max_workers"] == 6
    assert parsed["worker_backend"] == "process"


def test_missing_trace_driver_omits_llm_subcall_section_for_same_root_mode(tmp_path) -> None:
    rendered = run_missing_with_retry.render_trace_config(
        run_name="run-a",
        output_dir=tmp_path,
        endpoints_path="configs/endpoints.toml",
        model_cfg={
            "model": "openai/gpt-5.4",
            "url": "https://api.pinference.ai/api/v1",
            "api_key_env": "PRIME_API_KEY",
            "extra_body": {"reasoning": {"enabled": False}},
        },
        llm_subcall_cfg=None,
        dataset_cfg={"dataset_id": "lsteno/BEEG-agents", "split": "sft_traces", "seed": 42},
        rollout_cfg={
            "prompt_variants": ["sanjaya_text_depth1_llm_only_v1"],
            "max_iterations": 15,
            "max_depth": 1,
            "disable_recursive_subcalls": True,
            "turn_max_tokens": 4096,
            "subcall_max_tokens": 4096,
            "max_prompt_tokens": 2000000,
            "max_total_subcalls": 80,
            "max_batched_subcalls": 80,
            "subcall_batch_max_workers": 2,
            "include_budget_reminder": False,
            "tokenizer_name": "Qwen/Qwen3-4B-Instruct-2507",
        },
        judge_cfg={
            "enabled": True,
            "provider": "vertex",
            "model": "gemini-3-flash-preview",
            "vertex_project_env": "GOOGLE_CLOUD_PROJECT",
            "vertex_location": "global",
            "thinking_level": "medium",
        },
        source_ids=["source-a"],
        max_workers=2,
        worker_backend="process",
        resume=True,
    )
    parsed_path = tmp_path / "config.toml"
    parsed_path.write_text(rendered)
    parsed = trace_run.load_toml(parsed_path)

    assert "llm_subcall" not in parsed
    assert parsed["rollout"]["subcall_batch_max_workers"] == 2
    assert parsed["judge"]["provider"] == "vertex"


def test_worker_error_record_does_not_mark_success() -> None:
    example = trace_run.Example(
        example_id=7,
        source_id="source-7",
        question="question?",
        answer="answer",
        acceptable_answers=["answer"],
        context="context text",
        info={},
    )
    endpoint = trace_run.EndpointConfig(
        endpoint_id=None,
        model="openai/gpt-5.4",
        url="https://api.pinference.ai/api/v1",
        api_key_env="PRIME_API_KEY",
        api_key="secret",
    )

    record = trace_run.make_error_record(
        example=example,
        endpoint=endpoint,
        prompt_variant="variant",
        error="RuntimeError: boom",
        elapsed_seconds=1.25,
    )

    assert record["source_id"] == "source-7"
    assert record["error"] == "RuntimeError: boom"
    assert record["endpoint"]["api_key"] == "<redacted>"
    assert record["exact_match"] is False
    assert trace_run.record_is_accepted(record) is False


def test_run_rollout_worker_is_spawn_picklable() -> None:
    example = trace_run.Example(
        example_id=7,
        source_id="spawn-source-7",
        question="question?",
        answer="answer",
        acceptable_answers=["answer"],
        context="context text",
        info={},
    )
    endpoint = trace_run.EndpointConfig(
        endpoint_id=None,
        model="openai/gpt-5.4",
        url="https://api.pinference.ai/api/v1",
        api_key_env="PRIME_API_KEY",
        api_key="secret",
    )
    plain_subcall = trace_run.PlainSubcallConfig(
        provider="vertex",
        model="gemini-3-flash-preview",
        vertex_project=None,
    )
    runtime_payload = {
        "max_depth": 0,
        "max_iterations": 1,
        "turn_max_tokens": 16,
        "subcall_max_tokens": 16,
        "max_prompt_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "tokenizer_name": None,
        "inference_mode": "local",
        "inference_base_url": endpoint.url,
        "inference_api_key": endpoint.api_key,
        "repl_backend": "local",
        "repl_timeout_seconds": None,
        "repl_fast_timeout_seconds": None,
        "prompt_variant": "sanjaya_text_depth1_llm_only_v1",
        "subcall_budget_enabled": True,
        "max_total_subcalls": 1,
        "max_batched_subcalls": 1,
        "capture_prompt_messages": True,
        "include_budget_reminder": False,
        "live_trace_dir": None,
    }

    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        future = pool.submit(
            trace_run.run_rollout_from_payload,
            example=example,
            endpoint=endpoint,
            runtime_config_payload=runtime_payload,
            prompt_variant="sanjaya_text_depth1_llm_only_v1",
            judge_config=None,
            plain_subcall_config=plain_subcall,
            disable_recursive_subcalls=True,
        )
        record = future.result(timeout=30)

    assert record["source_id"] == "spawn-source-7"
    assert record["error"]
    assert "GOOGLE_CLOUD_PROJECT must be set" in record["error"]
