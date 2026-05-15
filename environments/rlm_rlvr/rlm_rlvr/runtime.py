from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random
import threading
import time
from typing import Any, Callable, TypeVar

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .external_rlm import CodeBlock, QueryMetadata, RLMIteration, build_system_prompt, build_user_prompt, find_code_blocks, make_feedback_messages
from .live_trace import write_live_trace
from .parsing import extract_final_answer
from .prompt_variants import DEFAULT_PROMPT_VARIANT
from .repl import create_repl
from .trace import append_step_trace, make_call_trace, make_segment, prompt_provenance


class SubcallPromptTooLargeError(ValueError):
    pass


@dataclass
class RuntimeConfig:
    max_depth: int = 2
    max_iterations: int = 15
    turn_max_tokens: int = 192
    subcall_max_tokens: int = 128
    max_prompt_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    execution_output_char_limit: int = 4000
    tokenizer_name: str | None = None
    inference_mode: str = "hosted"
    inference_base_url: str | None = None
    inference_api_key: str | None = None
    llm_subcall_provider: str = "openai_compatible"
    llm_subcall_model: str | None = None
    llm_subcall_base_url: str | None = None
    llm_subcall_api_key: str | None = None
    llm_subcall_default_headers: dict[str, str] | None = None
    llm_subcall_vertex_project: str | None = None
    llm_subcall_vertex_location: str = "global"
    llm_subcall_thinking_level: str | None = "medium"
    llm_subcall_empty_response_max_attempts: int = 1
    llm_subcall_empty_response_base_retry_seconds: float = 1.0
    llm_subcall_empty_response_max_retry_seconds: float = 30.0
    repl_backend: str = "local"
    repl_backend_kwargs: dict[str, Any] | None = None
    repl_timeout_seconds: float | None = None
    repl_fast_timeout_seconds: float | None = None
    prompt_variant: str = DEFAULT_PROMPT_VARIANT
    live_trace_dir: str | None = "outputs/rlm_rlvr/live_traces"
    subcall_prompt_limit_ratio: float = 0.85
    subcall_budget_enabled: bool = False
    max_total_subcalls: int = 80
    max_batched_subcalls: int = 80
    capture_prompt_messages: bool = False
    include_budget_reminder: bool = True


@dataclass
class TokenPayload:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]
    completion_mask: list[bool]
    prompt_token_count: int | None = None
    completion_token_count: int | None = None
    metadata: dict[str, Any] | None = None


_T = TypeVar("_T")

_SUBCALL_RETRY_MAX_ATTEMPTS = 6
_SUBCALL_RETRY_BASE_SECONDS = 1.0
_SUBCALL_RETRY_MAX_SECONDS = 30.0


def _exception_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def _is_retryable_subcall_exception(exc: BaseException) -> bool:
    status_code = _exception_status_code(exc)
    if status_code == 429 or status_code in {500, 502, 503, 504}:
        return True

    marker = str(exc).upper()
    return any(
        token in marker
        for token in (
            "429",
            "RATE_LIMIT",
            "RESOURCE_EXHAUSTED",
            "TOO MANY REQUESTS",
            "UNAVAILABLE",
            "SERVICE UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "INTERNAL",
        )
    )


def _sleep_before_subcall_retry(attempt: int) -> None:
    delay = min(_SUBCALL_RETRY_MAX_SECONDS, _SUBCALL_RETRY_BASE_SECONDS * (2**attempt))
    jitter = random.uniform(0.0, min(1.0, delay * 0.25))
    time.sleep(delay + jitter)


def _call_subcall_with_retries(request: Callable[[], _T]) -> _T:
    for attempt in range(_SUBCALL_RETRY_MAX_ATTEMPTS):
        try:
            return request()
        except Exception as exc:
            if attempt == _SUBCALL_RETRY_MAX_ATTEMPTS - 1 or not _is_retryable_subcall_exception(exc):
                raise
            _sleep_before_subcall_retry(attempt)
    raise RuntimeError("unreachable subcall retry state")


class SyncInferenceSession:
    _tokenizers: dict[str, PreTrainedTokenizerBase] = {}

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None,
        model_name: str,
        tokenizer_name: str | None,
        max_prompt_tokens: int | None,
        enable_vllm_extra_body: bool = False,
        request_logprobs: bool = True,
        retry_transient_errors: bool = False,
        openai_extra_body: dict[str, Any] | None = None,
        enable_token_accounting: bool = True,
    ):
        self.model_name = model_name
        self.max_prompt_tokens = max_prompt_tokens
        self.enable_vllm_extra_body = enable_vllm_extra_body
        self.request_logprobs = request_logprobs
        self.retry_transient_errors = retry_transient_errors
        self.openai_extra_body = openai_extra_body or {}
        self.enable_token_accounting = enable_token_accounting
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            default_headers=default_headers,
        )
        self.tokenizer = None
        if self.enable_token_accounting:
            tokenizer_key = tokenizer_name or model_name
            if tokenizer_key not in self._tokenizers:
                self._tokenizers[tokenizer_key] = AutoTokenizer.from_pretrained(tokenizer_key, trust_remote_code=True)
            self.tokenizer = self._tokenizers[tokenizer_key]

    def render_prompt_ids(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
        if self.tokenizer is None:
            return []
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
        )
        return list(rendered["input_ids"])

    def count_text_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _coerce_token_ids(token_ids: Any, *, fallback: list[int]) -> list[int]:
        if token_ids is None:
            return list(fallback)
        return [int(token_id) for token_id in token_ids]

    @staticmethod
    def _coerce_logprobs(logprobs: Any, *, completion_len: int) -> list[float]:
        if hasattr(logprobs, "content") and logprobs.content is not None:
            values = [float(item.logprob) for item in logprobs.content]
        elif isinstance(logprobs, dict) and logprobs.get("content") is not None:
            values = [float(item["logprob"]) for item in logprobs["content"]]
        else:
            values = []

        if len(values) < completion_len:
            values.extend([0.0] * (completion_len - len(values)))
        return values[:completion_len]

    @staticmethod
    def _usage_token_count(usage: Any, key: str) -> int | None:
        if usage is None:
            return None
        if isinstance(usage, dict):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        if value is None:
            return None
        return int(value)

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, TokenPayload]:
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if getattr(self, "request_logprobs", True):
            request_body["logprobs"] = True
        if self.enable_vllm_extra_body:
            request_body.update({
                "return_token_ids": True,
                "top_k": -1,
                "min_p": 0.0,
            })
        request_body.update(getattr(self, "openai_extra_body", {}))
        request = lambda: self.client.post(
            "chat/completions",
            body=request_body,
            cast_to=ChatCompletion,
        )
        response = _call_subcall_with_retries(request) if getattr(self, "retry_transient_errors", False) else request()
        assert response.choices is not None and len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message is not None
        text = (choice.message.content or "").strip()
        fallback_completion_ids = self.tokenizer.encode(text, add_special_tokens=False) if self.tokenizer is not None else []
        completion_ids = self._coerce_token_ids(
            getattr(choice, "token_ids", None),
            fallback=fallback_completion_ids,
        )
        prompt_ids = self.render_prompt_ids(messages, add_generation_prompt=True)
        prompt_token_ids = self._coerce_token_ids(getattr(response, "prompt_token_ids", None), fallback=prompt_ids)
        completion_logprobs = self._coerce_logprobs(getattr(choice, "logprobs", None), completion_len=len(completion_ids))
        usage = getattr(response, "usage", None)
        prompt_token_count = self._usage_token_count(usage, "prompt_tokens")
        completion_token_count = self._usage_token_count(usage, "completion_tokens")
        payload = TokenPayload(
            prompt_ids=prompt_token_ids,
            completion_ids=completion_ids,
            completion_logprobs=completion_logprobs,
            completion_mask=[True] * len(completion_ids),
            prompt_token_count=prompt_token_count if prompt_token_count is not None else len(prompt_token_ids),
            completion_token_count=completion_token_count if completion_token_count is not None else len(completion_ids),
        )
        return text, payload


def _load_google_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Vertex Gemini subcalls require the google-genai package. "
            "Install environments/rlm_rlvr with google-genai[aiohttp]>=1.51.0."
        ) from exc
    return genai, types


def _normalise_vertex_model_name(model_name: str) -> str:
    if model_name.startswith("google/"):
        model_name = model_name.removeprefix("google/")
    if model_name == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return model_name


def _usage_field(usage: Any, key: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    if value is None:
        return None
    return int(value)


class VertexGeminiSession:
    def __init__(
        self,
        *,
        model_name: str,
        project: str,
        location: str,
        tokenizer_name: str | None,
        max_prompt_tokens: int | None,
        thinking_level: str | None = "medium",
        empty_response_max_attempts: int = 1,
        empty_response_base_retry_seconds: float = 1.0,
        empty_response_max_retry_seconds: float = 30.0,
    ):
        genai, _ = _load_google_genai()
        self.model_name = _normalise_vertex_model_name(model_name)
        self.max_prompt_tokens = max_prompt_tokens
        self.thinking_level = thinking_level
        self.retry_transient_errors = True
        self.empty_response_max_attempts = max(1, int(empty_response_max_attempts))
        self.empty_response_base_retry_seconds = float(empty_response_base_retry_seconds)
        self.empty_response_max_retry_seconds = float(empty_response_max_retry_seconds)
        self.client = genai.Client(vertexai=True, project=project, location=location)
        tokenizer_key = tokenizer_name or model_name
        if tokenizer_key not in SyncInferenceSession._tokenizers:
            SyncInferenceSession._tokenizers[tokenizer_key] = AutoTokenizer.from_pretrained(tokenizer_key, trust_remote_code=True)
        self.tokenizer = SyncInferenceSession._tokenizers[tokenizer_key]

    def render_prompt_ids(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
        )
        return list(rendered["input_ids"])

    def count_text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _message_text(message: dict[str, str]) -> str:
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)

    def _convert_messages(self, messages: list[dict[str, str]]) -> tuple[str | None, list[Any]]:
        _, types = _load_google_genai()
        system_parts: list[str] = []
        contents: list[Any] = []
        for message in messages:
            role = str(message.get("role", "user"))
            text = self._message_text(message)
            if role == "system":
                system_parts.append(text)
                continue
            vertex_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=vertex_role, parts=[types.Part.from_text(text=text)]))
        system_instruction = "\n\n".join(part for part in system_parts if part).strip() or None
        if not contents:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="")]))
        return system_instruction, contents

    def _generate_config(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        system_instruction: str | None,
    ) -> Any:
        _, types = _load_google_genai()
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction
        if self.thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _usage_metadata(response: Any) -> Any:
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            return usage
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped.get("usage_metadata")
        return None

    @staticmethod
    def _completion_token_count(usage: Any, *, prompt_token_count: int | None, fallback: int) -> int:
        total_token_count = _usage_field(usage, "total_token_count")
        if total_token_count is not None and prompt_token_count is not None:
            return max(0, total_token_count - prompt_token_count)
        candidates_token_count = _usage_field(usage, "candidates_token_count")
        thoughts_token_count = _usage_field(usage, "thoughts_token_count")
        if candidates_token_count is not None or thoughts_token_count is not None:
            return int(candidates_token_count or 0) + int(thoughts_token_count or 0)
        return fallback

    @staticmethod
    def _completion_token_count_from_usage(usage: Any) -> int | None:
        prompt_token_count = _usage_field(usage, "prompt_token_count")
        total_token_count = _usage_field(usage, "total_token_count")
        if total_token_count is not None and prompt_token_count is not None:
            return max(0, total_token_count - prompt_token_count)
        candidates_token_count = _usage_field(usage, "candidates_token_count")
        thoughts_token_count = _usage_field(usage, "thoughts_token_count")
        if candidates_token_count is not None or thoughts_token_count is not None:
            return int(candidates_token_count or 0) + int(thoughts_token_count or 0)
        return None

    @staticmethod
    def _sum_optional_counts(values: list[int | None]) -> int | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        return sum(present)

    def _empty_response_retry_attempts(self) -> int:
        return max(1, int(getattr(self, "empty_response_max_attempts", 1)))

    def _sleep_before_empty_response_retry(self, attempt: int) -> None:
        base_seconds = float(getattr(self, "empty_response_base_retry_seconds", 1.0))
        max_seconds = float(getattr(self, "empty_response_max_retry_seconds", 30.0))
        delay = min(max_seconds, base_seconds * (2**attempt))
        jitter = random.uniform(0.0, min(1.0, delay * 0.25))
        time.sleep(delay + jitter)

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, TokenPayload]:
        system_instruction, contents = self._convert_messages(messages)
        config = self._generate_config(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system_instruction=system_instruction,
        )
        responses: list[Any] = []
        text = ""
        max_attempts = self._empty_response_retry_attempts()
        for attempt in range(max_attempts):
            response = _call_subcall_with_retries(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            )
            responses.append(response)
            text = str(getattr(response, "text", "") or "").strip()
            if text or attempt == max_attempts - 1:
                break
            self._sleep_before_empty_response_retry(attempt)
        prompt_ids = self.render_prompt_ids(messages, add_generation_prompt=True)
        completion_ids = self.tokenizer.encode(text, add_special_tokens=False)
        usages = [self._usage_metadata(response) for response in responses]
        prompt_token_count = self._sum_optional_counts([_usage_field(usage, "prompt_token_count") for usage in usages])
        completion_token_count = self._sum_optional_counts(
            [self._completion_token_count_from_usage(usage) for usage in usages if usage is not None]
        )
        if completion_token_count is None:
            completion_token_count = len(completion_ids)
        payload = TokenPayload(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=[0.0] * len(completion_ids),
            completion_mask=[True] * len(completion_ids),
            prompt_token_count=prompt_token_count if prompt_token_count is not None else len(prompt_ids),
            completion_token_count=completion_token_count,
            metadata={
                "generation_attempt_count": len(responses),
                "empty_response_retry_count": max(0, len(responses) - 1),
                "empty_response_exhausted": not text and len(responses) >= max_attempts,
            },
        )
        return text, payload


class RecursiveRuntime:
    _MESSAGE_OVERHEAD_CHARS = 32
    _GENERATION_PROMPT_OVERHEAD_CHARS = 16
    _CHARS_PER_TOKEN_ESTIMATE = 4.0
    _DEFAULT_BATCH_MAX_WORKERS = 8

    def __init__(self, state: dict[str, Any], config: RuntimeConfig):
        self.state = state
        self.config = config
        self._state_lock = threading.RLock()
        self._thread_context = threading.local()

    @property
    def session(self) -> SyncInferenceSession:
        return self.state["_sync_session"]

    @property
    def plain_llm_session(self) -> SyncInferenceSession:
        return self.state.get("_plain_llm_session") or self.session

    def _context_window_tokens(self) -> int | None:
        if self.config.max_prompt_tokens is not None:
            return int(self.config.max_prompt_tokens)

        tokenizer = getattr(self.session, "tokenizer", None)
        model_max_length = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and 0 < model_max_length <= 10_000_000:
            return model_max_length
        return None

    def _subcall_prompt_limit_tokens(self) -> int | None:
        context_window = self._context_window_tokens()
        if context_window is None:
            return None
        return max(1, int(context_window * float(self.config.subcall_prompt_limit_ratio)))

    def _current_call_depth(self) -> int:
        depth = getattr(self._thread_context, "call_depth", None)
        if depth is not None:
            return int(depth)
        return int(self.state.get("current_call_depth", 0))

    def _current_call_id(self) -> int:
        call_id = getattr(self._thread_context, "call_id", None)
        if call_id is not None:
            return int(call_id)
        return int(self.state.get("current_call_id", 0))

    def _current_parent_call_id(self) -> int | None:
        parent_call_id = getattr(self._thread_context, "parent_call_id", None)
        if parent_call_id is not None:
            return int(parent_call_id)
        state_parent = self.state.get("current_parent_call_id")
        return None if state_parent is None else int(state_parent)

    def _current_branch_max_depth(self) -> int:
        max_depth = getattr(self._thread_context, "branch_max_depth", None)
        if max_depth is not None:
            return int(max_depth)
        return int(self.state.get("current_branch_max_depth", self.config.max_depth))

    def _set_thread_context(
        self,
        *,
        call_id: int,
        parent_call_id: int | None,
        depth: int,
        max_depth: int,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        previous_call_id = getattr(self._thread_context, "call_id", None)
        previous_parent_call_id = getattr(self._thread_context, "parent_call_id", None)
        previous_depth = getattr(self._thread_context, "call_depth", None)
        previous_max_depth = getattr(self._thread_context, "branch_max_depth", None)
        self._thread_context.call_id = int(call_id)
        if parent_call_id is None:
            self._thread_context.parent_call_id = None
        else:
            self._thread_context.parent_call_id = int(parent_call_id)
        self._thread_context.call_depth = int(depth)
        self._thread_context.branch_max_depth = int(max_depth)
        return previous_call_id, previous_parent_call_id, previous_depth, previous_max_depth

    def _restore_thread_context(self, previous: tuple[int | None, int | None, int | None, int | None]) -> None:
        previous_call_id, previous_parent_call_id, previous_depth, previous_max_depth = previous
        if previous_call_id is None:
            if hasattr(self._thread_context, "call_id"):
                del self._thread_context.call_id
        else:
            self._thread_context.call_id = int(previous_call_id)
        if previous_parent_call_id is None:
            self._thread_context.parent_call_id = None
        else:
            self._thread_context.parent_call_id = int(previous_parent_call_id)
        if previous_depth is None:
            if hasattr(self._thread_context, "call_depth"):
                del self._thread_context.call_depth
        else:
            self._thread_context.call_depth = int(previous_depth)
        if previous_max_depth is None:
            if hasattr(self._thread_context, "branch_max_depth"):
                del self._thread_context.branch_max_depth
        else:
            self._thread_context.branch_max_depth = int(previous_max_depth)

    def _estimate_message_chars(self, messages: list[dict[str, str]]) -> int:
        total = self._GENERATION_PROMPT_OVERHEAD_CHARS
        for message in messages:
            total += self._MESSAGE_OVERHEAD_CHARS
            total += len(str(message.get("role", "")))
            total += len(str(message.get("content", "")))
        return total

    def _subcall_prompt_limit_chars(self) -> tuple[int | None, int | None, int | None]:
        context_window = self._context_window_tokens()
        limit_tokens = self._subcall_prompt_limit_tokens()
        if context_window is None or limit_tokens is None:
            return context_window, limit_tokens, None
        limit_chars = max(1, int(limit_tokens * self._CHARS_PER_TOKEN_ESTIMATE))
        return context_window, limit_tokens, limit_chars

    def _estimate_subcall_prompt_chars(self, messages: list[dict[str, str]]) -> int:
        return self._estimate_message_chars(messages)

    def _oversized_subcall_message(
        self,
        *,
        kind: str,
        oversized: list[tuple[int, int]],
        limit_chars: int,
        limit_tokens: int,
        context_window: int,
    ) -> str:
        percent = int(float(self.config.subcall_prompt_limit_ratio) * 100)
        if len(oversized) == 1:
            index, prompt_chars = oversized[0]
            prefix = f"{kind} prompt is too large" if index == 0 else f"{kind} prompt at index {index} is too large"
            return (
                f"{prefix}: estimated prompt size is {prompt_chars} characters, which exceeds the approximate "
                f"{percent}% subcall prompt budget ({limit_chars} chars ~= {limit_tokens}/{context_window} tokens). "
                "Shorten the prompt or pass a smaller excerpt, then retry."
            )

        details = ", ".join(f"{index} ({prompt_chars} chars)" for index, prompt_chars in oversized)
        return (
            f"{kind} prompts are too large: estimated prompt sizes exceed the approximate {percent}% "
            f"subcall prompt budget ({limit_chars} chars ~= {limit_tokens}/{context_window} tokens) for indices {details}. "
            "Shorten those prompts or pass smaller excerpts, then retry."
        )

    def _validate_subcall_messages(
        self,
        *,
        kind: str,
        message_batches: list[list[dict[str, str]]],
    ) -> None:
        context_window, limit_tokens, limit_chars = self._subcall_prompt_limit_chars()
        if context_window is None or limit_tokens is None or limit_chars is None:
            return
        oversized: list[tuple[int, int]] = []
        for index, messages in enumerate(message_batches):
            prompt_chars = self._estimate_subcall_prompt_chars(messages)
            if prompt_chars > limit_chars:
                oversized.append((index, prompt_chars))
        if oversized:
            raise SubcallPromptTooLargeError(
                self._oversized_subcall_message(
                    kind=kind,
                    oversized=oversized,
                    limit_chars=limit_chars,
                    limit_tokens=limit_tokens,
                    context_window=context_window,
                )
            )

    def _write_live_trace(self, *, event: str, active_trace: dict[str, Any] | None = None) -> None:
        with self._state_lock:
            write_live_trace(self.state, event=event, active_trace=active_trace)

    def _recursive_initial_messages(
        self,
        *,
        prompt: str,
        depth: int,
        max_depth: int,
        context_payload: str | None,
    ) -> list[dict[str, str]]:
        repl_context = prompt if context_payload is None else context_payload
        context_metadata = QueryMetadata(repl_context)
        return [
            {
                "role": "system",
                "content": build_system_prompt(
                    depth=depth,
                    max_depth=max_depth,
                    prompt_variant=self.config.prompt_variant,
                    max_prompt_tokens=self.config.max_prompt_tokens,
                    turn_max_tokens=self.config.turn_max_tokens,
                    subcall_max_tokens=self.config.subcall_max_tokens,
                    subcall_budget_enabled=self.config.subcall_budget_enabled,
                    max_total_subcalls=self.config.max_total_subcalls,
                    max_batched_subcalls=self.config.max_batched_subcalls,
                    include_budget_reminder=self.config.include_budget_reminder,
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Your context is a {context_metadata.context_type} with "
                    f"{context_metadata.context_total_length} total characters, and is broken up into chunks "
                    f"of char lengths: {context_metadata.context_lengths}."
                ),
            },
        ]

    def _recursive_first_turn_messages(
        self,
        *,
        prompt: str,
        depth: int,
        max_depth: int,
        context_payload: str | None,
    ) -> list[dict[str, str]]:
        return [
            *self._recursive_initial_messages(
                prompt=prompt,
                depth=depth,
                max_depth=max_depth,
                context_payload=context_payload,
            ),
            build_user_prompt(
                root_prompt=prompt,
                iteration=0,
                context_count=1,
                history_count=0,
            ),
        ]

    def build_finalize_message(self) -> str:
        return (
            "Provide only the final answer now. No explanation. Use FINAL(...) or FINAL_VAR(...). "
            "FINAL(...) is not a Python function; if you put it inside a ```repl code block, the REPL will try "
            "to execute it and fail. Put FINAL(...) outside code blocks with the actual final answer inside it, "
            "not an expression like json.dumps(...)."
        )

    def budget_feedback_message(self) -> dict[str, str] | None:
        if not self.config.subcall_budget_enabled:
            return None
        with self._state_lock:
            remaining = int(self.state.get("subcall_budget_remaining", self.config.max_total_subcalls))
            total = int(self.state.get("subcall_budget_total", self.config.max_total_subcalls))
        return {"role": "user", "content": f"Subcall budget remaining: {remaining}/{total}."}

    def _next_segment_order(self) -> int:
        with self._state_lock:
            order = int(self.state["rlm_segment_counter"])
            self.state["rlm_segment_counter"] = order + 1
            return order

    def _next_call_id(self) -> int:
        with self._state_lock:
            call_id = int(self.state["rlm_call_counter"])
            self.state["rlm_call_counter"] = call_id + 1
            return call_id

    def _append_segment(
        self,
        *,
        payload: TokenPayload,
        depth: int,
        turn_index: int,
        kind: str,
        train_scope: str,
        is_trainable_rlm_turn: bool,
        response_source: str,
        response_text: str,
        messages: list[dict[str, str]],
        call_id: int | None = None,
        parent_call_id: int | None = None,
    ) -> None:
        provenance = prompt_provenance(messages)
        segment = make_segment(
            order=self._next_segment_order(),
            call_id=self._current_call_id() if call_id is None else call_id,
            parent_call_id=self._current_parent_call_id() if parent_call_id is None else parent_call_id,
            depth=depth,
            turn_index=turn_index,
            kind=kind,
            train_scope=train_scope,
            is_trainable_rlm_turn=is_trainable_rlm_turn,
            response_source=response_source,
            prompt_ids=payload.prompt_ids,
            completion_ids=payload.completion_ids,
            completion_logprobs=payload.completion_logprobs,
            completion_mask=payload.completion_mask,
            prompt_token_count=payload.prompt_token_count,
            completion_token_count=payload.completion_token_count,
            temperature=float(self.state.get("sampling_temperature", self.config.temperature)),
            response_text=response_text,
            prompt_fingerprint=provenance["prompt_fingerprint"],
            prompt_message_count=provenance["prompt_message_count"],
            prompt_char_count=provenance["prompt_char_count"],
        )
        if self.config.capture_prompt_messages:
            segment["prompt_messages"] = [
                {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
                for message in messages
            ]
            request_model = self.plain_llm_session.model_name if kind == "plain_query" else self.session.model_name
            request_max_tokens = self.config.subcall_max_tokens if kind == "plain_query" else self.config.turn_max_tokens
            segment["request"] = {
                "model": request_model,
                "max_tokens": int(request_max_tokens),
                "temperature": float(self.state.get("sampling_temperature", self.config.temperature)),
                "top_p": float(self.config.top_p),
            }
        if payload.metadata:
            segment.update(payload.metadata)
        with self._state_lock:
            self.state["rlm_segments"].append(segment)
            prompt_token_count = float(payload.prompt_token_count if payload.prompt_token_count is not None else len(payload.prompt_ids))
            completion_token_count = float(
                payload.completion_token_count if payload.completion_token_count is not None else len(payload.completion_ids)
            )
            self.state["total_model_tokens"] += completion_token_count
            self.state["total_prompt_tokens"] = float(self.state.get("total_prompt_tokens", 0.0)) + prompt_token_count
            self.state["total_completion_tokens"] = float(self.state.get("total_completion_tokens", 0.0)) + completion_token_count
            self.state["total_rollout_tokens"] = float(self.state.get("total_rollout_tokens", 0.0)) + prompt_token_count + completion_token_count
            self.state["max_depth_reached"] = max(int(self.state["max_depth_reached"]), depth)
        self._write_live_trace(event=f"segment:{kind}")

    def _record_subcall(self, *, kind: str, depth: int) -> None:
        with self._state_lock:
            self.state["used_recursion"] = True
            self.state["num_subcalls"] += 1
            self.state["max_depth_reached"] = max(int(self.state["max_depth_reached"]), depth)
            if kind == "plain_query":
                self.state["used_llm_subcalls"] = True
                self.state["num_llm_subcalls"] += 1
            elif kind == "recursive_query":
                self.state["used_rlm_subcalls"] = True
                self.state["num_rlm_subcalls"] += 1
            else:
                raise ValueError(f"Unsupported subcall kind: {kind}")

    def _budget_error_payload(self, *, prompt: str, model: str | None, kind: str) -> dict[str, Any]:
        with self._state_lock:
            remaining = int(self.state.get("subcall_budget_remaining", 0))
            total = int(self.state.get("subcall_budget_total", self.config.max_total_subcalls))
        if remaining > 0:
            message = (
                f"Error: subcall batch fanout limit reached ({self.config.max_batched_subcalls} prompts maximum; "
                f"{remaining}/{total} calls remaining)."
            )
        else:
            message = f"Error: subcall budget exhausted ({remaining}/{total} calls remaining)."
        return {
            "prompt": prompt,
            "model": model or (self.plain_llm_session.model_name if kind == "plain_query" else self.session.model_name),
            "response": message,
            "final_answer": None,
            "depth": self._current_call_depth(),
            "kind": kind,
            "execution_time": 0.0,
            "budget_error": True,
        }

    def _reserve_subcall_budget(self, requested: int) -> int:
        if not self.config.subcall_budget_enabled:
            return requested
        if requested <= 0:
            return 0
        with self._state_lock:
            remaining = int(self.state.get("subcall_budget_remaining", self.config.max_total_subcalls))
            allowed = max(0, min(requested, remaining, int(self.config.max_batched_subcalls)))
            self.state["subcall_budget_remaining"] = remaining - allowed
            if self.state["subcall_budget_remaining"] <= 0:
                self.state["subcall_budget_exhausted"] = True
            return allowed

    def _try_consume_subcall_budget(self) -> bool:
        return self._reserve_subcall_budget(1) == 1

    def _plain_query_messages(self, prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def _recursive_query_messages(
        self,
        prompt: str,
        *,
        child_depth: int,
        effective_max_depth: int,
    ) -> list[dict[str, str]]:
        return self._recursive_first_turn_messages(
            prompt=prompt,
            depth=child_depth,
            max_depth=effective_max_depth,
            context_payload=None,
        )

    def validate_plain_query_batch(self, prompts: list[str]) -> None:
        self._validate_subcall_messages(
            kind="llm_query",
            message_batches=[self._plain_query_messages(prompt) for prompt in prompts],
        )

    def validate_recursive_query_batch(
        self,
        prompts: list[str],
        *,
        max_depth: int | None = None,
    ) -> None:
        parent_depth = self._current_call_depth()
        branch_max_depth = self._current_branch_max_depth()
        message_batches: list[list[dict[str, str]]] = []
        for prompt in prompts:
            child_depth = parent_depth + 1
            if max_depth is None:
                effective_max_depth = branch_max_depth
            else:
                effective_max_depth = min(branch_max_depth, child_depth + max(0, int(max_depth)))
            if child_depth > effective_max_depth:
                message_batches.append(self._plain_query_messages(prompt))
            else:
                message_batches.append(
                    self._recursive_query_messages(
                        prompt,
                        child_depth=child_depth,
                        effective_max_depth=effective_max_depth,
                    )
                )
        self._validate_subcall_messages(kind="rlm_query", message_batches=message_batches)

    def batch_max_workers(self, prompt_count: int, requested_max_workers: int | None = None) -> int:
        if prompt_count <= 0:
            return 1
        if requested_max_workers is not None:
            return max(1, min(int(requested_max_workers), prompt_count))
        return max(1, min(prompt_count, self._DEFAULT_BATCH_MAX_WORKERS))

    def run_plain_query_batch(
        self,
        prompts: list[str],
        *,
        model: str | None = None,
        max_workers: int | None = None,
    ) -> list[dict[str, Any]]:
        if not prompts:
            return []
        allowed = self._reserve_subcall_budget(len(prompts))
        scheduled_prompts = prompts[:allowed]
        skipped_prompts = prompts[allowed:]
        self.validate_plain_query_batch(scheduled_prompts)
        payloads: list[dict[str, Any]] = []
        if scheduled_prompts:
            workers = self.batch_max_workers(len(scheduled_prompts), requested_max_workers=max_workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._plain_query, prompt, model, False) for prompt in scheduled_prompts]
                payloads.extend(future.result() for future in futures)
        payloads.extend(
            self._budget_error_payload(prompt=prompt, model=model, kind="plain_query")
            for prompt in skipped_prompts
        )
        return payloads

    def run_recursive_query_batch(
        self,
        prompts: list[str],
        *,
        model: str | None = None,
        max_depth: int | None = None,
        max_workers: int | None = None,
    ) -> list[dict[str, Any]]:
        if not prompts:
            return []
        allowed = self._reserve_subcall_budget(len(prompts))
        scheduled_prompts = prompts[:allowed]
        skipped_prompts = prompts[allowed:]
        self.validate_recursive_query_batch(scheduled_prompts, max_depth=max_depth)
        payloads: list[dict[str, Any]] = []
        if scheduled_prompts:
            workers = self.batch_max_workers(len(scheduled_prompts), requested_max_workers=max_workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._recursive_query, prompt, model, max_depth, False) for prompt in scheduled_prompts]
                payloads.extend(future.result() for future in futures)
        payloads.extend(
            self._budget_error_payload(prompt=prompt, model=model, kind="recursive_query")
            for prompt in skipped_prompts
        )
        return payloads

    def _plain_query(self, prompt: str, model: str | None = None, consume_budget: bool = True) -> dict[str, Any]:
        start_time = time.perf_counter()
        messages = self._plain_query_messages(prompt)
        self._validate_subcall_messages(kind="llm_query", message_batches=[messages])
        if consume_budget and not self._try_consume_subcall_budget():
            return self._budget_error_payload(prompt=prompt, model=model, kind="plain_query")
        depth = max(1, self._current_call_depth())
        self._record_subcall(kind="plain_query", depth=depth)
        session = self.plain_llm_session
        text, payload = session.generate(
            messages=messages,
            max_tokens=self.config.subcall_max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        self._append_segment(
            payload=payload,
            depth=depth,
            turn_index=-1,
            kind="plain_query",
            train_scope="llm_subcall",
            is_trainable_rlm_turn=False,
            response_source="llm_subcall",
            response_text=text,
            messages=messages,
        )
        return {
            "prompt": prompt,
            "model": model or session.model_name,
            "response": text,
            "final_answer": extract_final_answer(text) or text,
            "depth": depth,
            "kind": "plain_query",
            "execution_time": time.perf_counter() - start_time,
        }

    def _recursive_query(
        self,
        prompt: str,
        model: str | None = None,
        max_depth: int | None = None,
        consume_budget: bool = True,
    ) -> dict[str, Any]:
        child_depth = self._current_call_depth() + 1
        branch_max_depth = self._current_branch_max_depth()
        if max_depth is None:
            effective_max_depth = branch_max_depth
        else:
            effective_max_depth = min(branch_max_depth, child_depth + max(0, int(max_depth)))

        if child_depth > effective_max_depth:
            return self._plain_query(prompt, model, consume_budget=consume_budget)

        self._validate_subcall_messages(
            kind="rlm_query",
            message_batches=[
                self._recursive_query_messages(
                    prompt,
                    child_depth=child_depth,
                    effective_max_depth=effective_max_depth,
                )
            ],
        )
        if consume_budget and not self._try_consume_subcall_budget():
            return self._budget_error_payload(prompt=prompt, model=model, kind="recursive_query")
        self._record_subcall(kind="recursive_query", depth=child_depth)

        result = self.run_call(
            prompt=prompt,
            depth=child_depth,
            max_depth=effective_max_depth,
            context_payload=None,
            parent_call_id=self._current_call_id(),
        )
        return {
            "prompt": prompt,
            "model": model or self.session.model_name,
            "response": result["final_answer"] or result["response_text"],
            "final_answer": result["final_answer"],
            "depth": child_depth,
            "kind": "recursive_query",
            "call_id": result["call_id"],
            "parent_call_id": result["parent_call_id"],
            "trace": result["trace"],
            "execution_time": float(result.get("execution_time", 0.0)),
        }

    def run_call(
        self,
        *,
        prompt: str,
        depth: int,
        max_depth: int,
        context_payload: str | None,
        parent_call_id: int | None = None,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        call_id = self._next_call_id()
        repl_context = prompt if context_payload is None else context_payload
        trace = make_call_trace(call_id=call_id, depth=depth, prompt=prompt)
        repl = create_repl(
            backend=self.config.repl_backend,
            backend_kwargs=self.config.repl_backend_kwargs,
            context_payload=repl_context,
            llm_query_fn=self._plain_query,
            rlm_query_fn=self._recursive_query,
            llm_query_batch_fn=lambda prompts, model, max_workers: self.run_plain_query_batch(
                prompts,
                model=model,
                max_workers=max_workers,
            ),
            rlm_query_batch_fn=lambda prompts, model, child_max_depth, max_workers: self.run_recursive_query_batch(
                prompts,
                model=model,
                max_depth=child_max_depth,
                max_workers=max_workers,
            ),
            repl_timeout_seconds=self.config.repl_timeout_seconds,
            repl_fast_timeout_seconds=self.config.repl_fast_timeout_seconds,
        )
        message_history = self._recursive_initial_messages(
            prompt=prompt,
            depth=depth,
            max_depth=max_depth,
            context_payload=context_payload,
        )
        final_answer: str | None = None
        response_text = ""

        previous_context = self._set_thread_context(
            call_id=call_id,
            parent_call_id=parent_call_id,
            depth=depth,
            max_depth=max_depth,
        )
        try:
            for iteration_index in range(self.config.max_iterations):
                current_prompt = message_history + [
                    build_user_prompt(
                        root_prompt=prompt,
                        iteration=iteration_index,
                        context_count=int(repl.get_context_count()),
                        history_count=int(repl.get_history_count()),
                    )
                ]
                response_text, payload = self.session.generate(
                    messages=current_prompt,
                    max_tokens=self.config.turn_max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                self._append_segment(
                    payload=payload,
                    depth=depth,
                    turn_index=iteration_index,
                    kind="recursive_turn",
                    train_scope="recursive_turn",
                    is_trainable_rlm_turn=True,
                    response_source="recursive",
                    response_text=response_text,
                    messages=current_prompt,
                    call_id=call_id,
                    parent_call_id=parent_call_id,
                )

                code_block_strs = find_code_blocks(response_text)
                code_blocks: list[CodeBlock] = []
                if code_block_strs:
                    with self._state_lock:
                        self.state["used_repl"] = True

                for code in code_block_strs:
                    execution = repl.execute_code(code)
                    code_blocks.append(CodeBlock(code=code, result=execution))
                    if final_answer is None and execution.final_answer is not None:
                        final_answer = execution.final_answer

                if final_answer is None:
                    final_answer = extract_final_answer(response_text, environment=repl)

                iteration = RLMIteration(
                    prompt=current_prompt,
                    response=response_text,
                    code_blocks=code_blocks,
                    final_answer=final_answer,
                )
                feedback_messages = [
                    message["content"]
                    for message in make_feedback_messages(
                        iteration,
                        max_chars=self.config.execution_output_char_limit,
                    )
                ]
                budget_message = self.budget_feedback_message()
                if budget_message is not None:
                    feedback_messages.append(budget_message["content"])

                append_step_trace(
                    trace,
                    assistant=response_text,
                    code_blocks=code_block_strs,
                    feedback=feedback_messages,
                    final_answer=final_answer,
                )
                self._write_live_trace(
                    event=f"recursive_step:{depth}:{iteration_index}",
                    active_trace=trace,
                )

                if final_answer is not None:
                    break

                message_history.append({"role": "assistant", "content": response_text})
                formatted_feedback = make_feedback_messages(
                    iteration,
                    max_chars=self.config.execution_output_char_limit,
                )
                if budget_message is not None:
                    formatted_feedback.append(budget_message)
                message_history.extend(formatted_feedback)

            if final_answer is None:
                finalize_prompt = message_history + [
                    {
                        "role": "user",
                        "content": self.build_finalize_message(),
                    }
                ]
                response_text, payload = self.session.generate(
                    messages=finalize_prompt,
                    max_tokens=self.config.turn_max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                self._append_segment(
                    payload=payload,
                    depth=depth,
                    turn_index=len(trace["steps"]),
                    kind="finalize_turn",
                    train_scope="finalize_turn",
                    is_trainable_rlm_turn=True,
                    response_source="recursive",
                    response_text=response_text,
                    messages=finalize_prompt,
                    call_id=call_id,
                    parent_call_id=parent_call_id,
                )
                final_answer = extract_final_answer(response_text, environment=repl) or response_text.strip()
                append_step_trace(
                    trace,
                    assistant=response_text,
                    code_blocks=[],
                    feedback=[],
                    final_answer=final_answer,
                )
                self._write_live_trace(event=f"recursive_finalize:{depth}", active_trace=trace)
        finally:
            self._restore_thread_context(previous_context)
            close = getattr(repl, "close", None)
            if callable(close):
                close()

        with self._state_lock:
            self.state["rlm_trace"].append(trace)
        self._write_live_trace(event=f"recursive_complete:{depth}")
        return {
            "call_id": call_id,
            "parent_call_id": parent_call_id,
            "response_text": response_text,
            "final_answer": final_answer,
            "trace": trace,
            "execution_time": time.perf_counter() - start_time,
        }
