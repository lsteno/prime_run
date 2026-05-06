from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Any

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .external_rlm import CodeBlock, QueryMetadata, RLMIteration, build_system_prompt, build_user_prompt, find_code_blocks, find_final_answer, make_feedback_messages
from .live_trace import write_live_trace
from .prompt_variants import DEFAULT_PROMPT_VARIANT
from .repl import create_repl
from .trace import append_step_trace, make_call_trace, make_segment, prompt_provenance


class SubcallPromptTooLargeError(ValueError):
    pass


@dataclass
class RuntimeConfig:
    max_depth: int = 2
    max_iterations: int = 4
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
    repl_backend: str = "local"
    repl_backend_kwargs: dict[str, Any] | None = None
    prompt_variant: str = DEFAULT_PROMPT_VARIANT
    live_trace_dir: str | None = "outputs/rlm_rlvr/live_traces"
    subcall_prompt_limit_ratio: float = 0.85


@dataclass
class TokenPayload:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]
    completion_mask: list[bool]


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
    ):
        self.model_name = model_name
        self.max_prompt_tokens = max_prompt_tokens
        self.enable_vllm_extra_body = enable_vllm_extra_body
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            default_headers=default_headers,
        )
        tokenizer_key = tokenizer_name or model_name
        if tokenizer_key not in self._tokenizers:
            self._tokenizers[tokenizer_key] = AutoTokenizer.from_pretrained(tokenizer_key, trust_remote_code=True)
        self.tokenizer = self._tokenizers[tokenizer_key]

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
            "logprobs": True,
        }
        if self.enable_vllm_extra_body:
            request_body.update({
                "return_token_ids": True,
                "top_k": -1,
                "min_p": 0.0,
            })
        response = self.client.post(
            "chat/completions",
            body=request_body,
            cast_to=ChatCompletion,
        )
        assert response.choices is not None and len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message is not None
        text = (choice.message.content or "").strip()
        completion_ids = self._coerce_token_ids(
            getattr(choice, "token_ids", None),
            fallback=self.tokenizer.encode(text, add_special_tokens=False),
        )
        prompt_ids = self.render_prompt_ids(messages, add_generation_prompt=True)
        prompt_token_ids = self._coerce_token_ids(getattr(response, "prompt_token_ids", None), fallback=prompt_ids)
        completion_logprobs = self._coerce_logprobs(getattr(choice, "logprobs", None), completion_len=len(completion_ids))
        payload = TokenPayload(
            prompt_ids=prompt_token_ids,
            completion_ids=completion_ids,
            completion_logprobs=completion_logprobs,
            completion_mask=[True] * len(completion_ids),
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
        return "Provide only the final answer now. No explanation. Use FINAL(...) or FINAL_VAR(...)."

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
            temperature=float(self.state.get("sampling_temperature", self.config.temperature)),
            response_text=response_text,
            prompt_fingerprint=provenance["prompt_fingerprint"],
            prompt_message_count=provenance["prompt_message_count"],
            prompt_char_count=provenance["prompt_char_count"],
        )
        with self._state_lock:
            self.state["rlm_segments"].append(segment)
            prompt_token_count = float(len(payload.prompt_ids))
            completion_token_count = float(len(payload.completion_ids))
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
        self.validate_plain_query_batch(prompts)
        if not prompts:
            return []
        workers = self.batch_max_workers(len(prompts), requested_max_workers=max_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._plain_query, prompt, model) for prompt in prompts]
            return [future.result() for future in futures]

    def run_recursive_query_batch(
        self,
        prompts: list[str],
        *,
        model: str | None = None,
        max_depth: int | None = None,
        max_workers: int | None = None,
    ) -> list[dict[str, Any]]:
        self.validate_recursive_query_batch(prompts, max_depth=max_depth)
        if not prompts:
            return []
        workers = self.batch_max_workers(len(prompts), requested_max_workers=max_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._recursive_query, prompt, model, max_depth) for prompt in prompts]
            return [future.result() for future in futures]

    def _plain_query(self, prompt: str, model: str | None = None) -> dict[str, Any]:
        start_time = time.perf_counter()
        messages = self._plain_query_messages(prompt)
        self._validate_subcall_messages(kind="llm_query", message_batches=[messages])
        depth = max(1, self._current_call_depth())
        self._record_subcall(kind="plain_query", depth=depth)
        text, payload = self.session.generate(
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
            "model": model or self.session.model_name,
            "response": text,
            "final_answer": find_final_answer(text) or text,
            "depth": depth,
            "kind": "plain_query",
            "execution_time": time.perf_counter() - start_time,
        }

    def _recursive_query(
        self,
        prompt: str,
        model: str | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        child_depth = self._current_call_depth() + 1
        branch_max_depth = self._current_branch_max_depth()
        if max_depth is None:
            effective_max_depth = branch_max_depth
        else:
            effective_max_depth = min(branch_max_depth, child_depth + max(0, int(max_depth)))

        if child_depth > effective_max_depth:
            return self._plain_query(prompt, model)

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
                    final_answer = find_final_answer(response_text, environment=repl)

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
                message_history.extend(
                    make_feedback_messages(
                        iteration,
                        max_chars=self.config.execution_output_char_limit,
                    )
                )

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
                final_answer = find_final_answer(response_text, environment=repl) or response_text.strip()
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
