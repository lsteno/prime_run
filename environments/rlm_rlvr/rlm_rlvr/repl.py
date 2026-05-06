from __future__ import annotations

import ast
import re
import signal
import threading
from typing import Any, Callable

from rlm.core.types import REPLResult

from .external_rlm import LocalREPL, RLMChatCompletion, empty_usage_summary


SUBCALL_TOOL_NAMES = {
    "llm_query",
    "llm_query_batched",
    "rlm_query",
    "rlm_query_batched",
}


class _ReplExecutionTimeout(TimeoutError):
    pass


def code_uses_subcalls(code: str) -> bool:
    """Return whether a generated REPL block calls an LLM/RLM subcall helper."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return any(re.search(rf"\b{name}\s*\(", code) for name in SUBCALL_TOOL_NAMES)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in SUBCALL_TOOL_NAMES:
            return True
    return False


class RecursiveLocalRepl(LocalREPL):
    def __init__(
        self,
        *,
        context_payload: str,
        llm_query_fn: Callable[[str, str | None], dict[str, Any]],
        rlm_query_fn: Callable[[str, str | None, int | None], dict[str, Any]],
        llm_query_batch_fn: Callable[[list[str], str | None, int | None], list[dict[str, Any]]] | None = None,
        rlm_query_batch_fn: Callable[[list[str], str | None, int | None, int | None], list[dict[str, Any]]] | None = None,
        repl_timeout_seconds: float | None = None,
        repl_fast_timeout_seconds: float | None = None,
    ):
        self._llm_query_fn = llm_query_fn
        self._rlm_query_fn = rlm_query_fn
        self._llm_query_batch_fn = llm_query_batch_fn
        self._rlm_query_batch_fn = rlm_query_batch_fn
        self._pending_call_lock = threading.Lock()
        self.repl_timeout_seconds = repl_timeout_seconds
        self.repl_fast_timeout_seconds = repl_fast_timeout_seconds
        super().__init__(
            lm_handler_address=None,
            context_payload=context_payload,
            enable_rlm_query_batched_async=True,
        )

    def _timeout_for_code(self, code: str) -> float | None:
        if self.repl_timeout_seconds is None:
            return None
        if code_uses_subcalls(code):
            return self.repl_timeout_seconds
        return self.repl_fast_timeout_seconds or self.repl_timeout_seconds

    def execute_code(self, code: str) -> REPLResult:
        timeout = self._timeout_for_code(code)
        if timeout is None:
            return super().execute_code(code)
        if threading.current_thread() is not threading.main_thread():
            return super().execute_code(code)

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)

        def _raise_timeout(signum, frame):
            del signum, frame
            raise _ReplExecutionTimeout(f"REPL execution timed out after {timeout:.3g}s")

        try:
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            return super().execute_code(code)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    def _completion_from_payload(self, payload: dict[str, Any], prompt: str) -> RLMChatCompletion:
        metadata: dict[str, Any] | None = None
        if payload.get("trace") is not None or payload.get("kind") is not None:
            metadata = {
                "kind": payload.get("kind"),
                "depth": payload.get("depth"),
                "trace": payload.get("trace"),
            }
        return RLMChatCompletion(
            root_model=str(payload.get("model") or "local-model"),
            prompt=payload.get("prompt", prompt),
            response=str(payload.get("response", "")),
            usage_summary=empty_usage_summary(),
            execution_time=float(payload.get("execution_time", 0.0)),
            metadata=metadata,
        )

    def _append_pending_completion(self, completion: RLMChatCompletion) -> None:
        with self._pending_call_lock:
            self._pending_llm_calls.append(completion)

    def _extend_pending_completions(self, completions: list[RLMChatCompletion]) -> None:
        if not completions:
            return
        with self._pending_call_lock:
            self._pending_llm_calls.extend(completions)

    def _run_llm_query(self, prompt: str, model: str | None = None) -> str:
        payload = self._llm_query_fn(prompt, model)
        self._append_pending_completion(self._completion_from_payload(payload, prompt))
        return str(payload.get("response", ""))

    def _run_rlm_query(self, prompt: str, model: str | None = None, max_depth: int | None = None) -> str:
        payload = self._rlm_query_fn(prompt, model, max_depth)
        self._append_pending_completion(self._completion_from_payload(payload, prompt))
        return str(payload.get("response", ""))

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        try:
            return self._run_llm_query(prompt, model)
        except Exception as exc:
            message = f"Error: LM query failed - {exc}"
            print(message)
            return message

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        if self._llm_query_batch_fn is None:
            return [self._llm_query(prompt, model) for prompt in prompts]
        payloads = self._llm_query_batch_fn(prompts, model, None)
        completions = [self._completion_from_payload(payload, prompt) for prompt, payload in zip(prompts, payloads, strict=False)]
        self._extend_pending_completions(completions)
        return [str(payload.get("response", "")) for payload in payloads]

    def _rlm_query(self, prompt: str, model: str | None = None, max_depth: int | None = None) -> str:
        try:
            return self._run_rlm_query(prompt, model, max_depth)
        except Exception as exc:
            message = f"Error: RLM query failed - {exc}"
            print(message)
            return message

    def _rlm_query_batched(
        self,
        prompts: list[str],
        model: str | None = None,
        max_depth: int | None = None,
    ) -> list[str]:
        if self._rlm_query_batch_fn is None:
            return [self._rlm_query(prompt, model, max_depth) for prompt in prompts]
        payloads = self._rlm_query_batch_fn(prompts, model, max_depth, None)
        completions = [self._completion_from_payload(payload, prompt) for prompt, payload in zip(prompts, payloads, strict=False)]
        self._extend_pending_completions(completions)
        return [str(payload.get("response", "")) for payload in payloads]

    def _rlm_query_batched_async(
        self,
        prompts: list[str],
        model: str | None = None,
        max_depth: int | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        if self._rlm_query_batch_fn is None:
            return self._rlm_query_batched(prompts, model, max_depth)
        payloads = self._rlm_query_batch_fn(prompts, model, max_depth, max_workers)
        completions = [self._completion_from_payload(payload, prompt) for prompt, payload in zip(prompts, payloads, strict=False)]
        self._extend_pending_completions(completions)
        return [str(payload.get("response", "")) for payload in payloads]


def create_repl(
    *,
    backend: str,
    backend_kwargs: dict[str, Any] | None,
    context_payload: str,
    llm_query_fn: Callable[[str, str | None], dict[str, Any]],
    rlm_query_fn: Callable[[str, str | None, int | None], dict[str, Any]],
    llm_query_batch_fn: Callable[[list[str], str | None, int | None], list[dict[str, Any]]] | None = None,
    rlm_query_batch_fn: Callable[[list[str], str | None, int | None, int | None], list[dict[str, Any]]] | None = None,
    repl_timeout_seconds: float | None = None,
    repl_fast_timeout_seconds: float | None = None,
) -> RecursiveLocalRepl:
    if backend == "local":
        backend_kwargs = backend_kwargs or {}
        return RecursiveLocalRepl(
            context_payload=context_payload,
            llm_query_fn=llm_query_fn,
            rlm_query_fn=rlm_query_fn,
            llm_query_batch_fn=llm_query_batch_fn,
            rlm_query_batch_fn=rlm_query_batch_fn,
            repl_timeout_seconds=backend_kwargs.get("repl_timeout_seconds", repl_timeout_seconds),
            repl_fast_timeout_seconds=backend_kwargs.get("repl_fast_timeout_seconds", repl_fast_timeout_seconds),
        )

    del backend_kwargs
    raise ValueError("rlm_rlvr currently supports only local REPL execution.")
