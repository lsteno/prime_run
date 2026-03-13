from __future__ import annotations

import concurrent.futures
from typing import Any, Callable

from .external_rlm import LocalREPL, RLMChatCompletion, empty_usage_summary


class RecursiveLocalRepl(LocalREPL):
    def __init__(
        self,
        *,
        context_payload: str,
        llm_query_fn: Callable[[str, str | None], dict[str, Any]],
        rlm_query_fn: Callable[[str, str | None, int | None], dict[str, Any]],
        enable_rlm_query_batched_async: bool = True,
    ):
        self._llm_query_fn = llm_query_fn
        self._rlm_query_fn = rlm_query_fn
        super().__init__(
            lm_handler_address=None,
            context_payload=context_payload,
            enable_rlm_query_batched_async=enable_rlm_query_batched_async,
        )

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

    def _run_llm_query(self, prompt: str, model: str | None = None) -> str:
        payload = self._llm_query_fn(prompt, model)
        self._pending_llm_calls.append(self._completion_from_payload(payload, prompt))
        return str(payload.get("response", ""))

    def _run_rlm_query(self, prompt: str, model: str | None = None, max_depth: int | None = None) -> str:
        payload = self._rlm_query_fn(prompt, model, max_depth)
        self._pending_llm_calls.append(self._completion_from_payload(payload, prompt))
        return str(payload.get("response", ""))

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        try:
            return self._run_llm_query(prompt, model)
        except Exception as exc:
            return f"Error: LM query failed - {exc}"

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        return [self._llm_query(prompt, model) for prompt in prompts]

    def _rlm_query(self, prompt: str, model: str | None = None, max_depth: int | None = None) -> str:
        try:
            return self._run_rlm_query(prompt, model, max_depth)
        except Exception as exc:
            return f"Error: RLM query failed - {exc}"

    def _rlm_query_batched(
        self,
        prompts: list[str],
        model: str | None = None,
        max_depth: int | None = None,
    ) -> list[str]:
        return [self._rlm_query(prompt, model, max_depth) for prompt in prompts]

    def _rlm_query_batched_async(
        self,
        prompts: list[str],
        model: str | None = None,
        max_depth: int | None = None,
        max_workers: int | None = None,
    ) -> list[str]:
        if not prompts:
            return []

        workers = max(1, max_workers if max_workers is not None else min(32, len(prompts)))

        def run_one(prompt: str) -> tuple[bool, dict[str, Any] | str]:
            try:
                return True, self._rlm_query_fn(prompt, model, max_depth)
            except Exception as exc:
                return False, f"Error: RLM query failed - {exc}"

        ordered_results: list[tuple[bool, dict[str, Any] | str] | None] = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {executor.submit(run_one, prompt): index for index, prompt in enumerate(prompts)}
            for future in concurrent.futures.as_completed(future_to_index):
                ordered_results[future_to_index[future]] = future.result()

        outputs: list[str] = []
        for index, result in enumerate(ordered_results):
            if result is None:
                outputs.append("Error: RLM query failed - internal scheduling error")
                continue
            success, payload = result
            if not success:
                outputs.append(str(payload))
                continue
            assert isinstance(payload, dict)
            self._pending_llm_calls.append(self._completion_from_payload(payload, prompts[index]))
            outputs.append(str(payload.get("response", "")))
        return outputs