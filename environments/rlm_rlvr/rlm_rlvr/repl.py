from __future__ import annotations

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
        del max_workers
        return self._rlm_query_batched(prompts, model, max_depth)


def create_repl(
    *,
    backend: str,
    backend_kwargs: dict[str, Any] | None,
    context_payload: str,
    llm_query_fn: Callable[[str, str | None], dict[str, Any]],
    rlm_query_fn: Callable[[str, str | None, int | None], dict[str, Any]],
) -> RecursiveLocalRepl:
    if backend == "local":
        return RecursiveLocalRepl(
            context_payload=context_payload,
            llm_query_fn=llm_query_fn,
            rlm_query_fn=rlm_query_fn,
        )

    del backend_kwargs
    raise ValueError("rlm_rlvr currently supports only local REPL execution.")
