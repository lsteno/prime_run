from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .external_rlm import CodeBlock, QueryMetadata, RLMIteration, build_system_prompt, build_user_prompt, find_code_blocks, find_final_answer, make_feedback_messages
from .repl import create_repl
from .trace import append_step_trace, make_call_trace, make_segment


@dataclass
class RuntimeConfig:
    max_depth: int = 2
    max_iterations: int = 4
    turn_max_tokens: int = 192
    subcall_max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    execution_output_char_limit: int = 4000
    tokenizer_name: str | None = None
    inference_mode: str = "hosted"
    inference_base_url: str | None = None
    inference_api_key: str | None = None
    repl_backend: str = "local"
    repl_backend_kwargs: dict[str, Any] | None = None


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
    ):
        self.model_name = model_name
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
        prompt_ids = self.render_prompt_ids(messages, add_generation_prompt=True)
        response = self.client.post(
            "/chat/completions/tokens",
            body={
                "model": self.model_name,
                "messages": messages,
                "tokens": prompt_ids,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "logprobs": True,
                "extra_body": {
                    "return_token_ids": True,
                    "top_k": -1,
                    "min_p": 0.0,
                },
            },
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
    def __init__(self, state: dict[str, Any], config: RuntimeConfig):
        self.state = state
        self.config = config

    @property
    def session(self) -> SyncInferenceSession:
        return self.state["_sync_session"]

    def build_finalize_message(self) -> str:
        return "Provide the final answer now. Use FINAL(...) or FINAL_VAR(...)."

    def _next_segment_order(self) -> int:
        order = int(self.state["rlm_segment_counter"])
        self.state["rlm_segment_counter"] = order + 1
        return order

    def _next_call_id(self) -> int:
        call_id = int(self.state["rlm_call_counter"])
        self.state["rlm_call_counter"] = call_id + 1
        return call_id

    def _append_segment(self, *, payload: TokenPayload, depth: int, kind: str, response_text: str) -> None:
        segment = make_segment(
            order=self._next_segment_order(),
            depth=depth,
            kind=kind,
            prompt_ids=payload.prompt_ids,
            completion_ids=payload.completion_ids,
            completion_logprobs=payload.completion_logprobs,
            completion_mask=payload.completion_mask,
            temperature=float(self.state.get("sampling_temperature", self.config.temperature)),
            response_text=response_text,
        )
        self.state["rlm_segments"].append(segment)
        self.state["total_model_tokens"] += float(sum(payload.completion_mask))
        self.state["max_depth_reached"] = max(int(self.state["max_depth_reached"]), depth)

    def _plain_query(self, prompt: str, model: str | None = None) -> dict[str, Any]:
        start_time = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        text, payload = self.session.generate(
            messages=messages,
            max_tokens=self.config.subcall_max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        self._append_segment(
            payload=payload,
            depth=int(self.state["current_call_depth"]),
            kind="plain_query",
            response_text=text,
        )
        return {
            "prompt": prompt,
            "model": model or self.session.model_name,
            "response": text,
            "final_answer": find_final_answer(text) or text,
            "depth": int(self.state["current_call_depth"]),
            "kind": "plain_query",
            "execution_time": time.perf_counter() - start_time,
        }

    def _recursive_query(
        self,
        prompt: str,
        model: str | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        child_depth = int(self.state["current_call_depth"]) + 1
        branch_max_depth = int(self.state.get("current_branch_max_depth", self.config.max_depth))
        if max_depth is None:
            effective_max_depth = branch_max_depth
        else:
            effective_max_depth = min(branch_max_depth, child_depth + max(0, int(max_depth)))
        self.state["used_recursion"] = True
        self.state["num_subcalls"] += 1

        if child_depth > effective_max_depth:
            return self._plain_query(prompt, model)

        result = self.run_call(prompt=prompt, depth=child_depth, max_depth=effective_max_depth, context_payload=None)
        return {
            "prompt": prompt,
            "model": model or self.session.model_name,
            "response": result["final_answer"] or result["response_text"],
            "final_answer": result["final_answer"],
            "depth": child_depth,
            "kind": "recursive_query",
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
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        call_id = self._next_call_id()
        repl_context = prompt if context_payload is None else context_payload
        context_metadata = QueryMetadata(repl_context)
        trace = make_call_trace(call_id=call_id, depth=depth, prompt=prompt)
        repl = create_repl(
            backend=self.config.repl_backend,
            backend_kwargs=self.config.repl_backend_kwargs,
            context_payload=repl_context,
            llm_query_fn=self._plain_query,
            rlm_query_fn=self._recursive_query,
        )
        message_history: list[dict[str, str]] = [
            {"role": "system", "content": build_system_prompt(depth=depth, max_depth=max_depth)},
            {
                "role": "user",
                "content": (
                    f"Your context is a {context_metadata.context_type} with "
                    f"{context_metadata.context_total_length} total characters, and is broken up into chunks "
                    f"of char lengths: {context_metadata.context_lengths}."
                ),
            },
        ]
        final_answer: str | None = None
        response_text = ""

        previous_depth = int(self.state["current_call_depth"])
        previous_branch_max_depth = int(self.state.get("current_branch_max_depth", self.config.max_depth))
        self.state["current_call_depth"] = depth
        self.state["current_branch_max_depth"] = max_depth
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
                    kind="recursive_turn",
                    response_text=response_text,
                )

                code_block_strs = find_code_blocks(response_text)
                code_blocks: list[CodeBlock] = []
                if code_block_strs:
                    self.state["used_repl"] = True

                for code in code_block_strs:
                    execution = repl.execute_code(code)
                    code_blocks.append(CodeBlock(code=code, result=execution))
                    for child_call in execution.rlm_calls:
                        metadata = child_call.metadata or {}
                        if metadata.get("kind") == "recursive_query":
                            self.state["used_recursion"] = True
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
                        "role": "assistant",
                        "content": "Please provide a final answer to the user's question based on the information provided.",
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
                    kind="finalize_turn",
                    response_text=response_text,
                )
                final_answer = find_final_answer(response_text, environment=repl) or response_text.strip()
                append_step_trace(
                    trace,
                    assistant=response_text,
                    code_blocks=[],
                    feedback=[],
                    final_answer=final_answer,
                )
        finally:
            self.state["current_call_depth"] = previous_depth
            self.state["current_branch_max_depth"] = previous_branch_max_depth
            close = getattr(repl, "close", None)
            if callable(close):
                close()

        self.state["rlm_trace"].append(trace)
        return {
            "response_text": response_text,
            "final_answer": final_answer,
            "trace": trace,
            "execution_time": time.perf_counter() - start_time,
        }
