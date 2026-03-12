from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .parsing import extract_code_blocks, extract_final_answer, render_execution_output
from .repl import RecursiveLocalRepl
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
        completion_ids = list(getattr(choice, "token_ids"))
        prompt_token_ids = list(getattr(response, "prompt_token_ids"))
        logprobs = choice.logprobs
        if hasattr(logprobs, "content") and logprobs.content is not None:
            completion_logprobs = [float(item.logprob) for item in logprobs.content]
        else:
            completion_logprobs = [float(item["logprob"]) for item in logprobs["content"]]
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

    def build_system_prompt(self, depth: int, max_depth: int) -> str:
        remaining = max(0, max_depth - depth)
        return (
            "You are solving a task with a Python REPL. "
            "Use ```repl``` blocks for computation. The variable `context` contains the task context. "
            "Use llm_query(prompt) for a plain local model call. Use rlm_query(prompt) to recurse with the same local model. "
            "Finish with FINAL(answer) or FINAL_VAR(variable_name). "
            f"Current recursion depth: {depth}. Remaining recursion budget: {remaining}."
        )

    def build_continue_message(self) -> str:
        return (
            "Continue solving the original task. Use rlm_query(...) only when decomposition is necessary. "
            "If you know the answer, emit FINAL(...)."
        )

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
        del model
        messages = [{"role": "user", "content": prompt}]
        text, payload = self.session.generate(
            messages=messages,
            max_tokens=self.config.subcall_max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        self._append_segment(payload=payload, depth=int(self.state["current_call_depth"]), kind="plain_query", response_text=text)
        return {
            "response": text,
            "final_answer": extract_final_answer(text) or text,
            "depth": int(self.state["current_call_depth"]),
            "kind": "plain_query",
        }

    def _recursive_query(
        self,
        prompt: str,
        model: str | None = None,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        del model
        child_depth = int(self.state["current_call_depth"]) + 1
        effective_max_depth = self.config.max_depth if max_depth is None else min(self.config.max_depth, max_depth)
        self.state["used_recursion"] = True
        self.state["num_subcalls"] += 1

        if child_depth > effective_max_depth:
            return self._plain_query(prompt)

        result = self.run_call(prompt=prompt, depth=child_depth, max_depth=effective_max_depth, context_payload=None)
        return {
            "response": result["final_answer"] or result["response_text"],
            "final_answer": result["final_answer"],
            "depth": child_depth,
            "kind": "recursive_query",
            "trace": result["trace"],
        }

    def run_call(
        self,
        *,
        prompt: str,
        depth: int,
        max_depth: int,
        context_payload: str | None,
    ) -> dict[str, Any]:
        call_id = self._next_call_id()
        repl_context = prompt if context_payload is None else context_payload
        trace = make_call_trace(call_id=call_id, depth=depth, prompt=prompt)
        repl = RecursiveLocalRepl(
            context_payload=repl_context,
            llm_query_fn=self._plain_query,
            rlm_query_fn=self._recursive_query,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt(depth, max_depth)},
            {"role": "user", "content": prompt},
        ]
        final_answer: str | None = None
        response_text = ""

        previous_depth = int(self.state["current_call_depth"])
        self.state["current_call_depth"] = depth
        try:
            for _ in range(self.config.max_iterations):
                response_text, payload = self.session.generate(
                    messages=messages,
                    max_tokens=self.config.turn_max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                self._append_segment(payload=payload, depth=depth, kind="recursive_turn", response_text=response_text)

                code_blocks = extract_code_blocks(response_text)
                feedback_messages: list[str] = []
                if code_blocks:
                    self.state["used_repl"] = True

                for code in code_blocks:
                    execution = repl.execute(code)
                    for child_call in execution.child_calls:
                        if child_call.get("kind") == "recursive_query":
                            self.state["used_recursion"] = True
                    if final_answer is None and execution.final_answer is not None:
                        final_answer = execution.final_answer
                    feedback = render_execution_output(execution.stdout, execution.stderr, execution.final_answer)
                    if len(feedback) > self.config.execution_output_char_limit:
                        feedback = feedback[: self.config.execution_output_char_limit] + "\n... [truncated]"
                    feedback_messages.append(feedback)

                if final_answer is None:
                    final_answer = extract_final_answer(response_text)

                append_step_trace(
                    trace,
                    assistant=response_text,
                    code_blocks=code_blocks,
                    feedback=feedback_messages,
                    final_answer=final_answer,
                )

                if final_answer is not None:
                    break

                messages.append({"role": "assistant", "content": response_text})
                if feedback_messages:
                    messages.append(
                        {
                            "role": "user",
                            "content": "\n\n".join(feedback_messages + [self.build_continue_message()]),
                        }
                    )
                else:
                    messages.append({"role": "user", "content": self.build_continue_message()})

            if final_answer is None:
                messages.append({"role": "user", "content": self.build_finalize_message()})
                response_text, payload = self.session.generate(
                    messages=messages,
                    max_tokens=self.config.turn_max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                self._append_segment(payload=payload, depth=depth, kind="finalize_turn", response_text=response_text)
                final_answer = extract_final_answer(response_text) or response_text.strip()
                append_step_trace(
                    trace,
                    assistant=response_text,
                    code_blocks=[],
                    feedback=[],
                    final_answer=final_answer,
                )
        finally:
            self.state["current_call_depth"] = previous_depth

        self.state["rlm_trace"].append(trace)
        return {
            "response_text": response_text,
            "final_answer": final_answer,
            "trace": trace,
        }