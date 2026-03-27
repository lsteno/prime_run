from __future__ import annotations

import contextvars
import os
import uuid
from typing import Any, cast

import verifiers as vf

from .dataset import build_datasets
from .reward import add_metrics, build_rubric
from .trace import build_recursive_trace, segment_from_trajectory_step

try:
    from verifiers.envs.experimental.rlm_env import RLMEnv
    from verifiers.types import Messages, State, TrajectoryStep, UserMessage
except ImportError as exc:  # pragma: no cover - exercised only with older verifiers
    RLM_ENV_IMPORT_ERROR = exc
    RLMEnv = vf.MultiTurnEnv  # type: ignore[assignment]
    Messages = list[dict[str, Any]]  # type: ignore[misc,assignment]
    State = dict[str, Any]  # type: ignore[misc,assignment]
    TrajectoryStep = dict[str, Any]  # type: ignore[misc,assignment]
    UserMessage = dict[str, Any]  # type: ignore[misc,assignment]
else:
    RLM_ENV_IMPORT_ERROR = None


DEFAULT_EXECUTION_OUTPUT_CHAR_LIMIT = 4000


def build_root_system_prompt(*, max_depth: int) -> str:
    return f"""You are operating in a recursive RLVR environment inside a persistent Python REPL.

The task data is available through the filesystem context created by the environment. Solve tasks iteratively:
- inspect the filesystem and intermediate outputs through `call_python_repl`
- update `answer["content"]` as you refine your answer
- only set `answer["ready"] = True` once you are confident in the final answer

Use recursion deliberately:
- `rlm_query(prompt, max_depth=None)` delegates a sub-problem to another copy of the model
- recursive depth is capped at {max_depth}
- `llm_batch([...])` is still available for shallow parallel semantic delegation

Prefer short REPL steps over one-shot code. When recursion depth is exhausted, finish the task with the information already available.
"""


class RLMRLVREnv(RLMEnv):
    def __init__(
        self,
        *,
        max_depth: int,
        max_iterations: int,
        turn_max_tokens: int,
        subcall_max_tokens: int,
        temperature: float,
        top_p: float,
        execution_output_char_limit: int = DEFAULT_EXECUTION_OUTPUT_CHAR_LIMIT,
        **kwargs,
    ):
        if RLM_ENV_IMPORT_ERROR is not None:
            raise ImportError(
                "rlm_rlvr now depends on verifiers.envs.experimental.rlm_env. "
                "Upgrade `verifiers` to a release that includes the canonical RLMEnv."
            ) from RLM_ENV_IMPORT_ERROR

        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if turn_max_tokens < 1 or subcall_max_tokens < 1:
            raise ValueError("turn_max_tokens and subcall_max_tokens must be >= 1")
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0.0")
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")

        self.max_depth = int(max_depth)
        self.max_iterations = int(max_iterations)
        self.turn_max_tokens = int(turn_max_tokens)
        self.subcall_max_tokens = int(subcall_max_tokens)
        self.default_temperature = float(temperature)
        self.default_top_p = float(top_p)
        self.execution_output_char_limit = int(execution_output_char_limit)
        self._recursive_tool_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
            "rlm_recursive_tool_context", default=None
        )

        super().__init__(
            repl_language="python",
            max_turns=self.max_iterations + 1,
            sub_llm_max_turns=self.max_iterations,
            include_sub_llm_in_trajectory=True,
            max_output_length=self.execution_output_char_limit,
            system_prompt=build_root_system_prompt(max_depth=self.max_depth),
            root_tools=[self.rlm_query],
            sub_tools=[self.rlm_query],
            **kwargs,
        )

    async def setup_state(self, state: State, **kwargs) -> State:
        sampling_args = dict(state.get("sampling_args") or {})
        sampling_args.setdefault("temperature", self.default_temperature)
        sampling_args.setdefault("top_p", self.default_top_p)
        sampling_args.setdefault("max_tokens", self.turn_max_tokens)
        state["sampling_args"] = sampling_args

        state["used_repl"] = False
        state["used_recursion"] = False
        state["num_subcalls"] = 0
        state["max_depth_reached"] = 0
        state["rlm_segments"] = []
        state["rlm_trace"] = []
        state["_rlm_segment_counter"] = 0
        state["_rlm_request_depths"] = {}
        state["_rlm_recursive_calls"] = []
        state["_rlm_recursive_call_counter"] = 0
        return await super().setup_state(state, **kwargs)

    async def _call_sub_llm_api(
        self,
        state: State,
        client,
        model: str,
        messages: Messages,
        tools: list[vf.Tool] | None = None,
    ):
        adjusted_state = cast(State, dict(state))
        sampling_args = dict(state.get("sampling_args") or {})
        sampling_args.setdefault("temperature", self.default_temperature)
        sampling_args.setdefault("top_p", self.default_top_p)
        current_max_tokens = sampling_args.get("max_tokens")
        if current_max_tokens is None:
            sampling_args["max_tokens"] = self.subcall_max_tokens
        else:
            sampling_args["max_tokens"] = min(int(current_max_tokens), self.subcall_max_tokens)
        adjusted_state["sampling_args"] = sampling_args
        return await super()._call_sub_llm_api(adjusted_state, client, model, messages, tools)

    async def _run_sub_llm(self, state: State, client, model: str, messages: Messages):
        inherited_context = self._recursive_tool_context.get()
        current_context = dict(inherited_context or {})
        current_context.setdefault("current_depth", 0)
        current_context.setdefault("remaining_depth", self.max_depth)
        current_context.setdefault("parent_turn", self._main_turn_count(state))
        current_context["state"] = state
        current_context["client"] = client
        current_context["model"] = model

        token = self._recursive_tool_context.set(current_context)
        try:
            return await super()._run_sub_llm(state, client, model, messages)
        finally:
            self._recursive_tool_context.reset(token)

    async def rlm_query(self, prompt: str, max_depth: int | None = None) -> str:
        """
        Delegate a sub-problem to another recursive model call.

        Args:
            prompt: The sub-task to solve.
            max_depth: Optional remaining recursion depth budget for this branch.

        Returns:
            The final answer produced by the recursive sub-call.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if max_depth is not None and int(max_depth) < 0:
            raise ValueError("max_depth must be >= 0 when provided")

        context = self._recursive_tool_context.get() or self._root_tool_context_var.get()
        if context is None:
            raise RuntimeError("rlm_query called outside of an RLM request context")

        state = cast(State, context["state"])
        client = context.get("client")
        model = context.get("sub_model") or context.get("model")
        if client is None or not model:
            raise RuntimeError("RLM client/model context is unavailable")

        parent_turn = int(context.get("parent_turn", self._main_turn_count(state)))
        current_depth = int(context.get("current_depth", 0))
        remaining_depth = int(context.get("remaining_depth", self.max_depth))
        child_depth = current_depth + 1
        if child_depth > self.max_depth:
            return (
                f"Max recursion depth reached ({self.max_depth}). "
                "Solve the task without another recursive call."
            )

        child_remaining_depth = max(0, remaining_depth - 1)
        if max_depth is not None:
            child_remaining_depth = min(child_remaining_depth, int(max_depth))

        return await self._run_recursive_query(
            prompt=prompt.strip(),
            state=state,
            client=client,
            model=str(model),
            parent_turn=parent_turn,
            depth=child_depth,
            remaining_depth=child_remaining_depth,
        )

    async def _run_recursive_query(
        self,
        *,
        prompt: str,
        state: State,
        client,
        model: str,
        parent_turn: int,
        depth: int,
        remaining_depth: int,
    ) -> str:
        batch_id = f"rq{uuid.uuid4().hex[:8]}"
        request_id = uuid.uuid4().hex[:8]
        request_key = f"{batch_id}:{request_id}"

        state["used_recursion"] = True
        state["num_subcalls"] = int(state.get("num_subcalls", 0)) + 1
        state["max_depth_reached"] = max(int(state.get("max_depth_reached", 0)), depth)
        cast(dict[str, int], state["_rlm_request_depths"])[request_key] = depth

        call_id = int(state.get("_rlm_recursive_call_counter", 0))
        state["_rlm_recursive_call_counter"] = call_id + 1
        call_trace: dict[str, Any] = {
            "call_id": call_id,
            "depth": depth,
            "prompt": prompt,
            "parent_turn": parent_turn,
            "remaining_depth": remaining_depth,
            "batch_id": batch_id,
            "request_id": request_id,
        }

        token = self._recursive_tool_context.set(
            {
                "state": state,
                "client": client,
                "model": model,
                "parent_turn": parent_turn,
                "current_depth": depth,
                "remaining_depth": remaining_depth,
            }
        )
        try:
            response_dict = await self._run_sub_llm_request(
                state_ref=state,
                client=client,
                sub_model=model,
                messages=[UserMessage(content=prompt)],
                batch_id=batch_id,
                request_id=request_id,
                parent_turn=parent_turn,
            )
        finally:
            self._recursive_tool_context.reset(token)

        message = response_dict.get("choices", [{}])[0].get("message", {})
        answer = str(message.get("content", ""))
        metadata = dict(response_dict.get("_rlm_metadata", {}))
        call_trace["response"] = answer
        call_trace["metadata"] = metadata
        cast(list[dict[str, Any]], state["_rlm_recursive_calls"]).append(call_trace)
        return answer

    async def add_trajectory_step(self, state: State, trajectory_step: TrajectoryStep):
        await super().add_trajectory_step(state, trajectory_step)

        extras = trajectory_step.setdefault("extras", {})
        state["used_repl"] = bool(state.get("repl_call_count", 0))

        request_key = None
        if extras.get("is_sub_llm_call"):
            request_key = f"{extras.get('batch_id', '')}:{extras.get('request_id', '')}"

        request_depths = cast(dict[str, int], state.get("_rlm_request_depths", {}))
        depth = request_depths.get(request_key or "", 1 if extras.get("is_sub_llm_call") else 0)
        kind = "recursive_turn" if request_key in request_depths else ("sub_llm_turn" if extras.get("is_sub_llm_call") else "root_turn")

        segment = segment_from_trajectory_step(
            trajectory_step,
            order=int(state.get("_rlm_segment_counter", 0)),
            depth=depth,
            kind=kind,
            default_temperature=float((state.get("sampling_args") or {}).get("temperature", self.default_temperature)),
        )
        if segment is None:
            return

        extras["rlm_segment_order"] = segment["order"]
        state["_rlm_segment_counter"] = int(state.get("_rlm_segment_counter", 0)) + 1
        cast(list[dict[str, Any]], state["rlm_segments"]).append(segment)

    async def post_rollout(self, state: State):
        state["used_repl"] = bool(state.get("repl_call_count", 0))
        state["rlm_trace"] = build_recursive_trace(cast(list[dict[str, Any]], state.get("_rlm_recursive_calls", [])))
        await super().post_rollout(state)


def load_environment(
    data_paths: list[str] | None = None,
    eval_data_paths: list[str] | None = None,
    dataset_id: str | None = "lsteno/BEEG-agents",
    dataset_train_split: str = "train",
    dataset_eval_split: str = "eval",
    dataset_config: str | None = None,
    dataset_revision: str | None = None,
    seed: int = 42,
    max_examples: int = -1,
    max_eval_examples: int = -1,
    max_iterations: int = 4,
    max_depth: int = 2,
    turn_max_tokens: int = 192,
    subcall_max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    judge_model: str = "z-ai/glm-4.7-flash",
    judge_base_url: str = "https://openrouter.ai/api/v1",
    judge_api_key_var: str = "OPENROUTER_API_KEY",
    judge_http_referer: str | None = None,
    judge_app_title: str | None = None,
    code_execution_timeout: int = 120,
    execution_output_char_limit: int = DEFAULT_EXECUTION_OUTPUT_CHAR_LIMIT,
) -> vf.Environment:
    vf.ensure_keys([judge_api_key_var])

    judge_default_headers = {
        key: value
        for key, value in {
            "HTTP-Referer": judge_http_referer,
            "X-Title": judge_app_title,
        }.items()
        if value
    }

    train_builder, eval_builder = build_datasets(
        data_paths=data_paths,
        eval_data_paths=eval_data_paths,
        dataset_id=dataset_id,
        dataset_train_split=dataset_train_split,
        dataset_eval_split=dataset_eval_split,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        seed=seed,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
    )
    reward_rubric = build_rubric(
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=os.environ[judge_api_key_var],
        judge_default_headers=judge_default_headers or None,
    )
    add_metrics(reward_rubric)

    return RLMRLVREnv(
        dataset=train_builder,
        eval_dataset=eval_builder,
        rubric=reward_rubric,
        env_id="rlm_rlvr",
        max_depth=max_depth,
        max_iterations=max_iterations,
        turn_max_tokens=turn_max_tokens,
        subcall_max_tokens=subcall_max_tokens,
        temperature=temperature,
        top_p=top_p,
        code_execution_timeout=code_execution_timeout,
        execution_output_char_limit=execution_output_char_limit,
    )
