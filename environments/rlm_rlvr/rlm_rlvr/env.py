from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
import verifiers as vf

from .dataset import build_datasets
from .parsing import extract_code_blocks, extract_final_answer, render_execution_output
from .reward import build_rubric
from .runtime import RecursiveRuntime, RuntimeConfig, SyncInferenceSession
from .trace import make_segment


class RLMRLVREnv(vf.MultiTurnEnv):
    def __init__(self, *, runtime_config: RuntimeConfig, **kwargs):
        super().__init__(max_turns=runtime_config.max_iterations + 1, interleaved_rollouts=True, **kwargs)
        self.runtime_config = runtime_config

    async def setup_state(self, state: vf.State) -> vf.State:
        client = state["client"]
        assert isinstance(client, AsyncOpenAI)

        base_url = str(client.base_url)
        api_key = getattr(client, "api_key", None) or "EMPTY"
        default_headers = dict(getattr(client, "default_headers", {}) or {})
        model_name = str(state["model"])

        state["efficiency_penalty_coef"] = getattr(self, "efficiency_penalty_coef", 0.02)
        state["used_repl"] = False
        state["used_recursion"] = False
        state["max_depth_reached"] = 0
        state["num_subcalls"] = 0
        state["total_model_tokens"] = 0.0
        state["total_env_tokens"] = 0.0
        state["rlm_segments"] = []
        state["rlm_trace"] = []
        state["rlm_segment_counter"] = 0
        state["rlm_call_counter"] = 0
        state["current_call_depth"] = 0
        state["final_answer"] = None
        state["sampling_temperature"] = float((state.get("sampling_args") or {}).get("temperature", self.runtime_config.temperature))

        state["_sync_session"] = SyncInferenceSession(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
            model_name=model_name,
            tokenizer_name=self.runtime_config.tokenizer_name,
        )
        state["_runtime"] = RecursiveRuntime(state, self.runtime_config)
        state["_root_context"] = (state.get("info") or {}).get("context", "")
        return await super().setup_state(state)

    async def add_trajectory_step(self, state: vf.State, trajectory_step: vf.TrajectoryStep):
        await super().add_trajectory_step(state, trajectory_step)
        tokens = trajectory_step.get("tokens")
        if tokens is None:
            return

        temperature = float((state.get("sampling_args") or {}).get("temperature", self.runtime_config.temperature))
        segment = make_segment(
            order=int(state["rlm_segment_counter"]),
            depth=0,
            kind="root_turn",
            prompt_ids=list(tokens["prompt_ids"]),
            completion_ids=list(tokens["completion_ids"]),
            completion_logprobs=[float(value) for value in tokens["completion_logprobs"]],
            completion_mask=[bool(value) for value in tokens["completion_mask"]],
            temperature=temperature,
            response_text=trajectory_step["completion"][-1].get("content", ""),
        )
        trajectory_step["extras"]["rlm_segment_order"] = segment["order"]
        state["rlm_segment_counter"] += 1
        state["rlm_segments"].append(segment)
        state["total_model_tokens"] += float(sum(segment["completion_mask"]))

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        del kwargs
        runtime: RecursiveRuntime = state["_runtime"]
        assistant_text = str(messages[-1].get("content", "")) if messages else ""

        final_answer = extract_final_answer(assistant_text)
        if final_answer is not None:
            state["final_answer"] = final_answer
            state["final_env_response"] = []
            return []

        code_blocks = extract_code_blocks(assistant_text)
        feedback: list[str] = []
        if code_blocks:
            state["used_repl"] = True

        root_repl = runtime.run_call
        if not hasattr(state, "_root_repl_initialized"):
            state["_root_repl_initialized"] = True

        # Reuse the same recursive runtime for root-side code execution by routing through a dedicated REPL call.
        # The root call itself is already recorded as trajectory steps; here we only execute the emitted code.
        from .repl import RecursiveLocalRepl

        if "_root_repl" not in state:
            state["_root_repl"] = RecursiveLocalRepl(
                context_payload=state.get("_root_context", ""),
                llm_query_fn=runtime._plain_query,
                rlm_query_fn=runtime._recursive_query,
            )

        repl: RecursiveLocalRepl = state["_root_repl"]
        for code in code_blocks:
            execution = repl.execute(code)
            for child_call in execution.child_calls:
                if child_call.get("kind") == "recursive_query":
                    state["used_recursion"] = True
            if execution.final_answer is not None and state.get("final_answer") is None:
                state["final_answer"] = execution.final_answer
            rendered = render_execution_output(execution.stdout, execution.stderr, execution.final_answer)
            if len(rendered) > self.runtime_config.execution_output_char_limit:
                rendered = rendered[: self.runtime_config.execution_output_char_limit] + "\n... [truncated]"
            feedback.append(rendered)

        if state.get("final_answer") is not None:
            state["final_env_response"] = []
            return []

        session = state["_sync_session"]
        if feedback:
            feedback_text = "\n\n".join(feedback + [runtime.build_continue_message()])
        elif len(state["trajectory"]) >= self.runtime_config.max_iterations:
            feedback_text = runtime.build_finalize_message()
        else:
            feedback_text = runtime.build_continue_message()

        state["total_env_tokens"] += float(session.count_text_tokens(feedback_text))
        return [{"role": "user", "content": feedback_text}]


def load_environment(
    data_paths: list[str] | None = None,
    eval_data_paths: list[str] | None = None,
    seed: int = 42,
    eval_fraction: float = 0.05,
    eval_size: int | None = None,
    max_examples: int = -1,
    max_eval_examples: int = -1,
    max_iterations: int = 4,
    max_depth: int = 2,
    turn_max_tokens: int = 192,
    subcall_max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    tokenizer_name: str | None = None,
    efficiency_penalty_coef: float = 0.02,
) -> vf.Environment:
    train_builder, eval_builder = build_datasets(
        data_paths=data_paths,
        eval_data_paths=eval_data_paths,
        seed=seed,
        eval_fraction=eval_fraction,
        eval_size=eval_size,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
    )
    runtime_config = RuntimeConfig(
        max_depth=max_depth,
        max_iterations=max_iterations,
        turn_max_tokens=turn_max_tokens,
        subcall_max_tokens=subcall_max_tokens,
        temperature=temperature,
        top_p=top_p,
        tokenizer_name=tokenizer_name,
    )
    system_prompt = (
        "You are a recursive problem-solving model. Use ```repl``` blocks when computation helps. "
        "The task context is available as `context`. Use rlm_query(prompt) to recurse with the same local model. "
        "Finish with FINAL(answer) or FINAL_VAR(variable_name)."
    )
    return RLMRLVREnv(
        dataset=train_builder,
        eval_dataset=eval_builder,
        system_prompt=system_prompt,
        rubric=build_rubric(),
        runtime_config=runtime_config,
        efficiency_penalty_coef=efficiency_penalty_coef,
        env_id="rlm_rlvr",
    )