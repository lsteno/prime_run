from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI
import verifiers as vf

from .dataset import build_datasets
from .external_rlm import CodeBlock, RLMIteration, build_initial_messages, build_system_prompt, build_user_prompt, find_code_blocks, make_feedback_messages
from .live_trace import assign_live_trace_suffix, write_live_trace
from .parsing import extract_final_answer
from .prompt_variants import DEFAULT_PROMPT_VARIANT, PROMPT_VARIANTS
from .repl import create_repl
from .reward import add_metrics, build_rubric
from .runtime import RecursiveRuntime, RuntimeConfig, SyncInferenceSession, VertexGeminiSession
from .trace import append_step_trace, make_call_trace, make_segment, prompt_provenance


class RLMRLVREnv(vf.MultiTurnEnv):
    @staticmethod
    def _sample_metadata(info: dict[str, Any]) -> dict[str, Any]:
        metadata = info.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        selected_metadata = {
            key: metadata.get(key)
            for key in (
                "source_dataset",
                "task_group",
                "reasoning_types",
                "repo",
                "language",
                "n_docs",
                "n_wiki",
            )
            if key in metadata
        }
        return {
            "source_id": info.get("source_id"),
            "dataset_name": info.get("dataset_name"),
            "source_task": info.get("source_task"),
            "answer_type": info.get("answer_type"),
            "context_token_count": info.get("context_token_count"),
            "metadata": selected_metadata,
        }

    def _attach_debug_payload(self, state: vf.State) -> None:
        trajectory = state.get("trajectory") or []
        if not trajectory:
            return

        last_step = trajectory[-1]
        extras = last_step.setdefault("extras", {})
        extras["rlm_debug"] = {
            "used_repl": bool(state.get("used_repl", False)),
            "used_recursion": bool(state.get("used_recursion", False)),
            "used_llm_subcalls": bool(state.get("used_llm_subcalls", False)),
            "used_rlm_subcalls": bool(state.get("used_rlm_subcalls", False)),
            "max_depth_reached": int(state.get("max_depth_reached", 0)),
            "num_subcalls": int(state.get("num_subcalls", 0)),
            "num_llm_subcalls": int(state.get("num_llm_subcalls", 0)),
            "num_rlm_subcalls": int(state.get("num_rlm_subcalls", 0)),
            "subcall_budget_enabled": bool(state.get("subcall_budget_enabled", False)),
            "subcall_budget_total": int(state.get("subcall_budget_total", 0)),
            "subcall_budget_remaining": int(state.get("subcall_budget_remaining", 0)),
            "subcall_budget_exhausted": bool(state.get("subcall_budget_exhausted", False)),
            "final_answer": state.get("final_answer"),
            "used_forced_finalize_prompt": bool(state.get("used_forced_finalize_prompt", False)),
            "hit_max_turn_without_final": bool(state.get("hit_max_turn_without_final", False)),
            "missing_final": bool(state.get("missing_final", False)),
            "finalized_before_forced_prompt": bool(state.get("finalized_before_forced_prompt", False)),
            "finalized_on_forced_prompt": bool(state.get("finalized_on_forced_prompt", False)),
            "sample_metadata": self._sample_metadata(state.get("info") or {}),
            "trace": state.get("rlm_trace") or [],
            "segments": state.get("rlm_segments") or [],
        }

    def __init__(self, *, runtime_config: RuntimeConfig, **kwargs):
        # max_iterations normal turns + one forced-finalize prompt/response pair.
        super().__init__(max_turns=runtime_config.max_iterations + 2, interleaved_rollouts=True, **kwargs)
        self.runtime_config = runtime_config

    @vf.stop(priority=10)
    async def max_turns_reached(self, state: vf.State) -> bool:
        """Treat a failed forced-finalize response as the max-turn stop condition."""
        if state.get("hit_max_turn_without_final"):
            return True
        return len(state["trajectory"]) >= self.max_turns and self.max_turns > 0

    async def setup_state(self, state: vf.State) -> vf.State:
        client = state["client"]
        if not isinstance(client, AsyncOpenAI):
            client = client.client
        assert isinstance(client, AsyncOpenAI)

        base_url = self.runtime_config.inference_base_url or str(client.base_url)
        api_key = self.runtime_config.inference_api_key or getattr(client, "api_key", None) or "EMPTY"
        default_headers = dict(getattr(client, "default_headers", {}) or {})
        model_name = str(state["model"])

        state["efficiency_penalty_coef"] = getattr(self, "efficiency_penalty_coef", 0.02)
        state["efficiency_penalty_mode"] = getattr(self, "efficiency_penalty_mode", "static_per_1k")
        state["adaptive_efficiency_beta_max"] = getattr(self, "adaptive_efficiency_beta_max", 0.05)
        state["adaptive_efficiency_gamma"] = getattr(self, "adaptive_efficiency_gamma", 2.0)
        state["adaptive_efficiency_solve_rate_floor"] = getattr(self, "adaptive_efficiency_solve_rate_floor", 0.25)
        state["adaptive_efficiency_cost_basis"] = getattr(self, "adaptive_efficiency_cost_basis", "total_tokens")
        state["efficiency_penalty_applies_to"] = getattr(self, "efficiency_penalty_applies_to", "correct_only")
        state["reward_clip_min"] = getattr(self, "reward_clip_min", 0.0)
        state["reward_clip_max"] = getattr(self, "reward_clip_max", 1.0)
        state["used_repl"] = False
        state["used_recursion"] = False
        state["used_llm_subcalls"] = False
        state["used_rlm_subcalls"] = False
        state["max_depth_reached"] = 0
        state["num_subcalls"] = 0
        state["num_llm_subcalls"] = 0
        state["num_rlm_subcalls"] = 0
        state["subcall_budget_enabled"] = bool(self.runtime_config.subcall_budget_enabled)
        state["subcall_budget_total"] = int(self.runtime_config.max_total_subcalls)
        state["subcall_budget_remaining"] = int(self.runtime_config.max_total_subcalls)
        state["subcall_budget_exhausted"] = False
        state["total_model_tokens"] = 0.0
        state["total_env_tokens"] = 0.0
        state["total_prompt_tokens"] = 0.0
        state["total_completion_tokens"] = 0.0
        state["total_rollout_tokens"] = 0.0
        state["rlm_segments"] = []
        state["rlm_trace"] = []
        state["rlm_segment_counter"] = 0
        state["rlm_call_counter"] = 1
        state["current_call_depth"] = 0
        state["current_call_id"] = 0
        state["current_parent_call_id"] = None
        state["current_branch_max_depth"] = self.runtime_config.max_depth
        state["final_answer"] = None
        state["used_forced_finalize_prompt"] = False
        state["hit_max_turn_without_final"] = False
        state["missing_final"] = False
        state["finalized_before_forced_prompt"] = False
        state["finalized_on_forced_prompt"] = False
        state["prompt_variant"] = self.runtime_config.prompt_variant
        state["live_trace_dir"] = self.runtime_config.live_trace_dir
        assign_live_trace_suffix(state)
        state["sampling_temperature"] = float((state.get("sampling_args") or {}).get("temperature", self.runtime_config.temperature))

        state["_sync_session"] = SyncInferenceSession(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
            model_name=model_name,
            tokenizer_name=self.runtime_config.tokenizer_name,
            max_prompt_tokens=self.runtime_config.max_prompt_tokens,
            enable_vllm_extra_body=self.runtime_config.inference_mode == "local",
        )
        if self.runtime_config.llm_subcall_model is not None:
            if self.runtime_config.llm_subcall_provider == "vertex":
                if not self.runtime_config.llm_subcall_vertex_project:
                    raise ValueError("llm_subcall_vertex_project is required when llm_subcall_provider='vertex'")
                state["_plain_llm_session"] = VertexGeminiSession(
                    model_name=self.runtime_config.llm_subcall_model,
                    project=self.runtime_config.llm_subcall_vertex_project,
                    location=self.runtime_config.llm_subcall_vertex_location,
                    tokenizer_name=self.runtime_config.tokenizer_name or model_name,
                    max_prompt_tokens=self.runtime_config.max_prompt_tokens,
                    thinking_level=self.runtime_config.llm_subcall_thinking_level,
                    empty_response_max_attempts=self.runtime_config.llm_subcall_empty_response_max_attempts,
                    empty_response_base_retry_seconds=self.runtime_config.llm_subcall_empty_response_base_retry_seconds,
                    empty_response_max_retry_seconds=self.runtime_config.llm_subcall_empty_response_max_retry_seconds,
                )
            else:
                state["_plain_llm_session"] = SyncInferenceSession(
                    base_url=self.runtime_config.llm_subcall_base_url or base_url,
                    api_key=self.runtime_config.llm_subcall_api_key or api_key,
                    default_headers=self.runtime_config.llm_subcall_default_headers,
                    model_name=self.runtime_config.llm_subcall_model,
                    tokenizer_name=self.runtime_config.tokenizer_name,
                    max_prompt_tokens=self.runtime_config.max_prompt_tokens,
                    enable_vllm_extra_body=False,
                    request_logprobs=False,
                    retry_transient_errors=True,
                )
        else:
            state["_plain_llm_session"] = state["_sync_session"]
        state["_runtime"] = RecursiveRuntime(state, self.runtime_config)
        info = state.get("info") or {}
        state["_root_context"] = info.get("context", "")
        state["_root_trace"] = make_call_trace(
            call_id=0,
            depth=0,
            prompt=str(info.get("question", "")),
        )
        state["rlm_trace"].append(state["_root_trace"])
        state["prompt"] = build_initial_messages(
            context_payload=state.get("_root_context", ""),
            root_prompt=str(info.get("question", "")),
            prompt_variant=self.runtime_config.prompt_variant,
            max_prompt_tokens=self.runtime_config.max_prompt_tokens,
            turn_max_tokens=self.runtime_config.turn_max_tokens,
            subcall_max_tokens=self.runtime_config.subcall_max_tokens,
            subcall_budget_enabled=self.runtime_config.subcall_budget_enabled,
            max_total_subcalls=self.runtime_config.max_total_subcalls,
            max_batched_subcalls=self.runtime_config.max_batched_subcalls,
            include_budget_reminder=self.runtime_config.include_budget_reminder,
        )
        write_live_trace(state, event="setup_state")
        state["_root_repl"] = create_repl(
            backend=self.runtime_config.repl_backend,
            backend_kwargs=self.runtime_config.repl_backend_kwargs,
            context_payload=state.get("_root_context", ""),
            llm_query_fn=state["_runtime"]._plain_query,
            rlm_query_fn=state["_runtime"]._recursive_query,
            llm_query_batch_fn=lambda prompts, model, max_workers: state["_runtime"].run_plain_query_batch(
                prompts,
                model=model,
                max_workers=max_workers,
            ),
            rlm_query_batch_fn=lambda prompts, model, max_depth, max_workers: state["_runtime"].run_recursive_query_batch(
                prompts,
                model=model,
                max_depth=max_depth,
                max_workers=max_workers,
            ),
            repl_timeout_seconds=self.runtime_config.repl_timeout_seconds,
            repl_fast_timeout_seconds=self.runtime_config.repl_fast_timeout_seconds,
        )
        return await super().setup_state(state)

    async def add_trajectory_step(self, state: vf.State, trajectory_step: vf.TrajectoryStep):
        await super().add_trajectory_step(state, trajectory_step)
        tokens = trajectory_step.get("tokens")
        if tokens is None:
            return

        temperature = float((state.get("sampling_args") or {}).get("temperature", self.runtime_config.temperature))
        prompt_messages = trajectory_step.get("prompt") or []
        provenance = prompt_provenance(prompt_messages if isinstance(prompt_messages, list) else [])
        segment = make_segment(
            order=int(state["rlm_segment_counter"]),
            call_id=0,
            parent_call_id=None,
            depth=0,
            turn_index=max(0, len(state.get("trajectory") or []) - 1),
            kind="root_turn",
            train_scope="root_turn",
            is_trainable_rlm_turn=True,
            response_source="root",
            prompt_ids=list(tokens["prompt_ids"]),
            completion_ids=list(tokens["completion_ids"]),
            completion_logprobs=[float(value) for value in tokens["completion_logprobs"]],
            completion_mask=[bool(value) for value in tokens["completion_mask"]],
            temperature=temperature,
            response_text=trajectory_step["completion"][-1].get("content", ""),
            prompt_fingerprint=provenance["prompt_fingerprint"],
            prompt_message_count=provenance["prompt_message_count"],
            prompt_char_count=provenance["prompt_char_count"],
        )
        trajectory_step["extras"]["rlm_segment_order"] = segment["order"]
        state["rlm_segment_counter"] += 1
        state["rlm_segments"].append(segment)
        prompt_token_count = float(len(segment["prompt_ids"]))
        completion_token_count = float(len(segment["completion_ids"]))
        state["total_model_tokens"] = float(state.get("total_model_tokens", 0.0)) + completion_token_count
        state["total_prompt_tokens"] = float(state.get("total_prompt_tokens", 0.0)) + prompt_token_count
        state["total_completion_tokens"] = float(state.get("total_completion_tokens", 0.0)) + completion_token_count
        state["total_rollout_tokens"] = float(state.get("total_rollout_tokens", 0.0)) + prompt_token_count + completion_token_count
        write_live_trace(state, event="root_segment")

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        del kwargs
        runtime: RecursiveRuntime = state["_runtime"]
        assistant_text = str(messages[-1].get("content", "")) if messages else ""
        repl = state["_root_repl"]

        code_block_strs = find_code_blocks(assistant_text)
        code_blocks: list[CodeBlock] = []
        if code_block_strs:
            state["used_repl"] = True

        for code in code_block_strs:
            execution = repl.execute_code(code)
            code_blocks.append(CodeBlock(code=code, result=execution))
            if execution.final_answer is not None and state.get("final_answer") is None:
                state["final_answer"] = execution.final_answer

        if state.get("final_answer") is None:
            final_answer = extract_final_answer(assistant_text, environment=repl)
            if final_answer is not None:
                state["final_answer"] = final_answer

        session = state["_sync_session"]
        iteration = RLMIteration(
            prompt=messages,
            response=assistant_text,
            code_blocks=code_blocks,
            final_answer=state.get("final_answer"),
        )
        feedback_messages = make_feedback_messages(
            iteration,
            max_chars=self.runtime_config.execution_output_char_limit,
        )
        budget_feedback = getattr(runtime, "budget_feedback_message", None)
        budget_message = budget_feedback() if callable(budget_feedback) else None
        if budget_message is not None:
            feedback_messages.append(budget_message)
        root_trace = state.get("_root_trace")
        if root_trace is not None:
            append_step_trace(
                root_trace,
                assistant=assistant_text,
                code_blocks=code_block_strs,
                feedback=[message["content"] for message in feedback_messages],
                final_answer=state.get("final_answer"),
            )
            write_live_trace(state, event="root_step")

        if state.get("final_answer") is not None:
            if state.get("used_forced_finalize_prompt"):
                state["finalized_on_forced_prompt"] = True
            else:
                state["finalized_before_forced_prompt"] = True
            self._attach_debug_payload(state)
            write_live_trace(state, event="root_complete")
            state["final_env_response"] = []
            return []

        next_iteration = len(state["trajectory"])
        if state.get("used_forced_finalize_prompt"):
            state["hit_max_turn_without_final"] = True
            state["missing_final"] = True
            self._attach_debug_payload(state)
            write_live_trace(state, event="root_missing_final")
            state["final_env_response"] = []
            return []

        if next_iteration >= self.runtime_config.max_iterations:
            state["used_forced_finalize_prompt"] = True
            next_message = {"role": "user", "content": runtime.build_finalize_message()}
        else:
            next_message = build_user_prompt(
                root_prompt=str((state.get("info") or {}).get("question", "")),
                iteration=next_iteration,
                context_count=int(repl.get_context_count()),
                history_count=int(repl.get_history_count()),
            )

        response_messages = [*feedback_messages, next_message]
        state["total_env_tokens"] += float(sum(session.count_text_tokens(message["content"]) for message in response_messages))
        self._attach_debug_payload(state)
        write_live_trace(state, event="root_feedback")
        return response_messages


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
    max_iterations: int = 15,
    max_depth: int = 2,
    turn_max_tokens: int = 192,
    subcall_max_tokens: int = 128,
    max_prompt_tokens: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    tokenizer_name: str | None = None,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    live_trace_dir: str | None = "outputs/rlm_rlvr/live_traces",
    subcall_prompt_limit_ratio: float = 0.85,
    subcall_budget_enabled: bool = False,
    max_total_subcalls: int = 80,
    max_batched_subcalls: int = 80,
    include_budget_reminder: bool = True,
    efficiency_penalty_coef: float = 0.02,
    inference_mode: str = "hosted",
    inference_base_url: str | None = None,
    inference_api_key: str | None = None,
    llm_subcall_provider: str = "openai_compatible",
    llm_subcall_model: str | None = None,
    llm_subcall_base_url: str | None = None,
    llm_subcall_api_key_var: str = "OPENROUTER_API_KEY",
    llm_subcall_http_referer: str | None = None,
    llm_subcall_app_title: str | None = None,
    llm_subcall_vertex_project_env: str = "GOOGLE_CLOUD_PROJECT",
    llm_subcall_vertex_location: str | None = "global",
    llm_subcall_thinking_level: str | None = "medium",
    llm_subcall_empty_response_max_attempts: int = 1,
    llm_subcall_empty_response_base_retry_seconds: float = 1.0,
    llm_subcall_empty_response_max_retry_seconds: float = 30.0,
    judge_provider: str = "openai_compatible",
    judge_model: str = "z-ai/glm-5",
    judge_base_url: str = "https://openrouter.ai/api/v1",
    judge_api_key_var: str = "OPENROUTER_API_KEY",
    judge_http_referer: str | None = None,
    judge_app_title: str | None = None,
    judge_vertex_project_env: str = "GOOGLE_CLOUD_PROJECT",
    judge_vertex_location: str | None = "global",
    judge_thinking_level: str | None = "medium",
    efficiency_penalty_mode: str = "static_per_1k",
    adaptive_efficiency_beta_max: float = 0.05,
    adaptive_efficiency_gamma: float = 2.0,
    adaptive_efficiency_solve_rate_floor: float = 0.25,
    adaptive_efficiency_cost_basis: str = "total_tokens",
    efficiency_penalty_applies_to: str = "correct_only",
    reward_clip_min: float = 0.0,
    reward_clip_max: float = 1.0,
    max_turn_penalty_enabled: bool = False,
    max_turn_penalty: float = 0.25,
    missing_final_at_max_turn_zero_reward: bool = True,
    repl_backend: str = "local",
    repl_backend_kwargs: dict[str, Any] | None = None,
    repl_timeout_seconds: float | None = None,
    repl_fast_timeout_seconds: float | None = None,
    recursive_rlm_batch_mode: str = "serial",
    recursive_cap_prompt_variant: str | None = None,
) -> vf.Environment:
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if turn_max_tokens < 1 or subcall_max_tokens < 1:
        raise ValueError("turn_max_tokens and subcall_max_tokens must be >= 1")
    if max_prompt_tokens is not None and max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be >= 1")
    if not 0 < subcall_prompt_limit_ratio <= 1:
        raise ValueError("subcall_prompt_limit_ratio must be > 0 and <= 1")
    if max_total_subcalls < 1:
        raise ValueError("max_total_subcalls must be >= 1")
    if max_batched_subcalls < 1:
        raise ValueError("max_batched_subcalls must be >= 1")
    if llm_subcall_empty_response_max_attempts < 1:
        raise ValueError("llm_subcall_empty_response_max_attempts must be >= 1")
    if llm_subcall_empty_response_base_retry_seconds < 0:
        raise ValueError("llm_subcall_empty_response_base_retry_seconds must be >= 0")
    if llm_subcall_empty_response_max_retry_seconds < 0:
        raise ValueError("llm_subcall_empty_response_max_retry_seconds must be >= 0")
    if repl_timeout_seconds is not None and repl_timeout_seconds <= 0:
        raise ValueError("repl_timeout_seconds must be > 0")
    if repl_fast_timeout_seconds is not None and repl_fast_timeout_seconds <= 0:
        raise ValueError("repl_fast_timeout_seconds must be > 0")
    if not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0.0")
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"prompt_variant must be one of {sorted(PROMPT_VARIANTS)}")
    if recursive_cap_prompt_variant is not None and recursive_cap_prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"recursive_cap_prompt_variant must be one of {sorted(PROMPT_VARIANTS)}")
    valid_efficiency_penalty_modes = {"static_per_1k", "adaptive_group"}
    if efficiency_penalty_mode not in valid_efficiency_penalty_modes:
        raise ValueError(f"efficiency_penalty_mode must be one of {sorted(valid_efficiency_penalty_modes)}")
    if adaptive_efficiency_beta_max < 0.0:
        raise ValueError("adaptive_efficiency_beta_max must be >= 0.0")
    if adaptive_efficiency_gamma <= 0.0:
        raise ValueError("adaptive_efficiency_gamma must be > 0.0")
    if not 0.0 <= adaptive_efficiency_solve_rate_floor < 1.0:
        raise ValueError("adaptive_efficiency_solve_rate_floor must be >= 0.0 and < 1.0")
    valid_adaptive_cost_bases = {"total_tokens"}
    if adaptive_efficiency_cost_basis not in valid_adaptive_cost_bases:
        raise ValueError(f"adaptive_efficiency_cost_basis must be one of {sorted(valid_adaptive_cost_bases)}")
    valid_efficiency_penalty_scopes = {"correct_only", "all_rollouts"}
    if efficiency_penalty_applies_to not in valid_efficiency_penalty_scopes:
        raise ValueError(f"efficiency_penalty_applies_to must be one of {sorted(valid_efficiency_penalty_scopes)}")
    if reward_clip_min > reward_clip_max:
        raise ValueError("reward_clip_min must be <= reward_clip_max")
    if max_turn_penalty < 0.0:
        raise ValueError("max_turn_penalty must be >= 0.0")

    valid_api_providers = {"openai_compatible", "vertex"}
    if llm_subcall_provider not in valid_api_providers:
        raise ValueError(f"llm_subcall_provider must be one of {sorted(valid_api_providers)}")
    if judge_provider not in valid_api_providers:
        raise ValueError(f"judge_provider must be one of {sorted(valid_api_providers)}")

    valid_inference_modes = {"hosted", "local"}
    if inference_mode not in valid_inference_modes:
        raise ValueError(f"inference_mode must be one of {sorted(valid_inference_modes)}")

    valid_repl_backends = {"local"}
    if repl_backend not in valid_repl_backends:
        raise ValueError("rlm_rlvr currently supports only local REPL execution.")
    recursive_rlm_batch_mode = recursive_rlm_batch_mode.lower()
    valid_recursive_rlm_batch_modes = {"serial", "thread"}
    if recursive_rlm_batch_mode not in valid_recursive_rlm_batch_modes:
        raise ValueError(f"recursive_rlm_batch_mode must be one of {sorted(valid_recursive_rlm_batch_modes)}")

    if inference_base_url is None:
        if inference_mode == "local":
            inference_base_url = os.environ.get("RLM_LOCAL_INFERENCE_BASE_URL")
        elif inference_mode == "hosted":
            inference_base_url = os.environ.get("RLM_HOSTED_INFERENCE_BASE_URL")

    if inference_api_key is None:
        if inference_mode == "local":
            inference_api_key = os.environ.get("RLM_LOCAL_INFERENCE_API_KEY")
        elif inference_mode == "hosted":
            inference_api_key = os.environ.get("RLM_HOSTED_INFERENCE_API_KEY")

    required_keys: list[str] = []
    if judge_provider == "openai_compatible":
        required_keys.append(judge_api_key_var)
    else:
        required_keys.append(judge_vertex_project_env)
    if llm_subcall_model is not None and llm_subcall_provider == "openai_compatible":
        required_keys.append(llm_subcall_api_key_var)
    elif llm_subcall_model is not None and llm_subcall_provider == "vertex":
        required_keys.append(llm_subcall_vertex_project_env)
    vf.ensure_keys(sorted(set(required_keys)))
    judge_default_headers = {
        key: value
        for key, value in {
            "HTTP-Referer": judge_http_referer,
            "X-Title": judge_app_title,
        }.items()
        if value
    }
    llm_subcall_default_headers = {
        key: value
        for key, value in {
            "HTTP-Referer": llm_subcall_http_referer,
            "X-Title": llm_subcall_app_title,
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
    runtime_config = RuntimeConfig(
        max_depth=max_depth,
        max_iterations=max_iterations,
        turn_max_tokens=turn_max_tokens,
        subcall_max_tokens=subcall_max_tokens,
        max_prompt_tokens=max_prompt_tokens,
        temperature=temperature,
        top_p=top_p,
        tokenizer_name=tokenizer_name,
        inference_mode=inference_mode,
        inference_base_url=inference_base_url,
        inference_api_key=inference_api_key,
        llm_subcall_provider=llm_subcall_provider,
        llm_subcall_model=llm_subcall_model,
        llm_subcall_base_url=llm_subcall_base_url,
        llm_subcall_api_key=(
            os.environ.get(llm_subcall_api_key_var)
            if llm_subcall_model is not None and llm_subcall_provider == "openai_compatible"
            else None
        ),
        llm_subcall_default_headers=llm_subcall_default_headers or None,
        llm_subcall_vertex_project=(
            os.environ.get(llm_subcall_vertex_project_env)
            if llm_subcall_model is not None and llm_subcall_provider == "vertex"
            else None
        ),
        llm_subcall_vertex_location=llm_subcall_vertex_location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        llm_subcall_thinking_level=llm_subcall_thinking_level,
        llm_subcall_empty_response_max_attempts=llm_subcall_empty_response_max_attempts,
        llm_subcall_empty_response_base_retry_seconds=llm_subcall_empty_response_base_retry_seconds,
        llm_subcall_empty_response_max_retry_seconds=llm_subcall_empty_response_max_retry_seconds,
        repl_backend=repl_backend,
        repl_backend_kwargs=repl_backend_kwargs,
        repl_timeout_seconds=repl_timeout_seconds,
        repl_fast_timeout_seconds=repl_fast_timeout_seconds,
        prompt_variant=prompt_variant,
        live_trace_dir=live_trace_dir,
        subcall_prompt_limit_ratio=subcall_prompt_limit_ratio,
        subcall_budget_enabled=subcall_budget_enabled,
        max_total_subcalls=max_total_subcalls,
        max_batched_subcalls=max_batched_subcalls,
        include_budget_reminder=include_budget_reminder,
        recursive_rlm_batch_mode=recursive_rlm_batch_mode,
        recursive_cap_prompt_variant=recursive_cap_prompt_variant,
    )
    system_prompt = build_system_prompt(
        depth=0,
        max_depth=max_depth,
        prompt_variant=prompt_variant,
        max_prompt_tokens=max_prompt_tokens,
        turn_max_tokens=turn_max_tokens,
        subcall_max_tokens=subcall_max_tokens,
        subcall_budget_enabled=subcall_budget_enabled,
        max_total_subcalls=max_total_subcalls,
        max_batched_subcalls=max_batched_subcalls,
        include_budget_reminder=include_budget_reminder,
    )
    reward_rubric = build_rubric(
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=os.environ[judge_api_key_var] if judge_provider == "openai_compatible" else None,
        judge_default_headers=judge_default_headers or None,
        judge_vertex_project=os.environ.get(judge_vertex_project_env) if judge_provider == "vertex" else None,
        judge_vertex_location=judge_vertex_location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        judge_thinking_level=judge_thinking_level,
        efficiency_penalty_mode=efficiency_penalty_mode,
        adaptive_efficiency_beta_max=adaptive_efficiency_beta_max,
        adaptive_efficiency_gamma=adaptive_efficiency_gamma,
        adaptive_efficiency_solve_rate_floor=adaptive_efficiency_solve_rate_floor,
        adaptive_efficiency_cost_basis=adaptive_efficiency_cost_basis,
        efficiency_penalty_applies_to=efficiency_penalty_applies_to,
        reward_clip_min=reward_clip_min,
        reward_clip_max=reward_clip_max,
        max_turn_penalty_enabled=max_turn_penalty_enabled,
        max_turn_penalty=max_turn_penalty,
        missing_final_at_max_turn_zero_reward=missing_final_at_max_turn_zero_reward,
    )
    add_metrics(reward_rubric)
    return RLMRLVREnv(
        dataset=train_builder,
        eval_dataset=eval_builder,
        system_prompt=system_prompt,
        rubric=reward_rubric,
        runtime_config=runtime_config,
        efficiency_penalty_coef=efficiency_penalty_coef,
        efficiency_penalty_mode=efficiency_penalty_mode,
        adaptive_efficiency_beta_max=adaptive_efficiency_beta_max,
        adaptive_efficiency_gamma=adaptive_efficiency_gamma,
        adaptive_efficiency_solve_rate_floor=adaptive_efficiency_solve_rate_floor,
        adaptive_efficiency_cost_basis=adaptive_efficiency_cost_basis,
        efficiency_penalty_applies_to=efficiency_penalty_applies_to,
        reward_clip_min=reward_clip_min,
        reward_clip_max=reward_clip_max,
        max_turn_penalty_enabled=max_turn_penalty_enabled,
        max_turn_penalty=max_turn_penalty,
        missing_final_at_max_turn_zero_reward=missing_final_at_max_turn_zero_reward,
        env_id="rlm_rlvr",
    )
