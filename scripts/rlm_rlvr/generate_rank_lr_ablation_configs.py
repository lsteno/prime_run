#!/usr/bin/env python3
"""Generate the Qwen3-4B depth-1 LLM-only RLM RLVR rank/LR ablation configs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs/rlm_rlvr/qwen3_4b_instruct_sanjaya_medium_8xa100_40gb_budgeted.toml"
OUT_DIR = ROOT / "configs/rlm_rlvr/ablation_rank_lr"
PROMPT_VARIANT = "sanjaya_text_depth1_llm_only_v1"
RUNTIME_MAX_DEPTH = 0
EXPERIMENT_DEPTH = 1
TRAIN_WORKER_COUNT = 32
EVAL_WORKER_COUNT = 16
MAX_REQUESTS_PER_ENV_WORKER = 1
ROLLOUT_TIMEOUT_SECONDS = 400
REPL_TIMEOUT_SECONDS = 300
REPL_FAST_TIMEOUT_SECONDS = 30
ENV_WORKER_CANCEL_GRACE_SECONDS = 5
MAX_ROLLOUT_ATTEMPTS_PER_SLOT = 4
MAX_ATTEMPTS_COOLDOWN_STEPS = 5
DROP_GROUP_ON_FIRST_TIMEOUT = False
FIRST_TIMEOUT_COOLDOWN_STEPS = 5
MAX_TOTAL_SUBCALLS = 50
MAX_BATCHED_SUBCALLS = 50
LLM_SUBCALL_EMPTY_RESPONSE_MAX_ATTEMPTS = 3
LLM_SUBCALL_EMPTY_RESPONSE_BASE_RETRY_SECONDS = 1.0
LLM_SUBCALL_EMPTY_RESPONSE_MAX_RETRY_SECONDS = 10.0
ADAPTIVE_EFFICIENCY_BETA_MAX = 0.15
ADAPTIVE_EFFICIENCY_GAMMA = 1.0
ADAPTIVE_EFFICIENCY_SOLVE_RATE_FLOOR = 0.25
EFFICIENCY_PENALTY_APPLIES_TO = "all_rollouts"
REWARD_CLIP_MIN = -0.5
REWARD_CLIP_MAX = 1.0
MAX_TURN_PENALTY_ENABLED = True
MAX_TURN_PENALTY = 0.25
MISSING_FINAL_AT_MAX_TURN_ZERO_REWARD = True
MAX_ASYNC_LEVEL = 2
MAX_OFF_POLICY_STEPS = 4
HARD_COOLDOWN_STEPS = 5
PREFETCH_NEXT_BATCH = False
INFLIGHT_COMPLETION_CUSHION = 32
MAX_CROSS_STEP_CARRYOVER = 32
MAX_CARRYOVER_STEPS = 1
CANCEL_STALE_CARRYOVER = True
RESTART_WORKERS_FOR_STALE_CANCEL = True
BATCH_COMPLETE_CANCEL_GRACE_SECONDS = 2
GROUP_SCORING_ENABLED = True
GROUP_SCORING_MAX_CONCURRENCY = 8
GROUP_SCORING_MAX_PENDING_GROUPS = 64
DEPTH1_TRAIN_SEQ_LEN = 49152
DEPTH1_MAX_PROMPT_TOKENS = 47104
DEPTH1_CONTEXT_PARALLEL = 2
DEPTH1_ACTIVATION_CHECKPOINT_FREQ = 2
DATASET_TAG = "balanced35_40_25_frames40_v1"
RUN_SUFFIX = "bal35f40v1"
BALANCED_DATA_DIR_FROM_PRIME_RL = "../data/beeg_agents_balanced_35_40_25_frames40_v1"
BALANCED_TRAIN_PATH = f"{BALANCED_DATA_DIR_FROM_PRIME_RL}/train.parquet"
BALANCED_EVAL_PATH = f"{BALANCED_DATA_DIR_FROM_PRIME_RL}/eval.parquet"

MATRIX = (
    (4, 8, "5e-7"),
    (4, 8, "1e-5"),
    (4, 8, "1e-4"),
    (16, 32, "5e-7"),
    (16, 32, "1e-5"),
    (16, 32, "1e-4"),
    (64, 128, "5e-7"),
    (64, 128, "1e-5"),
    (64, 128, "1e-4"),
)


def replace_line(text: str, pattern: str, replacement: str) -> str:
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected exactly one replacement for pattern: {pattern}")
    return text


def replace_all(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Missing text to replace: {old}")
    return text.replace(old, new)


def insert_after_once(text: str, needle: str, insertion: str) -> str:
    if needle not in text:
        raise RuntimeError(f"Missing insertion point: {needle}")
    return text.replace(needle, needle + insertion, 1)


def replace_or_insert_line_after_once(text: str, *, line_pattern: str, needle: str, line: str) -> str:
    text, count = re.subn(line_pattern, line, text, count=1, flags=re.MULTILINE)
    if count == 1:
        return text
    return insert_after_once(text, needle, line + "\n")


def build_config(base: str, *, rank: int, alpha: int, lr: str, run_id: str) -> str:
    output_dir = f"../outputs/{run_id}"
    text = base
    text = replace_all(
        text,
        "# Medium local RLVR run for Qwen3 4B Instruct on a single 8x A100 40GB node,",
        "# LoRA rank/LR ablation run for Qwen3 4B Instruct on a single 8x A100 80GB node,",
    )
    text = replace_all(
        text,
        "# with lower rollout pressure and an enabled subcall budget.",
        "# with depth-1 LLM-only orchestration, lower rollout pressure, and an enabled subcall budget.",
    )
    text = replace_all(
        text,
        "# - GPUs 4-7: LoRA trainer, context-parallel over a reduced 262k flattened RLM trajectory.",
        "# - GPUs 4-7: LoRA trainer, context-parallel over a 48k flattened RLM trajectory.",
    )
    text = replace_all(
        text,
        "#   uv run rl @ ../configs/rlm_rlvr/qwen3_4b_instruct_sanjaya_medium_8xa100_40gb_budgeted.toml",
        f"#   uv run rl @ ../configs/rlm_rlvr/ablation_rank_lr/{config_filename(rank, alpha, lr)}",
    )
    text = replace_line(text, r'^output_dir = ".+"$', f'output_dir = "{output_dir}"')
    text = replace_line(text, r"^max_steps = \d+$", "max_steps = 150")
    text = replace_line(text, r"^max_async_level = \d+$", f"max_async_level = {MAX_ASYNC_LEVEL}")
    text = replace_line(text, r"^max_off_policy_steps = \d+$", f"max_off_policy_steps = {MAX_OFF_POLICY_STEPS}")
    text = insert_after_once(
        text,
        f"max_off_policy_steps = {MAX_OFF_POLICY_STEPS}\n",
        "\n"
        "[orchestrator.env_worker_recovery]\n"
        "enabled = true\n"
        f"cancel_grace_seconds = {ENV_WORKER_CANCEL_GRACE_SECONDS}\n"
        f"max_rollout_attempts_per_slot = {MAX_ROLLOUT_ATTEMPTS_PER_SLOT}\n"
        f"max_attempts_cooldown_steps = {MAX_ATTEMPTS_COOLDOWN_STEPS}\n"
        f"drop_group_on_first_timeout = {str(DROP_GROUP_ON_FIRST_TIMEOUT).lower()}\n"
        f"first_timeout_cooldown_steps = {FIRST_TIMEOUT_COOLDOWN_STEPS}\n"
        "restart_on_rollout_timeout = true\n"
        "restart_on_worker_health_failure = true\n"
        "\n"
        "[orchestrator.attempt_logging]\n"
        "enabled = true\n",
    )
    text = insert_after_once(
        text,
        "[orchestrator.attempt_logging]\n"
        "enabled = true\n",
        "\n"
        "[orchestrator.async_scheduling]\n"
        f"prefetch_next_batch = {str(PREFETCH_NEXT_BATCH).lower()}\n"
        f"inflight_completion_cushion = {INFLIGHT_COMPLETION_CUSHION}\n"
        f"max_requests_per_env_worker = {MAX_REQUESTS_PER_ENV_WORKER}\n"
        f"max_cross_step_carryover = {MAX_CROSS_STEP_CARRYOVER}\n"
        f"max_carryover_steps = {MAX_CARRYOVER_STEPS}\n"
        f"cancel_stale_carryover = {str(CANCEL_STALE_CARRYOVER).lower()}\n"
        f"restart_workers_for_stale_cancel = {str(RESTART_WORKERS_FOR_STALE_CANCEL).lower()}\n"
        f"batch_complete_cancel_grace_seconds = {BATCH_COMPLETE_CANCEL_GRACE_SECONDS}\n",
    )
    text = insert_after_once(
        text,
        f"batch_complete_cancel_grace_seconds = {BATCH_COMPLETE_CANCEL_GRACE_SECONDS}\n",
        "\n"
        "[orchestrator.group_scoring]\n"
        f"enabled = {str(GROUP_SCORING_ENABLED).lower()}\n"
        f"max_concurrency = {GROUP_SCORING_MAX_CONCURRENCY}\n"
        f"max_pending_groups = {GROUP_SCORING_MAX_PENDING_GROUPS}\n",
    )
    text = replace_line(text, r'^name = "qwen3-4b-instruct-sanjaya-medium-8xa100-40gb-budgeted"$', f'name = "{run_id}"')
    text = replace_all(
        text,
        "samples = true\n"
        "distributions = true\n"
        "interval = 1\n",
        "samples = true\n"
        "distributions = true\n"
        "interval = 10\n"
        "sample_max_chars = 8000\n"
        "sample_include_input_ids = false\n"
        "final_samples = false\n",
    )
    text = replace_line(text, r"^lr = .+$", f"lr = {lr}")
    text = replace_line(text, r"^rank = \d+$", f"rank = {rank}")
    text = replace_line(text, r"^alpha = \d+$", f"alpha = {alpha}")
    text = replace_all(text, "seq_len = 262144", f"seq_len = {DEPTH1_TRAIN_SEQ_LEN}")
    text = replace_line(text, r"^cp = \d+$", f"cp = {DEPTH1_CONTEXT_PARALLEL}")
    text = replace_all(text, "max_prompt_tokens = 61440", f"max_prompt_tokens = {DEPTH1_MAX_PROMPT_TOKENS}")
    text = replace_line(text, r"^max_model_len = \d+$", f"max_model_len = {DEPTH1_TRAIN_SEQ_LEN}")
    text = replace_all(
        text,
        "\n[trainer.model.ac]\nfreq = 1\n",
        f"\n[trainer.model.ac]\nfreq = {DEPTH1_ACTIVATION_CHECKPOINT_FREQ}\n",
    )
    text = insert_after_once(text, "max_concurrent = 32\n", f"rollout_timeout_seconds = {ROLLOUT_TIMEOUT_SECONDS}\n")
    text = replace_all(text, '[[orchestrator.env]]\nid = "rlm_rlvr"\n', f'[[orchestrator.env]]\nid = "rlm_rlvr"\nworker_count = {TRAIN_WORKER_COUNT}\n')
    text = replace_all(
        text,
        '[[orchestrator.eval.env]]\nid = "rlm_rlvr"\nname = "rlm_rlvr_eval"\n',
        f'[[orchestrator.eval.env]]\nid = "rlm_rlvr"\nname = "rlm_rlvr_eval"\nworker_count = {EVAL_WORKER_COUNT}\n',
    )
    text = replace_all(text, 'prompt_variant = "sanjaya_text_v1"', f'prompt_variant = "{PROMPT_VARIANT}"')
    text = replace_all(
        text,
        'dataset_id = "lsteno/BEEG-agents"\n'
        'dataset_train_split = "train"\n'
        'dataset_eval_split = "eval"\n',
        'dataset_id = "lsteno/BEEG-agents"\n'
        f'data_paths = ["{BALANCED_TRAIN_PATH}"]\n'
        f'eval_data_paths = ["{BALANCED_EVAL_PATH}"]\n'
        'dataset_train_split = "train"\n'
        'dataset_eval_split = "eval"\n',
    )
    text = replace_all(text, "max_depth = 2", f"max_depth = {RUNTIME_MAX_DEPTH}")
    text = replace_all(text, "max_total_subcalls = 80", f"max_total_subcalls = {MAX_TOTAL_SUBCALLS}")
    text = replace_all(text, "max_batched_subcalls = 80", f"max_batched_subcalls = {MAX_BATCHED_SUBCALLS}")
    text = replace_all(
        text,
        'llm_subcall_thinking_level = "medium"\n',
        'llm_subcall_thinking_level = "medium"\n'
        f"llm_subcall_empty_response_max_attempts = {LLM_SUBCALL_EMPTY_RESPONSE_MAX_ATTEMPTS}\n"
        f"llm_subcall_empty_response_base_retry_seconds = {LLM_SUBCALL_EMPTY_RESPONSE_BASE_RETRY_SECONDS}\n"
        f"llm_subcall_empty_response_max_retry_seconds = {LLM_SUBCALL_EMPTY_RESPONSE_MAX_RETRY_SECONDS}\n",
    )
    text = replace_line(
        text,
        r"^adaptive_efficiency_beta_max = .+$",
        f"adaptive_efficiency_beta_max = {ADAPTIVE_EFFICIENCY_BETA_MAX}",
    )
    text = replace_line(
        text,
        r"^adaptive_efficiency_gamma = .+$",
        f"adaptive_efficiency_gamma = {ADAPTIVE_EFFICIENCY_GAMMA}",
    )
    text = replace_line(
        text,
        r"^adaptive_efficiency_solve_rate_floor = .+$",
        f"adaptive_efficiency_solve_rate_floor = {ADAPTIVE_EFFICIENCY_SOLVE_RATE_FLOOR}",
    )
    text = replace_all(
        text,
        'adaptive_efficiency_cost_basis = "total_tokens"\n',
        'adaptive_efficiency_cost_basis = "total_tokens"\n'
        f'efficiency_penalty_applies_to = "{EFFICIENCY_PENALTY_APPLIES_TO}"\n'
        f"reward_clip_min = {REWARD_CLIP_MIN}\n"
        f"reward_clip_max = {REWARD_CLIP_MAX}\n"
        f"max_turn_penalty_enabled = {str(MAX_TURN_PENALTY_ENABLED).lower()}\n"
        f"max_turn_penalty = {MAX_TURN_PENALTY}\n"
        f"missing_final_at_max_turn_zero_reward = {str(MISSING_FINAL_AT_MAX_TURN_ZERO_REWARD).lower()}\n",
    )
    text = replace_all(
        text,
        "efficiency_penalty_coef = 0.0\n",
        "efficiency_penalty_coef = 0.0\n"
        f'efficiency_penalty_applies_to = "{EFFICIENCY_PENALTY_APPLIES_TO}"\n'
        f"reward_clip_min = {REWARD_CLIP_MIN}\n"
        f"reward_clip_max = {REWARD_CLIP_MAX}\n"
        f"max_turn_penalty_enabled = {str(MAX_TURN_PENALTY_ENABLED).lower()}\n"
        f"max_turn_penalty = {MAX_TURN_PENALTY}\n"
        f"missing_final_at_max_turn_zero_reward = {str(MISSING_FINAL_AT_MAX_TURN_ZERO_REWARD).lower()}\n",
    )
    text = replace_or_insert_line_after_once(
        text,
        line_pattern=r"^hard_cooldown_steps = \d+$",
        needle="hard_threshold = 0.0\n",
        line=f"hard_cooldown_steps = {HARD_COOLDOWN_STEPS}",
    )
    text = replace_all(text, "easy_threshold = 1.0\n", "")
    text = replace_all(
        text,
        "online_difficulty_filtering = true\n",
        "online_difficulty_filtering = true\n"
        "online_filter_hard = true\n"
        "online_filter_easy = false\n",
    )
    text = replace_all(text, "repl_timeout_seconds = 900", f"repl_timeout_seconds = {REPL_TIMEOUT_SECONDS}")
    text = replace_all(text, "repl_fast_timeout_seconds = 30", f"repl_fast_timeout_seconds = {REPL_FAST_TIMEOUT_SECONDS}")
    text = replace_all(
        text,
        'live_trace_dir = "../outputs/rlm-rlvr-qwen3-4b-instruct-sanjaya-medium-8xa100-40gb-budgeted/live_traces"',
        f'live_trace_dir = "{output_dir}/live_traces"',
    )
    text = replace_all(
        text,
        'live_trace_dir = "../outputs/rlm-rlvr-qwen3-4b-instruct-sanjaya-medium-8xa100-40gb-budgeted/live_traces_eval"',
        f'live_trace_dir = "{output_dir}/live_traces_eval"',
    )
    rank_note = (
        "# Note: rank 4 uses Prime-RL dense LoRA's non-grouped fallback path.\n"
        if rank == 4
        else ""
    )
    header = (
        "# Generated by scripts/rlm_rlvr/generate_rank_lr_ablation_configs.py.\n"
        "# Keep these as separate runs so LoRA rank is a true trainer-side ablation.\n"
        "# Depth-1 LLM-only setup: prompt omits RLM tools; runtime max_depth=0 blocks recursive RLM children.\n"
        f"# Dataset: local derived BEEG parquets ({DATASET_TAG}) with train/eval near 35% oolong, 40% frames, 25% longcodeu.\n"
        f"# Ablation parameters: rank={rank}, alpha={alpha}, lr={lr}, max_steps=150.\n"
        f"{rank_note}\n"
    )
    return header + text


def config_filename(rank: int, alpha: int, lr: str) -> str:
    return f"qwen3_4b_instruct_sanjaya_depth1_llmonly_r{rank:03d}_a{alpha:03d}_lr{lr}_s150_8xa10080_{RUN_SUFFIX}.toml"


def main() -> None:
    base = BASE_CONFIG.read_text()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale_patterns = (
        "qwen3_4b_instruct_sanjaya_r*_s150_8xa10080.toml",
        "qwen3_4b_instruct_sanjaya_depth1_llmonly_r*_s150_8xa10080.toml",
    )
    for stale_pattern in stale_patterns:
        for stale_config in OUT_DIR.glob(stale_pattern):
            stale_config.unlink()
    rows: list[dict[str, str | int]] = []
    for index, (rank, alpha, lr) in enumerate(MATRIX, start=1):
        run_id = f"rlm-rlvr-qwen3-4b-depth1-llmonly-r{rank}-a{alpha}-lr{lr}-s150-{RUN_SUFFIX}"
        filename = config_filename(rank, alpha, lr)
        config_path = OUT_DIR / filename
        config_path.write_text(build_config(base, rank=rank, alpha=alpha, lr=lr, run_id=run_id))
        rows.append(
            {
                "index": index,
                "run_id": run_id,
                "rank": rank,
                "alpha": alpha,
                "lr": lr,
                "max_steps": 150,
                "max_async_level": MAX_ASYNC_LEVEL,
                "max_off_policy_steps": MAX_OFF_POLICY_STEPS,
                "hard_cooldown_steps": HARD_COOLDOWN_STEPS,
                "prefetch_next_batch": str(PREFETCH_NEXT_BATCH).lower(),
                "inflight_completion_cushion": INFLIGHT_COMPLETION_CUSHION,
                "max_requests_per_env_worker": MAX_REQUESTS_PER_ENV_WORKER,
                "max_cross_step_carryover": MAX_CROSS_STEP_CARRYOVER,
                "max_carryover_steps": MAX_CARRYOVER_STEPS,
                "cancel_stale_carryover": str(CANCEL_STALE_CARRYOVER).lower(),
                "restart_workers_for_stale_cancel": str(RESTART_WORKERS_FOR_STALE_CANCEL).lower(),
                "batch_complete_cancel_grace_seconds": BATCH_COMPLETE_CANCEL_GRACE_SECONDS,
                "group_scoring_enabled": str(GROUP_SCORING_ENABLED).lower(),
                "group_scoring_max_concurrency": GROUP_SCORING_MAX_CONCURRENCY,
                "group_scoring_max_pending_groups": GROUP_SCORING_MAX_PENDING_GROUPS,
                "seq_len": DEPTH1_TRAIN_SEQ_LEN,
                "max_prompt_tokens": DEPTH1_MAX_PROMPT_TOKENS,
                "cp": DEPTH1_CONTEXT_PARALLEL,
                "seed": 42,
                "batch_size": 64,
                "rollouts_per_example": 4,
                "train_worker_count": TRAIN_WORKER_COUNT,
                "eval_worker_count": EVAL_WORKER_COUNT,
                "rollout_timeout_seconds": ROLLOUT_TIMEOUT_SECONDS,
                "env_worker_cancel_grace_seconds": ENV_WORKER_CANCEL_GRACE_SECONDS,
                "max_rollout_attempts_per_slot": MAX_ROLLOUT_ATTEMPTS_PER_SLOT,
                "max_attempts_cooldown_steps": MAX_ATTEMPTS_COOLDOWN_STEPS,
                "drop_group_on_first_timeout": str(DROP_GROUP_ON_FIRST_TIMEOUT).lower(),
                "first_timeout_cooldown_steps": FIRST_TIMEOUT_COOLDOWN_STEPS,
                "wandb_log_extras_interval": 10,
                "wandb_sample_max_chars": 8000,
                "wandb_sample_include_input_ids": "false",
                "wandb_final_samples": "false",
                "repl_timeout_seconds": REPL_TIMEOUT_SECONDS,
                "max_total_subcalls": MAX_TOTAL_SUBCALLS,
                "max_batched_subcalls": MAX_BATCHED_SUBCALLS,
                "llm_subcall_empty_response_max_attempts": LLM_SUBCALL_EMPTY_RESPONSE_MAX_ATTEMPTS,
                "llm_subcall_empty_response_base_retry_seconds": LLM_SUBCALL_EMPTY_RESPONSE_BASE_RETRY_SECONDS,
                "llm_subcall_empty_response_max_retry_seconds": LLM_SUBCALL_EMPTY_RESPONSE_MAX_RETRY_SECONDS,
                "adaptive_efficiency_beta_max": ADAPTIVE_EFFICIENCY_BETA_MAX,
                "adaptive_efficiency_gamma": ADAPTIVE_EFFICIENCY_GAMMA,
                "adaptive_efficiency_solve_rate_floor": ADAPTIVE_EFFICIENCY_SOLVE_RATE_FLOOR,
                "efficiency_penalty_applies_to": EFFICIENCY_PENALTY_APPLIES_TO,
                "reward_clip_min": REWARD_CLIP_MIN,
                "reward_clip_max": REWARD_CLIP_MAX,
                "max_turn_penalty_enabled": str(MAX_TURN_PENALTY_ENABLED).lower(),
                "max_turn_penalty": MAX_TURN_PENALTY,
                "missing_final_at_max_turn_zero_reward": str(MISSING_FINAL_AT_MAX_TURN_ZERO_REWARD).lower(),
                "experiment_depth": EXPERIMENT_DEPTH,
                "runtime_max_depth": RUNTIME_MAX_DEPTH,
                "prompt_variant": PROMPT_VARIANT,
                "config_path": config_path.relative_to(ROOT).as_posix(),
                "output_dir": f"outputs/{run_id}",
                "dataset_tag": DATASET_TAG,
                "train_data_path": BALANCED_TRAIN_PATH,
                "eval_data_path": BALANCED_EVAL_PATH,
                "wandb_project": "rlm-rlvr",
                "wandb_name": run_id,
                "base_config": BASE_CONFIG.relative_to(ROOT).as_posix(),
                "status": "pending",
                "notes": "rank4_non_grouped_lora_fallback" if rank == 4 else "",
            }
        )

    fieldnames = list(rows[0].keys())
    with (OUT_DIR / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {len(rows)} configs and manifests under {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
