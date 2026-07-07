#!/usr/bin/env python3
"""Generate depth-1 same-root LoRA sweep configs from the H100 depth-2 shape."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    ROOT
    / "configs/rlm_rlvr/local/qwen3_4b_instruct_sanjaya_depth2_recursive_r064_a128_lr1e-5_s150_8xh100_bal35f40v1.toml"
)
OUT_DIR = ROOT / "configs/rlm_rlvr/ablation_depth1_h100_lora_same_root"
PROMPT_VARIANT = "sanjaya_text_depth1_llm_only_v1"
EXPERIMENT_DEPTH = 1
RUNTIME_MAX_DEPTH = 0
MAX_STEPS = 150
INFERENCE_PORT = 8001
INFERENCE_GPU_COUNT = 4
TRAIN_GPU_COUNT = 4
MAX_CONCURRENT = 32
TRAIN_WORKER_COUNT = 32
EVAL_WORKER_COUNT = 16
MAX_CROSS_STEP_CARRYOVER = 32
RESTART_WORKERS_FOR_STALE_CANCEL = "true"
RUN_SUFFIX = "bal35f40v1"
BALANCED_DATA_DIR_FROM_PRIME_RL = "../data/beeg_agents_balanced_35_40_25_frames40_v1"
BALANCED_TRAIN_PATH = f"{BALANCED_DATA_DIR_FROM_PRIME_RL}/train.parquet"
BALANCED_EVAL_PATH = f"{BALANCED_DATA_DIR_FROM_PRIME_RL}/eval.parquet"
MATRIX = (
    (4, 8, "1e-6"),
    (4, 8, "1e-5"),
    (4, 8, "1e-4"),
    (16, 32, "1e-6"),
    (16, 32, "1e-5"),
    (16, 32, "1e-4"),
    (64, 128, "1e-6"),
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


def remove_line(text: str, pattern: str) -> str:
    text, count = re.subn(pattern, "", text, flags=re.MULTILINE)
    if count == 0:
        raise RuntimeError(f"Expected at least one line removal for pattern: {pattern}")
    return text


def config_filename(rank: int, alpha: int, lr: str) -> str:
    return f"qwen3_4b_instruct_sanjaya_depth1_llmonly_r{rank:03d}_a{alpha:03d}_lr{lr}_s150_8xh100_{RUN_SUFFIX}.toml"


def run_id_for(rank: int, alpha: int, lr: str) -> str:
    return f"rlm-rlvr-qwen3-4b-depth1-llmonly-h100-r{rank}-a{alpha}-lr{lr}-s150-{RUN_SUFFIX}"


def strip_template_header(text: str) -> str:
    return text[text.index("output_dir = ") :]


def build_config(base: str, *, rank: int, alpha: int, lr: str, run_id: str) -> str:
    output_dir = f"../outputs/{run_id}"
    text = strip_template_header(base)
    text = replace_line(text, r'^output_dir = ".+"$', f'output_dir = "{output_dir}"')
    text = replace_line(text, r"^max_steps = \d+$", f"max_steps = {MAX_STEPS}")
    text = replace_line(text, r'^name = "rlm-rlvr-qwen3-4b-depth2-recursive-r64-a128-lr1e-5-s150-bal35f40v1"$', f'name = "{run_id}"')
    text = replace_line(text, r"^num_infer_gpus = \d+$", f"num_infer_gpus = {INFERENCE_GPU_COUNT}")
    text = replace_line(text, r"^num_train_gpus = \d+$", f"num_train_gpus = {TRAIN_GPU_COUNT}")
    text = replace_line(text, r"^lr = .+$", f"lr = {lr}")
    text = replace_line(text, r"^rank = \d+$", f"rank = {rank}")
    text = replace_line(text, r"^alpha = \d+$", f"alpha = {alpha}")
    text = replace_line(text, r"^max_concurrent = \d+$", f"max_concurrent = {MAX_CONCURRENT}")
    text = replace_line(
        text,
        r"^max_cross_step_carryover = \d+$",
        f"max_cross_step_carryover = {MAX_CROSS_STEP_CARRYOVER}",
    )
    text = replace_line(
        text,
        r"^restart_workers_for_stale_cancel = (true|false)$",
        f"restart_workers_for_stale_cancel = {RESTART_WORKERS_FOR_STALE_CANCEL}",
    )
    text = replace_line(text, r"^worker_count = 48$", f"worker_count = {TRAIN_WORKER_COUNT}")
    text = replace_line(text, r"^worker_count = 8$", f"worker_count = {EVAL_WORKER_COUNT}")
    text = replace_all(
        text,
        "max_off_policy_steps = 4\n\n[orchestrator.env_worker_recovery]",
        "max_off_policy_steps = 4\n\n"
        "[orchestrator.client]\n"
        f'base_url = ["http://localhost:{INFERENCE_PORT}/v1"]\n\n'
        "[orchestrator.env_worker_recovery]",
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
    text = replace_all(text, "max_depth = 1", f"max_depth = {RUNTIME_MAX_DEPTH}")
    text = replace_all(
        text,
        'inference_base_url = "http://localhost:8000/v1"',
        f'inference_base_url = "http://localhost:{INFERENCE_PORT}/v1"',
    )
    text = replace_line(text, r"^port = 8000$", f"port = {INFERENCE_PORT}")
    text = replace_line(text, r"^dp = 4$", f"dp = {INFERENCE_GPU_COUNT}")
    text = replace_line(text, r"^gpus_per_node = 4$", f"gpus_per_node = {INFERENCE_GPU_COUNT}")
    text = remove_line(text, r'^recursive_cap_prompt_variant = ".+"\n')
    text = remove_line(text, r'^recursive_rlm_batch_mode = ".+"\n')
    text = replace_all(
        text,
        'live_trace_dir = "../outputs/rlm-rlvr-qwen3-4b-depth2-recursive-r64-a128-lr1e-5-s150-bal35f40v1/live_traces"',
        f'live_trace_dir = "{output_dir}/live_traces"',
    )
    text = replace_all(
        text,
        'live_trace_dir = "../outputs/rlm-rlvr-qwen3-4b-depth2-recursive-r64-a128-lr1e-5-s150-bal35f40v1/live_traces_eval"',
        f'live_trace_dir = "{output_dir}/live_traces_eval"',
    )
    text = replace_line(text, r"^online_filter_easy = (true|false)$", "online_filter_easy = true")

    header = (
        "# Generated by scripts/rlm_rlvr/generate_depth1_h100_lora_sweep_configs.py.\n"
        "# Keep these as separate runs so LoRA rank is a true trainer-side ablation.\n"
        "# Depth-1 LLM-only setup: root policy can call plain llm_query* subcalls; recursive child RLMs are disabled.\n"
        "# Plain llm_query* subcalls reuse the root policy endpoint; Vertex is used only for judging.\n"
        "# Template shape: latest 8xH100 depth-2 run config, with depth/runtime knobs changed to depth 1.\n"
        "# Stable LoRA/depth-2 setup: 4 inference / 4 train with full checkpointing enabled.\n"
        f"# Sweep parameters: rank={rank}, alpha={alpha}, lr={lr}, max_steps={MAX_STEPS}.\n\n"
    )
    return header + text


def main() -> None:
    base = BASE_CONFIG.read_text()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_config in OUT_DIR.glob("qwen3_4b_instruct_sanjaya_depth1_llmonly_r*_s150_8xh100_*.toml"):
        stale_config.unlink()

    rows: list[dict[str, str | int]] = []
    for index, (rank, alpha, lr) in enumerate(MATRIX, start=1):
        run_id = run_id_for(rank, alpha, lr)
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
                "max_steps": MAX_STEPS,
                "max_async_level": 2,
                "max_off_policy_steps": 4,
                "hard_cooldown_steps": 5,
                "seed": 42,
                "batch_size": 64,
                "rollouts_per_example": 4,
                "num_infer_gpus": INFERENCE_GPU_COUNT,
                "num_train_gpus": TRAIN_GPU_COUNT,
                "inference_dp": INFERENCE_GPU_COUNT,
                "train_worker_count": TRAIN_WORKER_COUNT,
                "eval_worker_count": EVAL_WORKER_COUNT,
                "rollout_timeout_seconds": 400,
                "repl_timeout_seconds": 300,
                "max_total_subcalls": 50,
                "max_batched_subcalls": 50,
                "subcall_batch_max_workers": 2,
                "experiment_depth": EXPERIMENT_DEPTH,
                "runtime_max_depth": RUNTIME_MAX_DEPTH,
                "prompt_variant": PROMPT_VARIANT,
                "config_path": config_path.relative_to(ROOT).as_posix(),
                "output_dir": f"outputs/{run_id}",
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
