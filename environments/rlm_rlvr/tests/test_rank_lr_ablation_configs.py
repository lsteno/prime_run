from __future__ import annotations

import csv
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ABLATION_DIR = ROOT / "configs/rlm_rlvr/ablation_rank_lr"
DEPTH1_H100_LORA_DIR = ROOT / "configs/rlm_rlvr/ablation_depth1_h100_lora_same_root"
FULL_FT_CONFIG = (
    ROOT
    / "configs/rlm_rlvr/full_ft/qwen3_4b_instruct_sanjaya_depth1_llmonly_fullft_lr1e-6_s150_8xa10080_bal35f40v1.toml"
)


def _load_manifest() -> list[dict[str, str]]:
    with (ABLATION_DIR / "manifest.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_depth1_h100_lora_manifest() -> list[dict[str, str]]:
    with (DEPTH1_H100_LORA_DIR / "manifest.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_rank_lr_ablation_manifest_has_expected_grid() -> None:
    rows = _load_manifest()
    assert len(rows) == 9
    observed = {(int(row["rank"]), int(row["alpha"]), row["lr"]) for row in rows}
    assert observed == {
        (4, 8, "5e-7"),
        (4, 8, "1e-5"),
        (4, 8, "1e-4"),
        (16, 32, "5e-7"),
        (16, 32, "1e-5"),
        (16, 32, "1e-4"),
        (64, 128, "5e-7"),
        (64, 128, "1e-5"),
        (64, 128, "1e-4"),
    }


def test_rank_lr_ablation_configs_match_manifest() -> None:
    for row in _load_manifest():
        config_path = ROOT / row["config_path"]
        config = tomllib.loads(config_path.read_text())

        assert config["max_steps"] == 150
        assert config["max_async_level"] == 2
        assert config["seq_len"] == 49152
        assert config["orchestrator"]["max_off_policy_steps"] == 4
        assert config["orchestrator"]["seq_len"] == 49152
        assert config["trainer"]["model"]["seq_len"] == 49152
        assert config["trainer"]["model"]["cp"] == 2
        assert config["inference"]["model"]["max_model_len"] == 49152
        assert config["output_dir"] == f"../{row['output_dir']}"
        assert config["wandb"]["project"] == "rlm-rlvr"
        assert config["wandb"]["name"] == row["wandb_name"]
        assert config["orchestrator"]["wandb"]["log_extras"]["interval"] == 10
        assert config["orchestrator"]["wandb"]["log_extras"]["sample_max_chars"] == 8000
        assert config["orchestrator"]["wandb"]["log_extras"]["sample_include_input_ids"] is False
        assert config["orchestrator"]["wandb"]["log_extras"]["final_samples"] is False

        assert config["trainer"]["optim"]["lr"] == float(row["lr"])
        assert config["trainer"]["model"]["lora"]["rank"] == int(row["rank"])
        assert config["trainer"]["model"]["lora"]["alpha"] == int(row["alpha"])
        assert config["trainer"]["model"]["lora"]["dropout"] == 0.0

        assert config["orchestrator"]["batch_size"] == 64
        assert config["orchestrator"]["rollouts_per_example"] == 4
        assert config["orchestrator"]["rollout_timeout_seconds"] == 400
        assert config["orchestrator"]["env_worker_recovery"]["enabled"] is True
        assert config["orchestrator"]["env_worker_recovery"]["cancel_grace_seconds"] == 5
        assert config["orchestrator"]["env_worker_recovery"]["max_rollout_attempts_per_slot"] == 4
        assert config["orchestrator"]["env_worker_recovery"]["max_attempts_cooldown_steps"] == 5
        assert config["orchestrator"]["env_worker_recovery"]["drop_group_on_first_timeout"] is False
        assert config["orchestrator"]["env_worker_recovery"]["first_timeout_cooldown_steps"] == 5
        assert config["orchestrator"]["env_worker_recovery"]["restart_on_rollout_timeout"] is True
        assert config["orchestrator"]["env_worker_recovery"]["restart_on_worker_health_failure"] is True
        assert config["orchestrator"]["attempt_logging"]["enabled"] is True
        assert config["orchestrator"]["async_scheduling"]["prefetch_next_batch"] is False
        assert config["orchestrator"]["async_scheduling"]["inflight_completion_cushion"] == 32
        assert config["orchestrator"]["async_scheduling"]["max_requests_per_env_worker"] == 1
        assert config["orchestrator"]["async_scheduling"]["max_cross_step_carryover"] == 32
        assert config["orchestrator"]["async_scheduling"]["max_carryover_steps"] == 1
        assert config["orchestrator"]["async_scheduling"]["cancel_stale_carryover"] is True
        assert config["orchestrator"]["async_scheduling"]["restart_workers_for_stale_cancel"] is True
        assert config["orchestrator"]["async_scheduling"]["batch_complete_cancel_grace_seconds"] == 2
        assert config["orchestrator"]["group_scoring"]["enabled"] is True
        assert config["orchestrator"]["group_scoring"]["max_concurrency"] == 8
        assert config["orchestrator"]["group_scoring"]["max_pending_groups"] == 64
        assert config["orchestrator"]["buffer"]["hard_cooldown_steps"] == 5
        assert "easy_threshold" not in config["orchestrator"]["buffer"]
        assert config["orchestrator"]["buffer"]["online_filter_hard"] is True
        assert config["orchestrator"]["buffer"]["online_filter_easy"] is True
        assert row["train_worker_count"] == "32"
        assert row["eval_worker_count"] == "16"
        assert row["rollout_timeout_seconds"] == "400"
        assert row["env_worker_cancel_grace_seconds"] == "5"
        assert row["max_rollout_attempts_per_slot"] == "4"
        assert row["max_attempts_cooldown_steps"] == "5"
        assert row["drop_group_on_first_timeout"] == "false"
        assert row["first_timeout_cooldown_steps"] == "5"
        assert row["wandb_log_extras_interval"] == "10"
        assert row["wandb_sample_max_chars"] == "8000"
        assert row["wandb_sample_include_input_ids"] == "false"
        assert row["wandb_final_samples"] == "false"
        assert row["repl_timeout_seconds"] == "300"
        assert row["hard_cooldown_steps"] == "5"
        assert row["prefetch_next_batch"] == "false"
        assert row["inflight_completion_cushion"] == "32"
        assert row["max_requests_per_env_worker"] == "1"
        assert row["max_cross_step_carryover"] == "32"
        assert row["max_carryover_steps"] == "1"
        assert row["cancel_stale_carryover"] == "true"
        assert row["restart_workers_for_stale_cancel"] == "true"
        assert row["batch_complete_cancel_grace_seconds"] == "2"
        assert row["group_scoring_enabled"] == "true"
        assert row["group_scoring_max_concurrency"] == "8"
        assert row["group_scoring_max_pending_groups"] == "64"
        assert row["seq_len"] == "49152"
        assert row["max_prompt_tokens"] == "47104"
        assert row["cp"] == "2"
        assert row["max_total_subcalls"] == "50"
        assert row["max_batched_subcalls"] == "50"
        assert row["subcall_batch_max_workers"] == "2"
        assert row["adaptive_efficiency_beta_max"] == "0.15"
        assert row["adaptive_efficiency_gamma"] == "1.0"
        assert row["adaptive_efficiency_solve_rate_floor"] == "0.25"
        assert row["efficiency_penalty_applies_to"] == "all_rollouts"
        assert row["reward_clip_min"] == "-0.5"
        assert row["reward_clip_max"] == "1.0"
        assert row["max_turn_penalty_enabled"] == "true"
        assert row["max_turn_penalty"] == "0.25"
        assert row["missing_final_at_max_turn_zero_reward"] == "true"
        assert row["max_async_level"] == "2"
        assert row["max_off_policy_steps"] == "4"
        assert row["experiment_depth"] == "1"
        assert row["runtime_max_depth"] == "0"
        assert row["prompt_variant"] == "sanjaya_text_depth1_llm_only_v1"
        assert config["orchestrator"]["env"][0]["worker_count"] == 32
        assert config["orchestrator"]["eval"]["env"][0]["worker_count"] == 16
        assert config["orchestrator"]["env"][0]["args"]["prompt_variant"] == "sanjaya_text_depth1_llm_only_v1"
        assert config["orchestrator"]["eval"]["env"][0]["args"]["prompt_variant"] == "sanjaya_text_depth1_llm_only_v1"
        assert config["orchestrator"]["env"][0]["args"]["efficiency_penalty_mode"] == "adaptive_group"
        assert config["orchestrator"]["env"][0]["args"]["adaptive_efficiency_beta_max"] == 0.15
        assert config["orchestrator"]["env"][0]["args"]["adaptive_efficiency_gamma"] == 1.0
        assert config["orchestrator"]["env"][0]["args"]["adaptive_efficiency_solve_rate_floor"] == 0.25
        assert config["orchestrator"]["env"][0]["args"]["efficiency_penalty_applies_to"] == "all_rollouts"
        assert config["orchestrator"]["env"][0]["args"]["reward_clip_min"] == -0.5
        assert config["orchestrator"]["env"][0]["args"]["reward_clip_max"] == 1.0
        assert config["orchestrator"]["env"][0]["args"]["max_iterations"] == 15
        assert config["orchestrator"]["env"][0]["args"]["max_depth"] == 0
        assert config["orchestrator"]["env"][0]["args"]["max_prompt_tokens"] == 47104
        assert config["orchestrator"]["env"][0]["args"]["repl_timeout_seconds"] == 300
        assert config["orchestrator"]["env"][0]["args"]["repl_fast_timeout_seconds"] == 30
        assert config["orchestrator"]["eval"]["env"][0]["args"]["max_depth"] == 0
        assert config["orchestrator"]["eval"]["env"][0]["args"]["max_prompt_tokens"] == 47104
        assert config["orchestrator"]["eval"]["env"][0]["args"]["repl_timeout_seconds"] == 300
        assert config["orchestrator"]["eval"]["env"][0]["args"]["repl_fast_timeout_seconds"] == 30
        assert config["orchestrator"]["env"][0]["args"]["max_total_subcalls"] == 50
        assert config["orchestrator"]["env"][0]["args"]["max_batched_subcalls"] == 50
        assert config["orchestrator"]["eval"]["env"][0]["args"]["max_total_subcalls"] == 50
        assert config["orchestrator"]["eval"]["env"][0]["args"]["max_batched_subcalls"] == 50
        for args in (config["orchestrator"]["env"][0]["args"], config["orchestrator"]["eval"]["env"][0]["args"]):
            assert args["subcall_batch_max_workers"] == 2
            assert "llm_subcall_provider" not in args
            assert "llm_subcall_model" not in args
            assert "llm_subcall_vertex_project_env" not in args
            assert "llm_subcall_vertex_location" not in args
            assert "llm_subcall_thinking_level" not in args
            assert "llm_subcall_empty_response_max_attempts" not in args
            assert "llm_subcall_empty_response_base_retry_seconds" not in args
            assert "llm_subcall_empty_response_max_retry_seconds" not in args
        assert config["orchestrator"]["env"][0]["args"]["max_turn_penalty_enabled"] is True
        assert config["orchestrator"]["env"][0]["args"]["max_turn_penalty"] == 0.25
        assert config["orchestrator"]["env"][0]["args"]["missing_final_at_max_turn_zero_reward"] is True
        assert config["orchestrator"]["eval"]["env"][0]["args"]["max_turn_penalty_enabled"] is True
        assert config["orchestrator"]["eval"]["env"][0]["args"]["max_turn_penalty"] == 0.25
        assert config["orchestrator"]["eval"]["env"][0]["args"]["missing_final_at_max_turn_zero_reward"] is True

        eval_args = config["orchestrator"]["eval"]["env"][0]["args"]
        assert eval_args["efficiency_penalty_mode"] == "static_per_1k"
        assert eval_args["efficiency_penalty_coef"] == 0.0
        assert eval_args["efficiency_penalty_applies_to"] == "all_rollouts"
        assert eval_args["reward_clip_min"] == -0.5
        assert eval_args["reward_clip_max"] == 1.0


def test_depth1_h100_lora_sweep_manifest_has_expected_grid() -> None:
    rows = _load_depth1_h100_lora_manifest()
    assert len(rows) == 9
    observed = {(int(row["rank"]), int(row["alpha"]), row["lr"]) for row in rows}
    assert observed == {
        (4, 8, "1e-6"),
        (4, 8, "1e-5"),
        (4, 8, "1e-4"),
        (16, 32, "1e-6"),
        (16, 32, "1e-5"),
        (16, 32, "1e-4"),
        (64, 128, "1e-6"),
        (64, 128, "1e-5"),
        (64, 128, "1e-4"),
    }


def test_depth1_h100_lora_sweep_configs_match_manifest() -> None:
    for row in _load_depth1_h100_lora_manifest():
        config_path = ROOT / row["config_path"]
        config = tomllib.loads(config_path.read_text())

        assert config["output_dir"] == f"../{row['output_dir']}"
        assert config["max_steps"] == 150
        assert config["max_async_level"] == 2
        assert config["seq_len"] == 49152
        assert config["deployment"]["gpus_per_node"] == 8
        assert config["deployment"]["num_infer_gpus"] == 4
        assert config["deployment"]["num_train_gpus"] == 4
        assert config["model"]["name"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert config["wandb"]["project"] == "rlm-rlvr"
        assert config["wandb"]["name"] == row["wandb_name"]
        assert "weight_broadcast" not in config
        assert config["ckpt"] == {"interval": 25, "keep_last": 2, "keep_interval": 100}

        assert config["trainer"]["optim"]["lr"] == float(row["lr"])
        assert config["trainer"]["model"]["seq_len"] == 49152
        assert config["trainer"]["model"]["cp"] == 2
        assert config["trainer"]["model"]["lora"]["rank"] == int(row["rank"])
        assert config["trainer"]["model"]["lora"]["alpha"] == int(row["alpha"])
        assert config["trainer"]["model"]["lora"]["dropout"] == 0.0

        orchestrator = config["orchestrator"]
        assert orchestrator["batch_size"] == 64
        assert orchestrator["rollouts_per_example"] == 4
        assert orchestrator["max_concurrent"] == 32
        assert orchestrator["rollout_timeout_seconds"] == 400
        assert orchestrator["seq_len"] == 49152
        assert orchestrator["max_off_policy_steps"] == 4
        assert orchestrator["client"]["base_url"] == ["http://localhost:8001/v1"]
        assert orchestrator["async_scheduling"]["prefetch_next_batch"] is False
        assert orchestrator["async_scheduling"]["inflight_completion_cushion"] == 32
        assert orchestrator["async_scheduling"]["max_requests_per_env_worker"] == 1
        assert orchestrator["async_scheduling"]["max_cross_step_carryover"] == 32
        assert orchestrator["async_scheduling"]["max_carryover_steps"] == 1
        assert orchestrator["async_scheduling"]["restart_workers_for_stale_cancel"] is True
        assert orchestrator["group_scoring"]["enabled"] is True
        assert orchestrator["group_scoring"]["max_concurrency"] == 8
        assert orchestrator["group_scoring"]["max_pending_groups"] == 64
        assert orchestrator["wandb"]["log_extras"]["interval"] == 10
        assert orchestrator["wandb"]["log_extras"]["sample_max_chars"] == 8000

        assert row["base_config"] == (
            "configs/rlm_rlvr/local/"
            "qwen3_4b_instruct_sanjaya_depth2_recursive_r064_a128_lr1e-5_s150_8xh100_bal35f40v1.toml"
        )
        assert row["max_steps"] == "150"
        assert row["max_async_level"] == "2"
        assert row["max_off_policy_steps"] == "4"
        assert row["batch_size"] == "64"
        assert row["rollouts_per_example"] == "4"
        assert row["num_infer_gpus"] == "4"
        assert row["num_train_gpus"] == "4"
        assert row["inference_dp"] == "4"
        assert row["train_worker_count"] == "32"
        assert row["eval_worker_count"] == "16"
        assert row["max_total_subcalls"] == "50"
        assert row["max_batched_subcalls"] == "50"
        assert row["subcall_batch_max_workers"] == "2"
        assert row["experiment_depth"] == "1"
        assert row["runtime_max_depth"] == "0"
        assert row["prompt_variant"] == "sanjaya_text_depth1_llm_only_v1"
        assert row["train_data_path"] == "../data/beeg_agents_balanced_35_40_25_frames40_v1/train.parquet"
        assert row["eval_data_path"] == "../data/beeg_agents_balanced_35_40_25_frames40_v1/eval.parquet"

        assert orchestrator["env"][0]["worker_count"] == 32
        assert orchestrator["eval"]["env"][0]["worker_count"] == 16
        expected_train_paths = ["../data/beeg_agents_balanced_35_40_25_frames40_v1/train.parquet"]
        expected_eval_paths = ["../data/beeg_agents_balanced_35_40_25_frames40_v1/eval.parquet"]
        for args in (orchestrator["env"][0]["args"], orchestrator["eval"]["env"][0]["args"]):
            assert args["data_paths"] == expected_train_paths
            assert args["eval_data_paths"] == expected_eval_paths
            assert args["prompt_variant"] == "sanjaya_text_depth1_llm_only_v1"
            assert args["max_depth"] == 0
            assert "recursive_cap_prompt_variant" not in args
            assert "recursive_rlm_batch_mode" not in args
            assert args["subcall_budget_enabled"] is True
            assert args["max_total_subcalls"] == 50
            assert args["max_batched_subcalls"] == 50
            assert args["subcall_batch_max_workers"] == 2
            assert args["judge_provider"] == "vertex"
            assert args["judge_model"] == "gemini-3-flash-preview"
            assert args["inference_base_url"] == "http://localhost:8001/v1"
            assert "llm_subcall_provider" not in args
            assert "llm_subcall_model" not in args
            assert "llm_subcall_vertex_project_env" not in args
            assert "llm_subcall_vertex_location" not in args
            assert "llm_subcall_thinking_level" not in args

        inference = config["inference"]
        assert inference["api_server_count"] == 1
        assert inference["server"]["port"] == 8001
        assert inference["model"]["max_model_len"] == 49152
        assert inference["parallel"]["dp"] == 4
        assert inference["parallel"]["tp"] == 1
        assert inference["deployment"]["gpus_per_node"] == 4


def test_full_ft_pilot_config_removes_lora_and_uses_nccl_broadcast() -> None:
    config = tomllib.loads(FULL_FT_CONFIG.read_text())

    assert config["output_dir"] == "../outputs/rlm-rlvr-qwen3-4b-depth1-llmonly-fullft-lr1e-6-s150-bal35f40v1"
    assert config["max_steps"] == 150
    assert config["max_async_level"] == 1
    assert config["weight_broadcast"]["type"] == "nccl"
    assert config["weight_broadcast"]["port"] == 29501
    assert config["weight_broadcast"]["timeout"] == 1200

    assert config["trainer"]["optim"]["lr"] == 1e-6
    assert config["trainer"]["optim"]["type"] == "adamw"
    assert config["trainer"]["optim"]["weight_decay"] == 0.0
    assert config["trainer"]["optim"]["max_norm"] == 1.0
    assert config["trainer"]["scheduler"]["type"] == "constant"
    assert "lora" not in config["trainer"]["model"]
    assert config["trainer"]["model"]["seq_len"] == 49152
    assert config["trainer"]["model"]["cp"] == 2
    assert config["trainer"]["model"]["tp"] == 1
    assert config["trainer"]["model"]["dp_replicate"] == 1
    assert config["trainer"]["model"]["optimization_dtype"] == "bfloat16"
    assert config["trainer"]["model"]["reduce_dtype"] == "bfloat16"
    assert config["trainer"]["model"]["ac"]["freq"] == 2
    assert config["trainer"]["loss"]["kl_tau"] == 1e-3

    assert config["orchestrator"]["batch_size"] == 64
    assert config["orchestrator"]["rollouts_per_example"] == 4
    assert config["orchestrator"]["max_off_policy_steps"] == 2
    assert config["orchestrator"]["env_worker_recovery"]["drop_group_on_first_timeout"] is False
    assert config["orchestrator"]["group_scoring"]["enabled"] is True
    assert config["orchestrator"]["env"][0]["worker_count"] == 32
    assert config["orchestrator"]["eval"]["env"][0]["worker_count"] == 16

    train_args = config["orchestrator"]["env"][0]["args"]
    eval_args = config["orchestrator"]["eval"]["env"][0]["args"]
    assert train_args["prompt_variant"] == "sanjaya_text_depth1_llm_only_v1"
    assert train_args["max_depth"] == 0
    assert train_args["efficiency_penalty_mode"] == "adaptive_group"
    assert train_args["adaptive_efficiency_beta_max"] == 0.15
    assert train_args["adaptive_efficiency_gamma"] == 1.0
    assert train_args["adaptive_efficiency_solve_rate_floor"] == 0.25
    assert train_args["efficiency_penalty_applies_to"] == "correct_only"
    assert train_args["max_turn_penalty_enabled"] is True
    assert train_args["missing_final_at_max_turn_zero_reward"] is True
    assert eval_args["efficiency_penalty_applies_to"] == "correct_only"
    for args in (train_args, eval_args):
        assert args["subcall_batch_max_workers"] == 2
        assert "llm_subcall_provider" not in args
        assert "llm_subcall_model" not in args
        assert "llm_subcall_vertex_project_env" not in args

    inference = config["inference"]
    assert "max_loras" not in inference
    assert "max_cpu_loras" not in inference
    assert inference["api_server_count"] == 4
    assert config["inference"]["parallel"]["dp"] == 4
    assert config["inference"]["parallel"]["tp"] == 1
