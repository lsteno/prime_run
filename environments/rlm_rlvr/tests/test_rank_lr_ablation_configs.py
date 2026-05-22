from __future__ import annotations

import csv
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ABLATION_DIR = ROOT / "configs/rlm_rlvr/ablation_rank_lr"


def _load_manifest() -> list[dict[str, str]]:
    with (ABLATION_DIR / "manifest.csv").open(newline="") as handle:
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
        assert config["orchestrator"]["buffer"]["online_filter_easy"] is False
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
        assert row["llm_subcall_empty_response_max_attempts"] == "3"
        assert row["llm_subcall_empty_response_base_retry_seconds"] == "1.0"
        assert row["llm_subcall_empty_response_max_retry_seconds"] == "10.0"
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
        assert config["orchestrator"]["env"][0]["args"]["llm_subcall_empty_response_max_attempts"] == 3
        assert config["orchestrator"]["env"][0]["args"]["llm_subcall_empty_response_base_retry_seconds"] == 1.0
        assert config["orchestrator"]["env"][0]["args"]["llm_subcall_empty_response_max_retry_seconds"] == 10.0
        assert config["orchestrator"]["eval"]["env"][0]["args"]["llm_subcall_empty_response_max_attempts"] == 3
        assert config["orchestrator"]["eval"]["env"][0]["args"]["llm_subcall_empty_response_base_retry_seconds"] == 1.0
        assert config["orchestrator"]["eval"]["env"][0]["args"]["llm_subcall_empty_response_max_retry_seconds"] == 10.0
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
