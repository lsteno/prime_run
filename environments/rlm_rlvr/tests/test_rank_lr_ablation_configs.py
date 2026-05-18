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
        assert config["orchestrator"]["env_worker_recovery"]["restart_on_rollout_timeout"] is True
        assert config["orchestrator"]["env_worker_recovery"]["restart_on_worker_health_failure"] is True
        assert config["orchestrator"]["attempt_logging"]["enabled"] is True
        assert config["orchestrator"]["buffer"]["hard_cooldown_steps"] == 5
        assert row["train_worker_count"] == "10"
        assert row["eval_worker_count"] == "2"
        assert row["rollout_timeout_seconds"] == "400"
        assert row["env_worker_cancel_grace_seconds"] == "5"
        assert row["max_rollout_attempts_per_slot"] == "4"
        assert row["max_attempts_cooldown_steps"] == "5"
        assert row["repl_timeout_seconds"] == "300"
        assert row["hard_cooldown_steps"] == "5"
        assert row["seq_len"] == "49152"
        assert row["max_prompt_tokens"] == "47104"
        assert row["cp"] == "2"
        assert row["max_total_subcalls"] == "50"
        assert row["max_batched_subcalls"] == "50"
        assert row["adaptive_efficiency_beta_max"] == "0.15"
        assert row["adaptive_efficiency_gamma"] == "1.0"
        assert row["adaptive_efficiency_solve_rate_floor"] == "0.25"
        assert row["max_turn_penalty_enabled"] == "true"
        assert row["max_turn_penalty"] == "0.25"
        assert row["missing_final_at_max_turn_zero_reward"] == "true"
        assert row["max_async_level"] == "2"
        assert row["max_off_policy_steps"] == "4"
        assert row["experiment_depth"] == "1"
        assert row["runtime_max_depth"] == "0"
        assert row["prompt_variant"] == "sanjaya_text_depth1_llm_only_v1"
        assert config["orchestrator"]["env"][0]["worker_count"] == 10
        assert config["orchestrator"]["eval"]["env"][0]["worker_count"] == 2
        assert config["orchestrator"]["env"][0]["args"]["prompt_variant"] == "sanjaya_text_depth1_llm_only_v1"
        assert config["orchestrator"]["eval"]["env"][0]["args"]["prompt_variant"] == "sanjaya_text_depth1_llm_only_v1"
        assert config["orchestrator"]["env"][0]["args"]["efficiency_penalty_mode"] == "adaptive_group"
        assert config["orchestrator"]["env"][0]["args"]["adaptive_efficiency_beta_max"] == 0.15
        assert config["orchestrator"]["env"][0]["args"]["adaptive_efficiency_gamma"] == 1.0
        assert config["orchestrator"]["env"][0]["args"]["adaptive_efficiency_solve_rate_floor"] == 0.25
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
        assert config["orchestrator"]["env"][0]["args"]["max_turn_penalty_enabled"] is True
        assert config["orchestrator"]["env"][0]["args"]["max_turn_penalty"] == 0.25
        assert config["orchestrator"]["env"][0]["args"]["missing_final_at_max_turn_zero_reward"] is True
        assert config["orchestrator"]["eval"]["env"][0]["args"]["max_turn_penalty_enabled"] is True
        assert config["orchestrator"]["eval"]["env"][0]["args"]["max_turn_penalty"] == 0.25
        assert config["orchestrator"]["eval"]["env"][0]["args"]["missing_final_at_max_turn_zero_reward"] is True

        eval_args = config["orchestrator"]["eval"]["env"][0]["args"]
        assert eval_args["efficiency_penalty_mode"] == "static_per_1k"
        assert eval_args["efficiency_penalty_coef"] == 0.0
