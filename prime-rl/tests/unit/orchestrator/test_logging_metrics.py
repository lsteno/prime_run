import pandas as pd

from prime_rl.orchestrator.logging_metrics import (
    rlm_cost_subcall_metrics,
    rlm_protocol_metrics,
    rollout_curriculum_bucket,
    stable_stop_condition_metrics,
)


def test_rollout_curriculum_bucket_reads_safe_trace_metadata() -> None:
    rollout = {
        "trajectory": [
            {
                "extras": {
                    "rlm_debug": {
                        "sample_metadata": {"metadata": {"curriculum_bucket": "semantic_8"}}
                    }
                }
            }
        ]
    }

    assert rollout_curriculum_bucket(rollout, base_bucket="base") == "semantic_8"
    assert rollout_curriculum_bucket({}, base_bucket="base") == "base"


def test_stable_stop_condition_metrics_emit_zero_defaults() -> None:
    results_df = pd.DataFrame(
        {
            "example_id": [1, 2],
            "is_truncated": [False, False],
            "stop_condition": ["has_final_env_response", None],
        }
    )

    metrics = stable_stop_condition_metrics(results_df, prefix="all")

    assert metrics["stop_condition/all/has_final_env_response"] == 0.5
    assert metrics["stop_condition/all/max_turns_reached"] == 0.0
    assert metrics["stop_condition/all/rollout_timeout"] == 0.0
    assert metrics["stop_condition/all/prompt_too_long"] == 0.0
    assert metrics["stop_condition/all/generation_truncated"] == 0.0


def test_rlm_protocol_metrics_emit_zero_defaults_and_max_turn_fallback() -> None:
    results_df = pd.DataFrame(
        {
            "example_id": [1, 1, 2, 2],
            "stop_condition": ["has_final_env_response", "max_turns_reached", "has_final_env_response", None],
        }
    )
    metrics_df = pd.DataFrame(
        {
            "finalized_before_forced_prompt_metric": [1.0, 0.0, 0.0, 0.0],
            "finalized_on_forced_prompt_metric": [0.0, 0.0, 1.0, 0.0],
            "used_forced_finalize_prompt_metric": [0.0, 1.0, 1.0, 0.0],
            "max_turn_penalty_metric": [0.0, 0.0, 0.25, 0.0],
        }
    )

    metrics = rlm_protocol_metrics(results_df, metrics_df, prefix="all")

    assert metrics["protocol/all/formal_final_rate"] == 0.5
    assert metrics["protocol/all/missing_final_rate"] == 0.25
    assert metrics["protocol/all/used_forced_finalize_prompt_rate"] == 0.5
    assert metrics["protocol/all/finalized_before_forced_rate"] == 0.25
    assert metrics["protocol/all/finalized_on_forced_rate"] == 0.25
    assert metrics["protocol/all/max_turn_penalty_mean"] == 0.0625


def test_rlm_cost_subcall_metrics_emit_primary_namespaces() -> None:
    results_df = pd.DataFrame(
        {
            "example_id": [1, 2],
        }
    )
    metrics_df = pd.DataFrame(
        {
            "cost_total_tokens_metric": [100.0, 200.0],
            "cost_rlm_turn_tokens_metric": [80.0, 160.0],
            "cost_plain_subcall_tokens_metric": [20.0, 40.0],
            "cost_weighted_tokens_metric": [660.0, 1320.0],
            "adaptive_cost_penalty_metric": [0.0, 0.1],
            "used_llm_subcalls_metric": [1.0, 0.0],
            "num_llm_subcalls_metric": [3.0, 0.0],
        }
    )

    metrics = rlm_cost_subcall_metrics(results_df, metrics_df, prefix="all")

    assert metrics["cost/all/total_tokens_mean"] == 150.0
    assert metrics["cost/all/rlm_turn_tokens_mean"] == 120.0
    assert metrics["cost/all/plain_subcall_tokens_mean"] == 30.0
    assert metrics["cost/all/weighted_tokens_mean"] == 990.0
    assert metrics["cost/all/adaptive_cost_penalty_mean"] == 0.05
    assert metrics["subcalls/all/llm_usage_rate"] == 0.5
    assert metrics["subcalls/all/num_llm_mean"] == 1.5
    assert metrics["subcalls/all/rlm_usage_rate"] == 0.0
