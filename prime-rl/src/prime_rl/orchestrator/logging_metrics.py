from __future__ import annotations

import pandas as pd

COMMON_STOP_CONDITIONS = (
    "has_final_env_response",
    "max_turns_reached",
    "rollout_timeout",
    "prompt_too_long",
    "generation_truncated",
)


def _numeric_metric_series(metrics_df: pd.DataFrame, metric: str, index: pd.Index) -> pd.Series:
    if metrics_df.empty or metric not in metrics_df.columns:
        return pd.Series(0.0, index=index, dtype="float64")
    return pd.to_numeric(metrics_df[metric], errors="coerce").reindex(index).fillna(0.0).astype("float64")


def _bool_series(series: pd.Series) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce").fillna(0.0) > 0.0).astype("float64")


def _mean_by_example(series: pd.Series, example_ids: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.groupby(example_ids).mean().mean())


def stable_stop_condition_metrics(results_df: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    stop_counts = results_df.stop_condition.dropna().value_counts()
    denominator = float(len(results_df)) if len(results_df) else 1.0

    for stop_condition in COMMON_STOP_CONDITIONS:
        key = f"stop_condition/{prefix}/{stop_condition}"
        if stop_condition == "generation_truncated":
            metrics[key] = float((results_df.is_truncated & (results_df.stop_condition != "prompt_too_long")).mean())
        else:
            metrics[key] = float(stop_counts.get(stop_condition, 0.0)) / denominator
    return metrics


def rlm_protocol_metrics(results_df: pd.DataFrame, metrics_df: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    index = results_df.index
    finalized_before = _bool_series(_numeric_metric_series(metrics_df, "finalized_before_forced_prompt_metric", index))
    finalized_on_forced = _bool_series(_numeric_metric_series(metrics_df, "finalized_on_forced_prompt_metric", index))
    hit_without_final = _bool_series(_numeric_metric_series(metrics_df, "hit_max_turn_without_final_metric", index))
    missing_final_metric = _bool_series(_numeric_metric_series(metrics_df, "missing_final_metric", index))
    forced_prompt = _bool_series(_numeric_metric_series(metrics_df, "used_forced_finalize_prompt_metric", index))
    max_turn_penalty = _numeric_metric_series(metrics_df, "max_turn_penalty_metric", index)

    missing_final = ((hit_without_final > 0.0) | (missing_final_metric > 0.0) | (results_df.stop_condition == "max_turns_reached")).astype(
        "float64"
    )
    formal_final = ((finalized_before > 0.0) | (finalized_on_forced > 0.0)).astype("float64")
    used_forced_prompt = ((forced_prompt > 0.0) | (finalized_on_forced > 0.0) | (hit_without_final > 0.0)).astype(
        "float64"
    )

    example_ids = results_df["example_id"]
    return {
        f"protocol/{prefix}/formal_final_rate": _mean_by_example(formal_final, example_ids),
        f"protocol/{prefix}/missing_final_rate": _mean_by_example(missing_final, example_ids),
        f"protocol/{prefix}/used_forced_finalize_prompt_rate": _mean_by_example(used_forced_prompt, example_ids),
        f"protocol/{prefix}/finalized_before_forced_rate": _mean_by_example(finalized_before, example_ids),
        f"protocol/{prefix}/finalized_on_forced_rate": _mean_by_example(finalized_on_forced, example_ids),
        f"protocol/{prefix}/max_turn_penalty_mean": _mean_by_example(max_turn_penalty, example_ids),
    }


def rlm_cost_subcall_metrics(results_df: pd.DataFrame, metrics_df: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    index = results_df.index
    example_ids = results_df["example_id"]
    metric_map = {
        "prompt_tokens_mean": "cost_prompt_tokens_metric",
        "completion_tokens_mean": "cost_completion_tokens_metric",
        "total_tokens_mean": "cost_total_tokens_metric",
        "trainable_tokens_mean": "cost_trainable_tokens_metric",
        "plain_subcall_tokens_mean": "cost_plain_subcall_tokens_metric",
        "adaptive_cost_penalty_mean": "adaptive_cost_penalty_metric",
        "incorrect_cost_penalty_mean": "incorrect_cost_penalty_metric",
        "adaptive_beta_mean": "adaptive_beta_metric",
        "adaptive_group_solve_rate_mean": "adaptive_group_solve_rate_metric",
        "llm_usage_rate": "used_llm_subcalls_metric",
        "rlm_usage_rate": "used_rlm_subcalls_metric",
        "repl_usage_rate": "used_repl_metric",
        "num_llm_mean": "num_llm_subcalls_metric",
        "num_rlm_mean": "num_rlm_subcalls_metric",
        "num_subcalls_mean": "num_subcalls_metric",
        "max_depth_mean": "max_depth_metric",
    }

    out: dict[str, float] = {}
    for name, metric in metric_map.items():
        namespace = "cost" if name.endswith("_tokens_mean") or name.startswith("adaptive_") else "subcalls"
        out[f"{namespace}/{prefix}/{name}"] = _mean_by_example(
            _numeric_metric_series(metrics_df, metric, index),
            example_ids,
        )
    return out
