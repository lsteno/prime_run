from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
import random
import statistics
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RLM_ENV_ROOT = REPO_ROOT / "environments" / "rlm_rlvr"
if str(RLM_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(RLM_ENV_ROOT))


JUDGE_PROMPT = """You are grading whether a model answer is semantically correct.

Question:
{question}

Reference answer(s):
{expected_answers}

Model answer:
{predicted_answer}

Scoring rules:
- Return 1 if the model answer is mostly correct in meaning.
- Semantic correctness matters more than exact wording or format.
- Return 1 if the answer contains the correct fact, entity, or number even if formatting is imperfect.
- Return 1 if the answer adds extra harmless text but still clearly gives the correct answer.
- Return 0 only if the answer is completely incorrect, missing the core correct information, contradictory on the final answer, or gives no answer.
- The only valid outputs are 0 or 1.

Return exactly one character: 0 or 1.
"""

JUDGE_SYSTEM_PROMPT = "Return exactly one character: 0 or 1. Never return JSON, tool calls, or explanations."


@dataclass
class EndpointConfig:
    endpoint_id: str | None
    model: str
    url: str
    api_key_env: str | None
    api_key: str


@dataclass
class Example:
    example_id: int
    question: str
    answer: str
    acceptable_answers: list[str]
    context: str
    info: dict[str, Any]


@lru_cache(maxsize=1)
def load_dataset_fn():
    from datasets import load_dataset

    return load_dataset


@lru_cache(maxsize=1)
def load_openai_client():
    from openai import OpenAI

    return OpenAI


@lru_cache(maxsize=1)
def load_rlm_modules() -> dict[str, Any]:
    from rlm_rlvr.external_rlm import (
        CodeBlock,
        RLMIteration,
        build_system_prompt,
        build_user_prompt,
        find_code_blocks,
        find_final_answer,
        make_feedback_messages,
    )
    from rlm_rlvr.parsing import normalize_text, parse_answer_candidates
    from rlm_rlvr.repl import create_repl
    from rlm_rlvr.runtime import RecursiveRuntime, RuntimeConfig, SyncInferenceSession
    from rlm_rlvr.trace import make_segment

    return {
        "CodeBlock": CodeBlock,
        "RLMIteration": RLMIteration,
        "RecursiveRuntime": RecursiveRuntime,
        "RuntimeConfig": RuntimeConfig,
        "SyncInferenceSession": SyncInferenceSession,
        "build_system_prompt": build_system_prompt,
        "build_user_prompt": build_user_prompt,
        "create_repl": create_repl,
        "find_code_blocks": find_code_blocks,
        "find_final_answer": find_final_answer,
        "make_feedback_messages": make_feedback_messages,
        "make_segment": make_segment,
        "normalize_text": normalize_text,
        "parse_answer_candidates": parse_answer_candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RLM traces and compare prompt variants.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the pipeline TOML config.")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(path_str: str, *, relative_to: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (relative_to / path).resolve()
    return path


def load_endpoints(path: Path) -> dict[str, dict[str, Any]]:
    config = load_toml(path)
    endpoints = {}
    for record in config.get("endpoint", []):
        endpoint_id = str(record["endpoint_id"])
        endpoints[endpoint_id] = record
    return endpoints


def resolve_endpoint(spec: dict[str, Any], *, endpoints_path: Path) -> EndpointConfig:
    endpoint_id = spec.get("endpoint_id")
    if endpoint_id is not None:
        endpoints = load_endpoints(endpoints_path)
        if endpoint_id not in endpoints:
            raise ValueError(f"Unknown endpoint_id: {endpoint_id}")
        record = endpoints[endpoint_id]
        endpoint_type = record.get("type")
        if endpoint_type != "openai_chat_completions":
            raise ValueError(
                f"Endpoint {endpoint_id!r} has unsupported type {endpoint_type!r}. "
                "This pipeline currently supports only openai_chat_completions endpoints."
            )
        model = str(spec.get("model") or record["model"])
        url = str(spec.get("url") or record["url"])
        api_key_env = str(spec.get("api_key_env") or record.get("key") or "")
        api_key = os.environ.get(api_key_env, "EMPTY") if api_key_env else "EMPTY"
        return EndpointConfig(
            endpoint_id=str(endpoint_id),
            model=model,
            url=url,
            api_key_env=api_key_env or None,
            api_key=api_key,
        )

    model = str(spec["model"])
    url = str(spec["url"])
    api_key_env = spec.get("api_key_env")
    api_key = os.environ.get(str(api_key_env), "EMPTY") if api_key_env else "EMPTY"
    return EndpointConfig(
        endpoint_id=None,
        model=model,
        url=url,
        api_key_env=str(api_key_env) if api_key_env else None,
        api_key=api_key,
    )


def _extract_by_path(row: dict[str, Any], path: str) -> object:
    current: object = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _pick_first(row: dict[str, Any], keys: list[str]) -> object:
    for key in keys:
        value = _extract_by_path(row, key)
        if value is not None:
            return value
    return None


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def load_examples(dataset_cfg: dict[str, Any]) -> list[Example]:
    load_dataset = load_dataset_fn()
    parse_answer_candidates = load_rlm_modules()["parse_answer_candidates"]
    dataset_id = str(dataset_cfg["dataset_id"])
    split = str(dataset_cfg.get("split", "eval"))
    dataset_config = dataset_cfg.get("dataset_config")
    revision = dataset_cfg.get("dataset_revision")
    max_examples = int(dataset_cfg.get("max_examples", -1))
    seed = int(dataset_cfg.get("seed", 42))

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision

    if dataset_config:
        dataset = load_dataset(dataset_id, dataset_config, **kwargs)
    else:
        dataset = load_dataset(dataset_id, **kwargs)

    rows = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_examples > 0:
        rows = rows[:max_examples]

    examples: list[Example] = []
    for index, row in enumerate(rows):
        question = _stringify(_pick_first(row, ["question", "prompt", "query", "instruction", "task"])).strip()
        if not question:
            continue
        acceptable_answers = parse_answer_candidates(
            _pick_first(
                row,
                [
                    "acceptable_answers",
                    "answers",
                    "answer",
                    "target",
                    "expected_answer",
                    "solution",
                ],
            )
        )
        if not acceptable_answers:
            continue
        context = _stringify(
            _pick_first(row, ["context", "context_payload", "input", "document", "passage", "metadata.context"])
        )
        examples.append(
            Example(
                example_id=index,
                question=question,
                answer=acceptable_answers[0],
                acceptable_answers=acceptable_answers,
                context=context,
                info=dict(row),
            )
        )
    if not examples:
        raise ValueError("No usable examples were loaded from the dataset.")
    return examples


def is_exact_match(predicted_answer: str, expected_answers: list[str]) -> bool:
    normalize_text = load_rlm_modules()["normalize_text"]
    predicted = normalize_text(predicted_answer)
    return any(predicted == normalize_text(answer) for answer in expected_answers if str(answer).strip())


def format_expected_answers(answers: list[str]) -> str:
    return "\n".join(f"- {answer}" for answer in answers)


def parse_binary_judge_score(raw_text: str) -> float:
    text = raw_text.strip()
    if text in {"0", "1"}:
        return float(text)
    if "1" in text and "0" not in text:
        return 1.0
    if "0" in text:
        return 0.0
    raise ValueError(f"Judge response did not contain a valid binary score: {raw_text!r}")


def judge_answer(
    *,
    judge_endpoint: EndpointConfig,
    question: str,
    expected_answers: list[str],
    predicted_answer: str,
) -> tuple[float, str]:
    OpenAI = load_openai_client()
    client = OpenAI(base_url=judge_endpoint.url, api_key=judge_endpoint.api_key or "EMPTY")
    response = client.chat.completions.create(
        model=judge_endpoint.model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    expected_answers=format_expected_answers(expected_answers),
                    predicted_answer=predicted_answer,
                ),
            },
        ],
        temperature=0,
        max_tokens=8,
        extra_body={"reasoning": {"enabled": False}},
    )
    raw = (response.choices[0].message.content or "").strip()
    return parse_binary_judge_score(raw), raw


def init_state(
    *,
    endpoint: EndpointConfig,
    runtime_config: Any,
) -> tuple[dict[str, Any], Any, Any, Any]:
    modules = load_rlm_modules()
    SyncInferenceSession = modules["SyncInferenceSession"]
    RecursiveRuntime = modules["RecursiveRuntime"]
    create_repl = modules["create_repl"]
    state: dict[str, Any] = {
        "used_repl": False,
        "used_recursion": False,
        "max_depth_reached": 0,
        "num_subcalls": 0,
        "total_model_tokens": 0.0,
        "total_env_tokens": 0.0,
        "rlm_segments": [],
        "rlm_trace": [],
        "rlm_segment_counter": 0,
        "rlm_call_counter": 0,
        "current_call_depth": 0,
        "current_branch_max_depth": runtime_config.max_depth,
        "final_answer": None,
        "sampling_temperature": runtime_config.temperature,
    }
    session = SyncInferenceSession(
        base_url=endpoint.url,
        api_key=endpoint.api_key,
        default_headers=None,
        model_name=endpoint.model,
        tokenizer_name=runtime_config.tokenizer_name,
        max_prompt_tokens=runtime_config.max_prompt_tokens,
    )
    state["_sync_session"] = session
    runtime = RecursiveRuntime(state, runtime_config)
    state["_runtime"] = runtime
    repl = create_repl(
        backend=runtime_config.repl_backend,
        backend_kwargs=runtime_config.repl_backend_kwargs,
        context_payload="",
        llm_query_fn=runtime._plain_query,
        rlm_query_fn=runtime._recursive_query,
    )
    state["_root_repl"] = repl
    return state, session, runtime, repl


def run_rollout(
    *,
    example: Example,
    endpoint: EndpointConfig,
    runtime_config: Any,
    prompt_variant: str,
    judge_endpoint: EndpointConfig | None,
) -> dict[str, Any]:
    modules = load_rlm_modules()
    CodeBlock = modules["CodeBlock"]
    RLMIteration = modules["RLMIteration"]
    build_system_prompt = modules["build_system_prompt"]
    build_user_prompt = modules["build_user_prompt"]
    create_repl = modules["create_repl"]
    find_code_blocks = modules["find_code_blocks"]
    find_final_answer = modules["find_final_answer"]
    make_feedback_messages = modules["make_feedback_messages"]
    make_segment = modules["make_segment"]
    state, session, runtime, repl = init_state(endpoint=endpoint, runtime_config=runtime_config)
    state["_root_context"] = example.context
    if hasattr(repl, "_env"):
        pass
    state["_root_repl"] = create_repl(
        backend=runtime_config.repl_backend,
        backend_kwargs=runtime_config.repl_backend_kwargs,
        context_payload=example.context,
        llm_query_fn=runtime._plain_query,
        rlm_query_fn=runtime._recursive_query,
    )
    repl = state["_root_repl"]

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(
                depth=0,
                max_depth=runtime_config.max_depth,
                prompt_variant=prompt_variant,
            ),
        },
        {"role": "user", "content": example.question},
    ]
    root_steps: list[dict[str, Any]] = []
    forced_finalize = False
    error: str | None = None
    start_time = time.perf_counter()

    try:
        for assistant_turn in range(runtime_config.max_iterations + 1):
            assistant_text, payload = session.generate(
                messages=messages,
                max_tokens=runtime_config.turn_max_tokens,
                temperature=runtime_config.temperature,
                top_p=runtime_config.top_p,
            )

            segment_kind = "root_finalize_turn" if forced_finalize else "root_turn"
            segment = make_segment(
                order=int(state["rlm_segment_counter"]),
                depth=0,
                kind=segment_kind,
                prompt_ids=list(payload.prompt_ids),
                completion_ids=list(payload.completion_ids),
                completion_logprobs=[float(value) for value in payload.completion_logprobs],
                completion_mask=[bool(value) for value in payload.completion_mask],
                temperature=float(runtime_config.temperature),
                response_text=assistant_text,
            )
            state["rlm_segment_counter"] += 1
            state["rlm_segments"].append(segment)
            state["total_model_tokens"] += float(sum(segment["completion_mask"]))

            code_block_strs = [block.strip() for block in find_code_blocks(assistant_text)]
            code_blocks: list[CodeBlock] = []
            if code_block_strs:
                state["used_repl"] = True

            for code in code_block_strs:
                execution = repl.execute_code(code)
                code_blocks.append(CodeBlock(code=code, result=execution))
                for child_call in execution.rlm_calls:
                    metadata = child_call.metadata or {}
                    if metadata.get("kind") == "recursive_query":
                        state["used_recursion"] = True
                if execution.final_answer is not None and state.get("final_answer") is None:
                    state["final_answer"] = execution.final_answer

            if state.get("final_answer") is None:
                found = find_final_answer(assistant_text, environment=repl)
                if found is not None:
                    state["final_answer"] = found

            iteration = RLMIteration(prompt=messages, response=assistant_text, code_blocks=code_blocks)
            feedback_messages = make_feedback_messages(
                iteration,
                max_chars=runtime_config.execution_output_char_limit,
            )

            root_steps.append(
                {
                    "assistant": assistant_text,
                    "code_blocks": code_block_strs,
                    "feedback": [message["content"] for message in feedback_messages],
                    "final_answer": state.get("final_answer"),
                    "forced_finalize": forced_finalize,
                }
            )

            if state.get("final_answer") is not None:
                break

            if forced_finalize:
                state["final_answer"] = assistant_text.strip()
                root_steps[-1]["final_answer"] = state["final_answer"]
                break

            next_iteration = assistant_turn + 1
            if next_iteration >= runtime_config.max_iterations:
                next_message = {"role": "user", "content": runtime.build_finalize_message()}
                forced_finalize = True
            else:
                next_message = build_user_prompt(
                    root_prompt=example.question,
                    iteration=next_iteration,
                    context_count=int(repl.get_context_count()),
                    history_count=int(repl.get_history_count()),
                )

            response_messages = [*feedback_messages, next_message]
            state["total_env_tokens"] += float(
                sum(session.count_text_tokens(message["content"]) for message in response_messages)
            )
            messages.append({"role": "assistant", "content": assistant_text})
            messages.extend(response_messages)
        else:
            state["final_answer"] = ""
    except Exception as exc:  # pragma: no cover - defensive capture for real runs
        error = str(exc)
    finally:
        close = getattr(repl, "close", None)
        if callable(close):
            close()

    final_answer = "" if state.get("final_answer") is None else str(state["final_answer"]).strip()
    exact_match = is_exact_match(final_answer, example.acceptable_answers) if final_answer else False
    judge_score = None
    judge_raw_response = None
    if final_answer and judge_endpoint is not None and not exact_match:
        try:
            judge_score, judge_raw_response = judge_answer(
                judge_endpoint=judge_endpoint,
                question=example.question,
                expected_answers=example.acceptable_answers,
                predicted_answer=final_answer,
            )
        except Exception as exc:  # pragma: no cover - best-effort judging
            judge_raw_response = f"[judge_error] {exc}"

    return {
        "example_id": example.example_id,
        "question": example.question,
        "acceptable_answers": example.acceptable_answers,
        "answer": example.answer,
        "context_length": len(example.context),
        "prompt_variant": prompt_variant,
        "endpoint": asdict(endpoint),
        "final_answer": final_answer,
        "exact_match": exact_match,
        "judge_score": judge_score,
        "judge_raw_response": judge_raw_response,
        "used_repl": bool(state["used_repl"]),
        "used_recursion": bool(state["used_recursion"]),
        "num_subcalls": int(state["num_subcalls"]),
        "max_depth_reached": int(state["max_depth_reached"]),
        "total_model_tokens": float(state["total_model_tokens"]),
        "total_env_tokens": float(state["total_env_tokens"]),
        "root_steps": root_steps,
        "trace": state["rlm_trace"],
        "segments": state["rlm_segments"],
        "error": error,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    exact_matches = [1.0 if record["exact_match"] else 0.0 for record in records]
    token_counts = [float(record["total_model_tokens"]) for record in records]
    subcalls = [int(record["num_subcalls"]) for record in records]
    recursion = [1.0 if record["used_recursion"] else 0.0 for record in records]
    judge_scores = [record["judge_score"] for record in records if record["judge_score"] is not None]

    summary = {
        "num_records": len(records),
        "exact_match_rate": statistics.mean(exact_matches) if exact_matches else 0.0,
        "mean_total_model_tokens": statistics.mean(token_counts) if token_counts else 0.0,
        "mean_num_subcalls": statistics.mean(subcalls) if subcalls else 0.0,
        "used_recursion_rate": statistics.mean(recursion) if recursion else 0.0,
        "num_errors": sum(1 for record in records if record["error"]),
    }
    if judge_scores:
        summary["mean_judge_score"] = statistics.mean(judge_scores)
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_comparison_markdown(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Prompt Comparison",
        "",
        "| Prompt Variant | Exact Match | Judge Score | Mean Tokens | Mean Subcalls | Recursion Rate | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for prompt_variant, summary in sorted(results.items()):
        lines.append(
            "| {name} | {exact:.3f} | {judge} | {tokens:.1f} | {subcalls:.2f} | {recursion:.3f} | {errors} |".format(
                name=prompt_variant,
                exact=summary["exact_match_rate"],
                judge=f"{summary['mean_judge_score']:.3f}" if "mean_judge_score" in summary else "-",
                tokens=summary["mean_total_model_tokens"],
                subcalls=summary["mean_num_subcalls"],
                recursion=summary["used_recursion_rate"],
                errors=summary["num_errors"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_toml(config_path)
    RuntimeConfig = load_rlm_modules()["RuntimeConfig"]

    endpoints_path = resolve_path(
        str(config.get("endpoints_path", "configs/endpoints.toml")),
        relative_to=config_path.parent,
    )
    output_root = resolve_path(str(config.get("output_dir", "outputs/rlm_traces")), relative_to=config_path.parent)
    run_name = str(config.get("run_name", config_path.stem))
    run_dir = output_root / run_name

    model_endpoint = resolve_endpoint(config["model"], endpoints_path=endpoints_path)
    judge_cfg = config.get("judge")
    judge_endpoint = None
    if judge_cfg and bool(judge_cfg.get("enabled", False)):
        judge_endpoint = resolve_endpoint(judge_cfg, endpoints_path=endpoints_path)

    rollout_cfg = config["rollout"]
    runtime_config = RuntimeConfig(
        max_depth=int(rollout_cfg.get("max_depth", 2)),
        max_iterations=int(rollout_cfg.get("max_iterations", 4)),
        turn_max_tokens=int(rollout_cfg.get("turn_max_tokens", 192)),
        subcall_max_tokens=int(rollout_cfg.get("subcall_max_tokens", 128)),
        max_prompt_tokens=(
            int(rollout_cfg["max_prompt_tokens"]) if rollout_cfg.get("max_prompt_tokens") is not None else None
        ),
        temperature=float(rollout_cfg.get("temperature", 1.0)),
        top_p=float(rollout_cfg.get("top_p", 1.0)),
        tokenizer_name=rollout_cfg.get("tokenizer_name"),
        inference_mode="local",
        inference_base_url=model_endpoint.url,
        inference_api_key=model_endpoint.api_key,
        repl_backend=str(rollout_cfg.get("repl_backend", "local")),
        prompt_variant=str((rollout_cfg.get("prompt_variants") or ["default"])[0]),
    )
    prompt_variants = [str(item) for item in rollout_cfg.get("prompt_variants", ["default"])]

    examples = load_examples(config["dataset"])
    write_json(
        run_dir / "run_config.json",
        {
            "config_path": str(config_path),
            "model_endpoint": asdict(model_endpoint),
            "judge_endpoint": asdict(judge_endpoint) if judge_endpoint is not None else None,
            "prompt_variants": prompt_variants,
            "num_examples": len(examples),
        },
    )

    comparison: dict[str, dict[str, Any]] = {}
    for prompt_variant in prompt_variants:
        variant_runtime = RuntimeConfig(**asdict(runtime_config))
        variant_runtime.prompt_variant = prompt_variant
        variant_dir = run_dir / prompt_variant
        records_path = variant_dir / "records.jsonl"
        successful_path = variant_dir / "successful_records.jsonl"
        records: list[dict[str, Any]] = []
        successful: list[dict[str, Any]] = []
        write_jsonl(records_path, [])
        write_jsonl(successful_path, [])

        for example in examples:
            record = run_rollout(
                example=example,
                endpoint=model_endpoint,
                runtime_config=variant_runtime,
                prompt_variant=prompt_variant,
                judge_endpoint=judge_endpoint,
            )
            records.append(record)
            append_jsonl(records_path, record)
            if not record["error"]:
                successful.append(record)
                append_jsonl(successful_path, record)

            summary = summarize(records)
            comparison[prompt_variant] = summary
            write_json(variant_dir / "summary.json", summary)
            write_json(run_dir / "comparison.json", comparison)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "comparison.md").write_text(render_comparison_markdown(comparison))

    write_json(run_dir / "comparison.json", comparison)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "comparison.md").write_text(render_comparison_markdown(comparison))


if __name__ == "__main__":
    main()
