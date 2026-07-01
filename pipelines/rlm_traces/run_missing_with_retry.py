from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.rlm_traces import run as trace_run


DEFAULT_VARIANT = "sanjaya_text_depth1_llm_only_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate missing SFT traces with GPT-5.4 and retry failures with GPT-5.5.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the missing-trace driver TOML config.")
    parser.add_argument("--dry-run", action="store_true", help="Compute IDs and write no files or trace records.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke-testing the missing source IDs.")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(path_str: str, *, relative_to: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    local_path = (relative_to / path).resolve()
    repo_path = (REPO_ROOT / path).resolve()
    return local_path if local_path.exists() else repo_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_id_from_record(record: dict[str, Any]) -> str:
    return str(record.get("source_id", record.get("example_id", "")))


def latest_record_by_source(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        source_id = source_id_from_record(record)
        if source_id:
            latest[source_id] = record
    return latest


def collect_attempted_source_ids(record_paths: list[Path]) -> set[str]:
    attempted: set[str] = set()
    for path in record_paths:
        for record in read_jsonl(path):
            source_id = source_id_from_record(record)
            if source_id:
                attempted.add(source_id)
    return attempted


def record_is_good_trace(record: dict[str, Any]) -> bool:
    if record.get("error"):
        return False
    if not str(record.get("final_answer") or "").strip():
        return False
    if not (bool(record.get("exact_match")) or record.get("judge_score") == 1.0):
        return False
    return int(record.get("num_llm_subcalls") or 0) > 0


def failed_source_ids(records: list[dict[str, Any]], expected_source_ids: list[str]) -> list[str]:
    latest = latest_record_by_source(records)
    return [source_id for source_id in expected_source_ids if not record_is_good_trace(latest.get(source_id, {}))]


def load_dataset_source_ids(dataset_cfg: dict[str, Any]) -> list[str]:
    examples = trace_run.load_examples(dataset_cfg)
    return [str(example.source_id) for example in examples]


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_array(values: list[str]) -> str:
    if not values:
        return "[]"
    lines = ["["]
    lines.extend(f"  {_toml_string(value)}," for value in values)
    lines.append("]")
    return "\n".join(lines)


def render_trace_config(
    *,
    run_name: str,
    output_dir: Path,
    endpoints_path: str,
    model_cfg: dict[str, Any],
    llm_subcall_cfg: dict[str, Any] | None,
    dataset_cfg: dict[str, Any],
    rollout_cfg: dict[str, Any],
    judge_cfg: dict[str, Any],
    source_ids: list[str],
    max_workers: int,
    worker_backend: str,
    resume: bool,
) -> str:
    reasoning_enabled = bool(model_cfg.get("extra_body", {}).get("reasoning", {}).get("enabled", False))
    prompt_variants = [str(item) for item in rollout_cfg.get("prompt_variants", [DEFAULT_VARIANT])]
    llm_subcall_section = ""
    if llm_subcall_cfg and str(llm_subcall_cfg.get("provider", "")).lower() not in {
        "same_as_root",
        "root_policy",
        "root",
    }:
        if not llm_subcall_cfg.get("model"):
            raise ValueError("llm_subcall_cfg requires an explicit model when separate subcalls are enabled.")
        llm_subcall_section = f"""
[llm_subcall]
provider = {_toml_string(str(llm_subcall_cfg.get("provider", "vertex")))}
model = {_toml_string(str(llm_subcall_cfg["model"]))}
vertex_project_env = {_toml_string(str(llm_subcall_cfg.get("vertex_project_env", "GOOGLE_CLOUD_PROJECT")))}
vertex_location = {_toml_string(str(llm_subcall_cfg.get("vertex_location", "global")))}
thinking_level = {_toml_string(str(llm_subcall_cfg.get("thinking_level", "medium")))}
empty_response_max_attempts = {int(llm_subcall_cfg.get("empty_response_max_attempts", 3))}
empty_response_base_retry_seconds = {float(llm_subcall_cfg.get("empty_response_base_retry_seconds", 1.0))}
empty_response_max_retry_seconds = {float(llm_subcall_cfg.get("empty_response_max_retry_seconds", 30.0))}
"""
    return f"""run_name = {_toml_string(run_name)}
output_dir = {_toml_string(str(output_dir))}
endpoints_path = {_toml_string(endpoints_path)}
resume = {str(bool(resume)).lower()}
max_workers = {int(max_workers)}
worker_backend = {_toml_string(worker_backend)}

[model]
model = {_toml_string(str(model_cfg["model"]))}
url = {_toml_string(str(model_cfg["url"]))}
api_key_env = {_toml_string(str(model_cfg["api_key_env"]))}

[model.extra_body.reasoning]
enabled = {str(reasoning_enabled).lower()}
{llm_subcall_section}

[dataset]
dataset_id = {_toml_string(str(dataset_cfg["dataset_id"]))}
split = {_toml_string(str(dataset_cfg.get("split", "sft_traces")))}
seed = {int(dataset_cfg.get("seed", 42))}
source_ids = {_toml_array(source_ids)}

[rollout]
prompt_variants = {_toml_array(prompt_variants)}
max_iterations = {int(rollout_cfg.get("max_iterations", 15))}
max_depth = {int(rollout_cfg.get("max_depth", 1))}
disable_recursive_subcalls = {str(bool(rollout_cfg.get("disable_recursive_subcalls", True))).lower()}
turn_max_tokens = {int(rollout_cfg.get("turn_max_tokens", 4096))}
subcall_max_tokens = {int(rollout_cfg.get("subcall_max_tokens", 4096))}
max_prompt_tokens = {int(rollout_cfg.get("max_prompt_tokens", 2000000))}
temperature = {float(rollout_cfg.get("temperature", 0.7))}
top_p = {float(rollout_cfg.get("top_p", 1.0))}
repl_backend = {_toml_string(str(rollout_cfg.get("repl_backend", "local")))}
subcall_budget_enabled = {str(bool(rollout_cfg.get("subcall_budget_enabled", True))).lower()}
max_total_subcalls = {int(rollout_cfg.get("max_total_subcalls", 80))}
max_batched_subcalls = {int(rollout_cfg.get("max_batched_subcalls", 80))}
subcall_batch_max_workers = {int(rollout_cfg["subcall_batch_max_workers"]) if rollout_cfg.get("subcall_batch_max_workers") is not None else 8}
include_budget_reminder = {str(bool(rollout_cfg.get("include_budget_reminder", False))).lower()}
tokenizer_name = {_toml_string(str(rollout_cfg.get("tokenizer_name", "Qwen/Qwen3-4B-Instruct-2507")))}

[judge]
enabled = {str(bool(judge_cfg.get("enabled", True))).lower()}
provider = {_toml_string(str(judge_cfg.get("provider", "vertex")))}
model = {_toml_string(str(judge_cfg.get("model", "gemini-3-flash-preview")))}
vertex_project_env = {_toml_string(str(judge_cfg.get("vertex_project_env", "GOOGLE_CLOUD_PROJECT")))}
vertex_location = {_toml_string(str(judge_cfg.get("vertex_location", "global")))}
thinking_level = {_toml_string(str(judge_cfg.get("thinking_level", "medium")))}
"""


def run_trace_config(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "pipelines" / "rlm_traces" / "run.py"), "--config", str(config_path)],
        cwd=REPO_ROOT,
        check=True,
    )


def records_path(output_dir: Path, run_name: str, variant: str) -> Path:
    return output_dir / run_name / variant / "records.jsonl"


def render_report(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Missing SFT Trace Generation",
            "",
            f"- Dataset examples: {manifest['dataset_count']}",
            f"- Previously attempted source IDs: {manifest['previously_attempted_count']}",
            f"- Missing source IDs selected: {manifest['missing_count']}",
            f"- GPT-5.4 failed/retry source IDs: {manifest['gpt54_failed_count']}",
            f"- Combined accepted missing traces: {manifest['combined_candidate_count']}",
            f"- GPT-5.5 retry still failed: {manifest['gpt55_retry_failed_count']}",
            "",
            "## Paths",
            "",
            f"- GPT-5.4 config: `{manifest['gpt54_config_path']}`",
            f"- GPT-5.4 records: `{manifest['gpt54_records_path']}`",
            f"- GPT-5.5 config: `{manifest['gpt55_config_path']}`",
            f"- GPT-5.5 records: `{manifest['gpt55_records_path']}`",
            f"- Combined candidates: `{manifest['combined_candidate_records_path']}`",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_toml(config_path)
    output_dir = resolve_path(str(config.get("output_dir", "outputs/rlm_traces")), relative_to=config_path.parent)
    endpoints_path = str(config.get("endpoints_path", "configs/endpoints.toml"))
    driver_cfg = config.get("driver", {})
    previous_record_paths = [
        resolve_path(str(path), relative_to=config_path.parent)
        for path in config.get("previous", {}).get("record_paths", [])
    ]
    dataset_source_ids = load_dataset_source_ids(config["dataset"])
    attempted_source_ids = collect_attempted_source_ids(previous_record_paths)
    missing_source_ids = [source_id for source_id in dataset_source_ids if source_id not in attempted_source_ids]
    if args.limit is not None:
        missing_source_ids = missing_source_ids[: max(0, int(args.limit))]

    primary_run_name = str(driver_cfg.get("primary_run_name", "prime-gpt54-vertex-flash-lite-sft-missing-v1"))
    retry_run_name = str(driver_cfg.get("retry_run_name", "prime-gpt55-vertex-flash-lite-sft-missing-v1-retry-failed-gpt54"))
    manifest_run_name = str(driver_cfg.get("manifest_run_name", "prime-gpt54-gpt55-sft-missing-auto-retry-v1"))
    variant = str((config.get("rollout", {}).get("prompt_variants") or [DEFAULT_VARIANT])[0])
    manifest_dir = output_dir / manifest_run_name
    config_dir = manifest_dir / "configs"

    print(
        json.dumps(
            {
                "dataset_count": len(dataset_source_ids),
                "previously_attempted_count": len(attempted_source_ids),
                "missing_count": len(missing_source_ids),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_dir / "missing_source_ids.json", missing_source_ids)
    if not missing_source_ids:
        manifest = {
            "created_at_unix": time.time(),
            "dataset_count": len(dataset_source_ids),
            "previously_attempted_count": len(attempted_source_ids),
            "missing_count": 0,
            "gpt54_failed_count": 0,
            "gpt55_retry_success_count": 0,
            "gpt55_retry_failed_count": 0,
            "combined_candidate_count": 0,
            "gpt54_config_path": "",
            "gpt54_records_path": "",
            "gpt55_config_path": "",
            "gpt55_records_path": "",
            "combined_candidate_records_path": str(manifest_dir / "combined_candidate_records.jsonl"),
        }
        write_jsonl(manifest_dir / "combined_candidate_records.jsonl", [])
        write_json(manifest_dir / "gpt54_failed_source_ids.json", [])
        write_json(manifest_dir / "gpt55_retry_success_source_ids.json", [])
        write_json(manifest_dir / "gpt55_retry_failed_source_ids.json", [])
        write_json(manifest_dir / "manifest.json", manifest)
        (manifest_dir / "report.md").write_text(render_report(manifest))
        print(render_report(manifest))
        return

    primary_config_path = config_dir / "gpt54_missing.toml"
    primary_config_path.write_text(
        render_trace_config(
            run_name=primary_run_name,
            output_dir=output_dir,
            endpoints_path=endpoints_path,
            model_cfg=config["primary_model"],
            llm_subcall_cfg=config.get("llm_subcall"),
            dataset_cfg=config["dataset"],
            rollout_cfg=config["rollout"],
            judge_cfg=config["judge"],
            source_ids=missing_source_ids,
            max_workers=int(config.get("max_workers", 6)),
            worker_backend=str(config.get("worker_backend", "process")),
            resume=bool(config.get("resume", True)),
        )
    )
    print(f"\n== GPT-5.4 primary generation: {len(missing_source_ids)} missing source IDs ==", flush=True)
    run_trace_config(primary_config_path)

    primary_records_path = records_path(output_dir, primary_run_name, variant)
    primary_records = read_jsonl(primary_records_path)
    retry_source_ids = failed_source_ids(primary_records, missing_source_ids)
    write_json(manifest_dir / "gpt54_failed_source_ids.json", retry_source_ids)
    print(f"\n== GPT-5.4 primary complete: retrying {len(retry_source_ids)} failed source IDs with GPT-5.5 ==", flush=True)

    retry_config_path = config_dir / "gpt55_retry.toml"
    retry_config_path.write_text(
        render_trace_config(
            run_name=retry_run_name,
            output_dir=output_dir,
            endpoints_path=endpoints_path,
            model_cfg=config["retry_model"],
            llm_subcall_cfg=config.get("llm_subcall"),
            dataset_cfg=config["dataset"],
            rollout_cfg=config["rollout"],
            judge_cfg=config["judge"],
            source_ids=retry_source_ids,
            max_workers=int(config.get("max_workers", 6)),
            worker_backend=str(config.get("worker_backend", "process")),
            resume=bool(config.get("resume", True)),
        )
    )
    if retry_source_ids:
        print(f"\n== GPT-5.5 retry generation: {len(retry_source_ids)} source IDs ==", flush=True)
        run_trace_config(retry_config_path)
    else:
        print("\n== GPT-5.5 retry skipped: no failed GPT-5.4 records ==", flush=True)

    retry_records_path = records_path(output_dir, retry_run_name, variant)
    retry_records = read_jsonl(retry_records_path)
    primary_latest = latest_record_by_source(primary_records)
    retry_latest = latest_record_by_source(retry_records)

    combined: list[dict[str, Any]] = []
    retry_success_ids: list[str] = []
    retry_failed_ids: list[str] = []
    for source_id in missing_source_ids:
        primary_record = primary_latest.get(source_id)
        if primary_record is not None and record_is_good_trace(primary_record):
            combined.append(primary_record)
            continue
        retry_record = retry_latest.get(source_id)
        if retry_record is not None and record_is_good_trace(retry_record):
            combined.append(retry_record)
            retry_success_ids.append(source_id)
        elif source_id in retry_source_ids:
            retry_failed_ids.append(source_id)

    combined_path = manifest_dir / "combined_candidate_records.jsonl"
    write_jsonl(combined_path, combined)
    write_json(manifest_dir / "gpt55_retry_success_source_ids.json", retry_success_ids)
    write_json(manifest_dir / "gpt55_retry_failed_source_ids.json", retry_failed_ids)

    manifest = {
        "created_at_unix": time.time(),
        "dataset_count": len(dataset_source_ids),
        "previously_attempted_count": len(attempted_source_ids),
        "missing_count": len(missing_source_ids),
        "gpt54_failed_count": len(retry_source_ids),
        "gpt55_retry_success_count": len(retry_success_ids),
        "gpt55_retry_failed_count": len(retry_failed_ids),
        "combined_candidate_count": len(combined),
        "gpt54_config_path": str(primary_config_path),
        "gpt54_records_path": str(primary_records_path),
        "gpt55_config_path": str(retry_config_path),
        "gpt55_records_path": str(retry_records_path),
        "combined_candidate_records_path": str(combined_path),
    }
    write_json(manifest_dir / "manifest.json", manifest)
    (manifest_dir / "report.md").write_text(render_report(manifest))
    print(render_report(manifest))


if __name__ == "__main__":
    main()
