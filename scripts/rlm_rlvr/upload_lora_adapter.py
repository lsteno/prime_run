from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_PREFIX = "lsteno/qwen3-rlm-depth1"


def step_number(path: Path) -> int | None:
    match = re.fullmatch(r"step_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def latest_adapter_step(output_dir: Path) -> Path:
    broadcasts_dir = output_dir / "run_default" / "broadcasts"
    candidates: list[tuple[int, Path]] = []
    if broadcasts_dir.exists():
        for child in broadcasts_dir.iterdir():
            if child.is_dir() and (number := step_number(child)) is not None:
                candidates.append((number, child))
    if not candidates:
        raise FileNotFoundError(f"No run_default/broadcasts/step_* directories found under {output_dir}")
    selected = max(candidates, key=lambda item: item[0])[1]
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (selected / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Selected adapter directory is missing {missing}: {selected}")
    return selected


def repo_id_for_run(run_id: str, repo_prefix: str) -> str:
    match = re.fullmatch(
        r"rlm-rlvr-qwen3-4b-depth1-llmonly-r(?P<rank>\d+)-a(?P<alpha>\d+)-lr(?P<lr>.+)-s(?P<steps>\d+)(?:-(?P<suffix>.+))?",
        run_id,
    )
    if match is None:
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)
        return f"{repo_prefix}-{safe_run_id}-lora"
    suffix = match.group("suffix")
    safe_suffix = f"-{re.sub(r'[^A-Za-z0-9_.-]+', '-', suffix)}" if suffix else ""
    return (
        f"{repo_prefix}-"
        f"r{int(match.group('rank'))}-"
        f"a{int(match.group('alpha'))}-"
        f"lr{match.group('lr')}-"
        f"s{int(match.group('steps'))}"
        f"{safe_suffix}-lora"
    )


def adapter_upload_files(adapter_path: Path) -> list[str]:
    return sorted(path.relative_to(adapter_path).as_posix() for path in adapter_path.rglob("*") if path.is_file())


def verify_uploaded_files(api: HfApi, repo_id: str, expected_paths: list[str]) -> None:
    uploaded = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    missing = [path for path in expected_paths if path not in uploaded]
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"HF upload verification failed for {repo_id}; missing {len(missing)} files: {preview}")
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(uploaded):
        raise RuntimeError(f"HF upload verification failed for {repo_id}; missing required adapter files")


def write_model_card(path: Path, *, run_id: str, base_model: str, adapter_step: int) -> None:
    body = f"""---
base_model: {base_model}
library_name: peft
license: apache-2.0
private: true
---

# {run_id}

LoRA adapter from the depth-1 LLM-only RLM RLVR rank/LR ablation.

- Base model: `{base_model}`
- Run id: `{run_id}`
- Adapter source: `run_default/broadcasts/step_{adapter_step}`
- Prompt variant: `sanjaya_text_depth1_llm_only_v1`
- Recursive RLM calls blocked at runtime; plain LLM subcalls remain enabled.
"""
    path.write_text(body)


def upload_lora_adapter(
    output_dir: Path,
    *,
    run_id: str,
    repo_id: str | None,
    repo_prefix: str,
    base_model: str,
    dry_run: bool = False,
) -> tuple[str, Path]:
    if not dry_run and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be set before uploading LoRA adapters to Hugging Face")

    adapter_path = latest_adapter_step(output_dir)
    adapter_step = step_number(adapter_path)
    if adapter_step is None:
        raise RuntimeError(f"Could not parse adapter step from {adapter_path}")
    selected_repo_id = repo_id or repo_id_for_run(run_id, repo_prefix)

    temp_readme = adapter_path / "README.md"
    original_readme = temp_readme.read_text() if temp_readme.exists() else None
    write_model_card(temp_readme, run_id=run_id, base_model=base_model, adapter_step=adapter_step)
    expected = adapter_upload_files(adapter_path)
    config_dir = output_dir / "configs"

    if dry_run:
        print(f"Would create private model repo if needed: {selected_repo_id}")
        print(f"Would upload adapter folder: {adapter_path}")
        print(f"Would upload run configs: {config_dir if config_dir.exists() else '(missing, skipped)'}")
        print(f"Would verify required adapter files plus {len(expected)} total adapter-root files")
        if original_readme is None:
            temp_readme.unlink(missing_ok=True)
        else:
            temp_readme.write_text(original_readme)
        return selected_repo_id, adapter_path

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=selected_repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=selected_repo_id,
        repo_type="model",
        folder_path=adapter_path,
        path_in_repo=".",
        commit_message=f"Upload {run_id} LoRA adapter step {adapter_step}",
    )
    if config_dir.exists():
        api.upload_folder(
            repo_id=selected_repo_id,
            repo_type="model",
            folder_path=config_dir,
            path_in_repo="run_configs",
            commit_message=f"Upload {run_id} run configs",
        )
    verify_uploaded_files(api, selected_repo_id, expected)
    return selected_repo_id, adapter_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a completed Prime-RL LoRA adapter to Hugging Face.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--repo-prefix", default=os.environ.get("HF_LORA_REPO_PREFIX", DEFAULT_REPO_PREFIX))
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_id, adapter_path = upload_lora_adapter(
        args.output_dir,
        run_id=args.run_id,
        repo_id=args.repo_id,
        repo_prefix=args.repo_prefix,
        base_model=args.base_model,
        dry_run=args.dry_run,
    )
    print(f"Uploaded LoRA adapter directory: {adapter_path}")
    print(f"HF repo: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
