#!/usr/bin/env python3
"""Upload one exact Prime-RL full-FT weights/step_* checkpoint to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED_FILES = {
    "STABLE",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def _files_under(path: Path) -> list[str]:
    return sorted(child.relative_to(path).as_posix() for child in path.rglob("*") if child.is_file())


def _validate_local_weights(weights_path: Path) -> None:
    if not weights_path.is_dir():
        raise FileNotFoundError(f"Missing weights directory: {weights_path}")
    missing = sorted(name for name in REQUIRED_FILES if not (weights_path / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing required checkpoint files under {weights_path}: {missing}")
    safetensors = sorted(weights_path.glob("*.safetensors"))
    if not safetensors:
        raise FileNotFoundError(f"No .safetensors files found under {weights_path}")


def _verify_remote(api: HfApi, repo_id: str, expected_files: list[str]) -> None:
    uploaded = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    missing = [name for name in expected_files if name not in uploaded]
    if missing:
        preview = ", ".join(missing[:20])
        raise RuntimeError(f"HF upload verification failed for {repo_id}; missing {len(missing)} files: {preview}")
    if not any(name.endswith(".safetensors") for name in uploaded):
        raise RuntimeError(f"HF upload verification failed for {repo_id}; no safetensors found")


def _write_model_card(path: Path, *, repo_id: str, source: Path, step: int, base_model: str) -> None:
    path.write_text(
        f"""---
base_model: {base_model}
library_name: transformers
license: apache-2.0
private: true
---

# {repo_id}

Full-parameter RLM RLVR checkpoint.

- Base model: `{base_model}`
- Source checkpoint: `{source}`
- Step: `{step}`
- Prompt variant: `sanjaya_text_depth1_llm_only_v1`
- Runtime: depth-1 LLM-only RLM harness, plain Gemini subcalls, recursive child RLMs disabled.
""",
        encoding="utf-8",
    )


def upload_full_ft_weights(
    weights_path: Path,
    *,
    repo_id: str,
    base_model: str,
    success_path: Path,
    dry_run: bool = False,
) -> dict[str, object]:
    _validate_local_weights(weights_path)
    expected_files = _files_under(weights_path)
    step = int(weights_path.name.removeprefix("step_")) if weights_path.name.startswith("step_") else -1

    readme_path = weights_path / "README.md"
    old_readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
    _write_model_card(readme_path, repo_id=repo_id, source=weights_path, step=step, base_model=base_model)
    expected_files = _files_under(weights_path)

    payload: dict[str, object] = {
        "repo_id": repo_id,
        "weights_path": str(weights_path),
        "step": step,
        "file_count": len(expected_files),
        "files": expected_files,
        "dry_run": dry_run,
        "uploaded_at_unix": None,
    }

    try:
        if dry_run:
            print(json.dumps(payload, indent=2))
            return payload
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN must be set before uploading full-FT weights")
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=weights_path,
            path_in_repo=".",
            commit_message=f"Upload full-FT RLVR checkpoint step {step}",
        )
        _verify_remote(api, repo_id, expected_files)
        payload["uploaded_at_unix"] = time.time()
        success_path.parent.mkdir(parents=True, exist_ok=True)
        success_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
    finally:
        if old_readme is None:
            readme_path.unlink(missing_ok=True)
        else:
            readme_path.write_text(old_readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--success-path", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = upload_full_ft_weights(
        args.weights_path,
        repo_id=args.repo_id,
        base_model=args.base_model,
        success_path=args.success_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
