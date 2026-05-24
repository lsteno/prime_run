#!/usr/bin/env python3
"""Upload final RLM eval artifacts to a private Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


def upload_eval_results(work_dir: Path, *, repo_id: str, success_path: Path, dry_run: bool = False) -> dict[str, object]:
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Missing eval work dir: {work_dir}")
    summary = work_dir / "summary" / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(f"Missing summary file: {summary}")
    result_files = sorted(work_dir.rglob("results.jsonl"))
    if not result_files:
        raise FileNotFoundError(f"No results.jsonl files found under {work_dir}")
    payload: dict[str, object] = {
        "repo_id": repo_id,
        "work_dir": str(work_dir),
        "result_file_count": len(result_files),
        "uploaded_at_unix": None,
        "dry_run": dry_run,
    }
    if dry_run:
        print(json.dumps(payload, indent=2))
        return payload
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be set before uploading eval results to Hugging Face")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=work_dir,
        path_in_repo=".",
        commit_message="Upload final LoRA vs full-FT BEEG pass@10 eval artifacts",
        ignore_patterns=["*.lock", "__pycache__/*"],
    )
    uploaded = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    required = {"summary/summary.json", "summary/aggregate.csv", "selected_models.json", "hf_upload_success.json"}
    missing = sorted(required - uploaded)
    if missing:
        raise RuntimeError(f"HF dataset upload verification failed for {repo_id}; missing {missing}")
    payload["uploaded_at_unix"] = time.time()
    success_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--success-path", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = upload_eval_results(args.work_dir, repo_id=args.repo_id, success_path=args.success_path, dry_run=args.dry_run)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
