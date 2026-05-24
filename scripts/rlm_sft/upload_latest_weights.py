from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "lsteno/Qwen3-4B-Instruct-2507-RLM-SFT-v2"


def step_number(path: Path) -> int | None:
    match = re.fullmatch(r"step_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def latest_weight_step(output_dir: Path) -> Path:
    weights_dir = output_dir / "weights"
    candidates: list[tuple[int, Path]] = []
    if weights_dir.exists():
        for child in weights_dir.iterdir():
            if child.is_dir() and (number := step_number(child)) is not None:
                candidates.append((number, child))
    if not candidates:
        raise FileNotFoundError(f"No weights/step_* directories found under {output_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def relative_upload_files(weights_path: Path) -> list[str]:
    files: list[str] = []
    for path in weights_path.rglob("*"):
        if path.is_file():
            files.append(path.relative_to(weights_path).as_posix())
    return sorted(files)


def verify_uploaded_files(api: HfApi, repo_id: str, weights_path: Path) -> None:
    expected = relative_upload_files(weights_path)
    if not expected:
        raise FileNotFoundError(f"No files found under selected weights directory: {weights_path}")
    uploaded = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    missing = [path for path in expected if path not in uploaded]
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"HF upload verification failed; missing {len(missing)} files from {repo_id}: {preview}")
    if not any(path.endswith(".safetensors") for path in expected):
        raise RuntimeError(f"Selected weights directory has no .safetensors files: {weights_path}")


def upload_latest_weights(output_dir: Path, repo_id: str, *, dry_run: bool = False) -> Path:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be set before uploading model weights to Hugging Face")
    weights_path = latest_weight_step(output_dir)
    if dry_run:
        print(f"Would create private model repo if needed: {repo_id}")
        print(f"Would upload folder: {weights_path}")
        print(f"Would verify {len(relative_upload_files(weights_path))} uploaded files")
        return weights_path
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=weights_path, path_in_repo=".")
    verify_uploaded_files(api, repo_id, weights_path)
    return weights_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the latest Prime SFT weights/step_* directory to Hugging Face.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=os.environ.get("HF_MODEL_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uploaded = upload_latest_weights(args.output_dir, args.repo_id, dry_run=args.dry_run)
    print(f"Selected weights directory: {uploaded}")


if __name__ == "__main__":
    main()
