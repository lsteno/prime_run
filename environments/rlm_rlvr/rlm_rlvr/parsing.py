from __future__ import annotations

import json
import re
from typing import Any

from .external_rlm import find_code_blocks, find_final_answer

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    text = str(value).strip()
    text = text.strip("\"'")
    text = WHITESPACE_RE.sub(" ", text)
    return text.casefold()


def parse_answer_candidates(raw_answer: Any) -> list[str]:
    if raw_answer is None:
        return []

    if isinstance(raw_answer, list):
        return [normalize_text(item) for item in raw_answer if str(item).strip()]

    if isinstance(raw_answer, dict):
        return [normalize_text(json.dumps(raw_answer, sort_keys=True, separators=(",", ":")))]

    text = str(raw_answer).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [normalize_text(text)]

    return parse_answer_candidates(parsed)


def extract_code_blocks(text: str) -> list[str]:
    return [match.strip() for match in find_code_blocks(text or "") if match.strip()]


def extract_final_answer(text: str, environment: Any | None = None) -> str | None:
    if not text:
        return None
    result = find_final_answer(text, environment=environment)
    if result is None:
        result = _find_markdown_wrapped_final_answer(text, environment=environment)
    if result is None:
        return None
    return str(result).strip().strip("\"'") or None


def _find_markdown_wrapped_final_answer(text: str, environment: Any | None = None) -> str | None:
    """Accept common markdown wrapping around otherwise valid FINAL calls."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        unwrapped = stripped
        if unwrapped.startswith("`") and unwrapped.endswith("`"):
            unwrapped = unwrapped.strip("`").strip()
        elif unwrapped.startswith("**") and unwrapped.endswith("**"):
            unwrapped = unwrapped.strip("*").strip()
        if unwrapped == stripped:
            continue
        result = find_final_answer(unwrapped, environment=environment)
        if result is not None:
            return str(result)
    return None


def render_execution_output(stdout: str, stderr: str, final_answer: str | None) -> str:
    parts: list[str] = []
    if stdout.strip():
        parts.append(f"stdout:\n{stdout.strip()}")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr.strip()}")
    if final_answer is not None:
        parts.append(f"final_answer:\n{final_answer}")
    if not parts:
        return "No output."
    return "\n\n".join(parts)
