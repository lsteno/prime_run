from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass


_RECORD_LINE_RE = re.compile(
    r"^\s*Record ID:\s*(r\d{5})\s*\|\|\s*Text:\s*(.*?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class SemanticPromptEvidence:
    record_hashes: dict[str, str]
    duplicate_record_ids: tuple[str, ...]
    malformed_record_lines: int


def canonical_record_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", text).strip()


def record_text_hash(value: str) -> str:
    return hashlib.sha256(canonical_record_text(value).encode("utf-8")).hexdigest()


def parse_semantic_prompt_evidence(text: str) -> SemanticPromptEvidence:
    record_hashes: dict[str, str] = {}
    duplicates: set[str] = set()
    for match in _RECORD_LINE_RE.finditer(text):
        record_id = match.group(1).casefold()
        if record_id in record_hashes:
            duplicates.add(record_id)
        record_hashes[record_id] = record_text_hash(match.group(2))

    record_line_count = sum(
        1 for line in text.splitlines() if re.match(r"^\s*Record ID\s*:", line, flags=re.IGNORECASE)
    )
    return SemanticPromptEvidence(
        record_hashes=record_hashes,
        duplicate_record_ids=tuple(sorted(duplicates)),
        malformed_record_lines=max(0, record_line_count - len(list(_RECORD_LINE_RE.finditer(text)))),
    )
