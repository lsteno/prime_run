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


@dataclass(frozen=True)
class NaturalSemanticEvidence:
    matched_text_hashes: tuple[str, ...]
    line_match_count: int
    substring_match_count: int


def canonical_record_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", text).strip()


def record_text_hash(value: str) -> str:
    return hashlib.sha256(canonical_record_text(value).encode("utf-8")).hexdigest()


def _visible_record_text(line: str) -> str | None:
    record_match = _RECORD_LINE_RE.fullmatch(line)
    if record_match:
        return record_match.group(2)
    bullet_match = re.fullmatch(r"\s*(?:[-*]|\d+[.)])\s+(.*?)\s*", line)
    if bullet_match:
        return bullet_match.group(1)
    return None


def natural_context_records(text: str) -> dict[str, str]:
    """Return canonical visible record text keyed by a content hash."""
    records: dict[str, str] = {}
    for line in text.splitlines():
        record_text = _visible_record_text(line)
        if record_text is None:
            continue
        canonical = canonical_record_text(record_text)
        if canonical:
            records[record_text_hash(canonical)] = canonical
    return records


def match_natural_semantic_evidence(prompt: str, context: str) -> NaturalSemanticEvidence:
    """Privately match source records copied into a natural-language child prompt."""
    context_records = natural_context_records(context)
    if not context_records:
        return NaturalSemanticEvidence((), 0, 0)

    matched: set[str] = set()
    for line in prompt.splitlines():
        record_text = _visible_record_text(line)
        candidates = [record_text] if record_text is not None else [line]
        for candidate in candidates:
            canonical = canonical_record_text(candidate)
            if not canonical:
                continue
            candidate_hash = record_text_hash(canonical)
            if candidate_hash in context_records:
                matched.add(candidate_hash)

    line_matches = len(matched)
    if line_matches < len(context_records):
        canonical_prompt = canonical_record_text(prompt)
        for text_hash, canonical_text in context_records.items():
            if text_hash not in matched and canonical_text in canonical_prompt:
                matched.add(text_hash)

    return NaturalSemanticEvidence(
        matched_text_hashes=tuple(sorted(matched)),
        line_match_count=line_matches,
        substring_match_count=len(matched) - line_matches,
    )


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
