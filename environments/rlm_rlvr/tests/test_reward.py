from __future__ import annotations

from types import SimpleNamespace

from rlm_rlvr.reward import _extract_message_text, _is_exact_match, _parse_binary_judge_score


def test_is_exact_match_handles_numeric_normalization() -> None:
    assert _is_exact_match("1,208", ["1208"])
    assert _is_exact_match('"1208"', ["1208"])


def test_is_exact_match_handles_json_normalization() -> None:
    assert _is_exact_match('{"b": 2, "a": 1}', ['{"a":1,"b":2}'])


def test_extract_message_text_falls_back_to_reasoning() -> None:
    message = SimpleNamespace(content=None, reasoning="1")
    assert _extract_message_text(message) == "1"


def test_parse_binary_judge_score_accepts_embedded_binary() -> None:
    assert _parse_binary_judge_score("score: 1") == 1.0
