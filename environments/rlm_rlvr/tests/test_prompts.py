from __future__ import annotations

import pytest

from rlm_rlvr.external_rlm import build_initial_messages, build_system_prompt
from rlm_rlvr.prompt_variants import PROMPT_VARIANTS, get_system_prompt_template


def test_prompt_variants_validate_names() -> None:
    with pytest.raises(ValueError):
        get_system_prompt_template("missing")


@pytest.mark.parametrize("prompt_variant", ["balanced_v1", "balanced_v2", "sanjaya_text_v1"])
def test_balanced_prompt_variants_keep_required_contracts(prompt_variant: str) -> None:
    prompt = get_system_prompt_template(prompt_variant)
    assert "sub-call" in prompt or "subcall" in prompt
    assert "`context`" in prompt
    assert "```repl" in prompt
    assert "Plain text or unfenced code will not run." in prompt
    assert "FINAL(" in prompt
    assert "FINAL_VAR(" in prompt


def test_build_system_prompt_uses_requested_variant() -> None:
    prompt = build_system_prompt(depth=0, max_depth=2, prompt_variant="balanced_v1")
    assert "Each turn should do useful work immediately" in prompt
    assert sorted(PROMPT_VARIANTS) == ["balanced_v1", "balanced_v2", "default", "sanjaya_text_v1"]


def test_initial_messages_include_system_context_metadata_and_question() -> None:
    messages = build_initial_messages(
        context_payload="alpha beta gamma",
        root_prompt="What is in the context?",
        prompt_variant="balanced_v1",
    )

    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert "Each turn should do useful work immediately" in messages[0]["content"]
    assert "16 total characters" in messages[1]["content"]
    assert "What is in the context?" in messages[2]["content"]
