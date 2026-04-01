from __future__ import annotations

import pytest

from rlm_rlvr.external_rlm import build_system_prompt
from rlm_rlvr.prompt_variants import PROMPT_VARIANTS, get_system_prompt_template


def test_prompt_variants_validate_names() -> None:
    with pytest.raises(ValueError):
        get_system_prompt_template("missing")


@pytest.mark.parametrize("prompt_variant", ["balanced_v1", "balanced_v2"])
def test_balanced_prompt_variants_keep_required_contracts(prompt_variant: str) -> None:
    prompt = get_system_prompt_template(prompt_variant)
    assert "sub-calls" in prompt
    assert "`context`" in prompt
    assert "```repl" in prompt
    assert "Plain text or unfenced code will not run." in prompt
    assert "FINAL(" in prompt
    assert "FINAL_VAR(" in prompt


def test_build_system_prompt_uses_requested_variant() -> None:
    prompt = build_system_prompt(depth=0, max_depth=2, prompt_variant="balanced_v1")
    assert "Each turn should do useful work immediately" in prompt
    assert sorted(PROMPT_VARIANTS) == ["balanced_v1", "balanced_v2", "default"]
