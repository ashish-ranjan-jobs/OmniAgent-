"""Fallback routing: failed/empty primary completions retry on the fallback."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from app.core.config import get_settings
from app.llm.client import LLMClient, LLMUnavailableError


def _completion(text):
    return NS(
        choices=[NS(message=NS(content=text))],
        usage=NS(prompt_tokens=1, completion_tokens=2),
    )


def _fake(result=None, exc=None):
    async def create(**kwargs):
        if exc is not None:
            raise exc
        return result

    return NS(chat=NS(completions=NS(create=create)))


def _ok(text):
    return _fake(result=_completion(text))


async def test_fallback_used_when_primary_fails():
    llm = LLMClient(primary=_fake(exc=RuntimeError("primary down")), fallback=_ok("fb"))
    trace = []
    result = await llm.complete("strong", "s", "u", trace=trace, node="n")
    assert result.text == "fb"
    assert result.model == llm._fallback_model
    assert trace[0]["node"] == "n"
    assert trace[0]["model"] == llm._fallback_model


async def test_fallback_on_empty_primary_completion():
    llm = LLMClient(primary=_ok("   "), fallback=_ok("real answer"))
    result = await llm.complete("strong", "s", "u")
    assert result.text == "real answer"


async def test_no_fallback_when_disabled():
    llm = LLMClient(primary=_fake(exc=RuntimeError("primary down")))
    with pytest.raises(LLMUnavailableError):
        await llm.complete("strong", "s", "u")


async def test_fallback_failure_raises_with_both_contexts():
    llm = LLMClient(
        primary=_fake(exc=RuntimeError("primary down")),
        fallback=_fake(exc=RuntimeError("fallback down")),
    )
    with pytest.raises(LLMUnavailableError, match="primary down.*fallback down"):
        await llm.complete("strong", "s", "u")


async def test_empty_primary_and_empty_fallback_raises():
    llm = LLMClient(primary=_ok(""), fallback=_ok(""))
    with pytest.raises(LLMUnavailableError, match="empty completion"):
        await llm.complete("strong", "s", "u")


def test_fallback_disabled_by_default_in_settings():
    assert get_settings().llm_fallback_enabled is False
