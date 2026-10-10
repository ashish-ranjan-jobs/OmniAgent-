"""First-party LLM client over any OpenAI-compatible backend (ADR-0002).

We own retries, timeouts, role routing and JSON parsing explicitly, instead
of hiding them behind a framework integration. Swapping Ollama -> vLLM -> a
commercial API is a .env change (LLM_BASE_URL / LLM_MODEL), not a code change.

Automated fallback (LLM_FALLBACK_ENABLED): when the primary backend fails
after retries, or returns an empty completion, the call retries once on the
fallback backend; the trace records which backend actually served it.
Streaming (stream()) does not fall back mid-stream.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

Role = Literal["fast", "strong"]

_UNSET = object()  # sentinel: "not injected; settings decide"


class LLMUnavailableError(RuntimeError):
    """Backend unreachable after retries; surfaces as HTTP 503."""


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0


def parse_json(text: str) -> dict | list | None:
    """Tolerant JSON extraction: strips markdown fences, finds first JSON value."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    for start, end in (("{", "}"), ("[", "]")):
        s = cleaned.find(start)
        e = cleaned.rfind(end)
        if s != -1 and e > s:
            try:
                return json.loads(cleaned[s : e + 1])
            except json.JSONDecodeError:
                continue
    return None


class LLMClient:
    """Thin, testable wrapper over the chat-completions endpoint."""

    def __init__(self, primary: object | None = None, fallback: object = _UNSET) -> None:
        s = get_settings()
        self._primary = (
            primary
            if primary is not None
            else AsyncOpenAI(
                base_url=s.llm_base_url,
                api_key=s.llm_api_key,
                timeout=s.llm_timeout_s,
                max_retries=0,  # retry policy is owned here, not by the SDK
            )
        )
        if fallback is _UNSET:
            fallback = (
                AsyncOpenAI(
                    base_url=s.llm_fallback_base_url,
                    api_key=s.llm_fallback_api_key,
                    timeout=s.llm_timeout_s,
                    max_retries=0,
                )
                if s.llm_fallback_enabled
                else None
            )
        self._fallback = fallback
        self._fallback_model = s.llm_fallback_model
        self._roles: dict[str, str] = {"fast": s.llm_fast_model, "strong": s.llm_model}
        attempts = max(1, s.llm_max_retries + 1)
        self._chat = self._retried_chat(self._primary, attempts)
        self._fallback_chat = (
            self._retried_chat(self._fallback, attempts) if self._fallback is not None else None
        )

    def model_for(self, role: Role) -> str:
        return self._roles.get(role, self._roles["strong"])

    def _retried_chat(self, client: object, attempts: int):
        async def impl(model: str, system: str, user: str, json_mode: bool):
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            return await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                **kwargs,
            )

        return retry(
            retry=retry_if_exception_type(
                (APIConnectionError, APITimeoutError, InternalServerError)
            ),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=1, max=8),
            reraise=True,
        )(impl)

    async def _call(self, chat, model: str, system: str, user: str, json_mode: bool):
        try:
            return await chat(model, system, user, json_mode)
        except APIStatusError:
            # Backend rejected json_object mode; degrade gracefully once.
            if not json_mode:
                raise
            return await chat(model, system, user, False)

    async def _complete_with_fallback(
        self, model: str, system: str, user: str, json_mode: bool
    ) -> tuple[object, str]:
        try:
            completion = await self._call(self._chat, model, system, user, json_mode)
        except Exception as exc:
            if self._fallback_chat is None:
                raise LLMUnavailableError(f"LLM backend failed after retries: {exc}") from exc
            return await self._fallback_attempt(system, user, json_mode, cause=str(exc))
        if (
            (completion.choices[0].message.content or "").strip()
            or self._fallback_chat is None
        ):
            return completion, model
        # Primary returned an empty completion: try the fallback once.
        return await self._fallback_attempt(system, user, json_mode, cause="empty completion")

    async def _fallback_attempt(
        self, system: str, user: str, json_mode: bool, cause: str
    ) -> tuple[object, str]:
        log.info("llm_fallback", cause=cause, model=self._fallback_model)
        try:
            completion = await self._call(
                self._fallback_chat, self._fallback_model, system, user, json_mode
            )
        except Exception as exc:
            raise LLMUnavailableError(
                f"LLM backend failed after retries (primary: {cause}; fallback: {exc})"
            ) from exc
        if not (completion.choices[0].message.content or "").strip():
            raise LLMUnavailableError(
                f"LLM backend failed (primary: {cause}; fallback: empty completion)"
            )
        return completion, self._fallback_model

    async def complete(
        self,
        role: Role,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        trace: list | None = None,
        node: str = "",
    ) -> LLMResult:
        model = self.model_for(role)
        start = time.perf_counter()
        completion, used_model = await self._complete_with_fallback(model, system, user, json_mode)
        latency_ms = int((time.perf_counter() - start) * 1000)
        choice = completion.choices[0]
        usage = completion.usage
        result = LLMResult(
            text=(choice.message.content or "").strip(),
            model=used_model,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        if trace is not None:
            trace.append(
                {
                    "node": node,
                    "model": used_model,
                    "latency_ms": latency_ms,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            )
        return result

    async def stream(self, role: Role, system: str, user: str):
        """Yield text deltas from the primary backend (no mid-stream fallback)."""
        model = self.model_for(role)
        response = await self._primary.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            stream=True,
        )
        async for event in response:
            delta = event.choices[0].delta if event.choices else None
            if delta is not None and delta.content:
                yield delta.content

    async def ping(self) -> bool:
        try:
            await self._primary.models.list()
            return True
        except Exception:
            return False
