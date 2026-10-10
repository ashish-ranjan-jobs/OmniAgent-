"""First-party LLM client over any OpenAI-compatible backend (ADR-0002).

We own retries, timeouts, role routing and JSON parsing explicitly, instead
of hiding them behind a framework integration. Swapping Ollama -> vLLM -> a
commercial API is a .env change (LLM_BASE_URL / LLM_MODEL), not a code change.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Literal

import openai
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

Role = Literal["fast", "strong"]


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

    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
            timeout=s.llm_timeout_s,
            max_retries=0,  # retry policy is owned here, not by the SDK
        )
        self._roles: dict[str, str] = {"fast": s.llm_fast_model, "strong": s.llm_model}
        attempts = max(1, s.llm_max_retries + 1)
        self._chat = retry(
            retry=retry_if_exception_type(
                (APIConnectionError, APITimeoutError, InternalServerError)
            ),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=1, max=8),
            reraise=True,
        )(self._chat_impl)

    def model_for(self, role: Role) -> str:
        return self._roles.get(role, self._roles["strong"])

    async def _chat_impl(
        self, model: str, system: str, user: str, json_mode: bool
    ) -> openai.types.chat.ChatCompletion:
        kwargs = (
            {"response_format": {"type": "json_object"}} if json_mode else {}
        )
        return await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            **kwargs,
        )

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
        try:
            try:
                completion = await self._chat(model, system, user, json_mode)
            except APIStatusError:
                # Backend rejected json_object mode; degrade gracefully once.
                if not json_mode:
                    raise
                completion = await self._chat(model, system, user, False)
        except Exception as exc:
            raise LLMUnavailableError(f"LLM backend failed after retries: {exc}") from exc
        latency_ms = int((time.perf_counter() - start) * 1000)
        choice = completion.choices[0]
        usage = completion.usage
        result = LLMResult(
            text=(choice.message.content or "").strip(),
            model=model,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        if trace is not None:
            trace.append(
                {
                    "node": node,
                    "model": model,
                    "latency_ms": latency_ms,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            )
        return result

    async def ping(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
