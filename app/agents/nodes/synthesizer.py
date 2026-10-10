"""Synthesizer node: grounded draft with [n] citations, or a refusal."""

from __future__ import annotations

import time

from langgraph.config import get_stream_writer

from app.agents.prompts import SYNTHESIZER_SYSTEM
from app.agents.state import OmniAgentState
from app.core.logging import get_logger
from app.llm.client import LLMClient

log = get_logger(__name__)

REFUSAL_MESSAGE = (
    "The retrieved evidence is insufficient to answer this question reliably, "
    "so no answer is provided rather than guessing from outside knowledge. "
    "Rephrase the question or ask about topics covered by the indexed documents."
)


async def synthesizer_node(
    state: OmniAgentState, llm: LLMClient, refusal_min_evidence: int
) -> dict:
    evidence = state.get("evidence") or []

    if len(evidence) < refusal_min_evidence:
        # Groundedness gate: refuse rather than hallucinate on thin evidence.
        log.info("synthesizer_refused", evidence_count=len(evidence))
        return {"draft": REFUSAL_MESSAGE, "refused": True}

    blocks = "\n\n".join(
        f"[{i}] ({ev['title']})\n{ev['content']}" for i, ev in enumerate(evidence, 1)
    )
    user = (
        f"Question: {state['query']}\n\nEvidence:\n{blocks}\n\n"
        "Write a grounded answer citing evidence as [n]."
    )

    if state.get("stream_tokens"):
        # SSE path: stream the draft live as custom graph events.
        draft, trace_acc = await _stream_draft(llm, user)
    else:
        trace_acc: list = []
        result = await llm.complete(
            "strong", SYNTHESIZER_SYSTEM, user, trace=trace_acc, node="synthesizer"
        )
        draft = result.text
    log.info("synthesizer_done", draft_len=len(draft))
    return {
        "draft": draft,
        "refused": False,
        "trace": (state.get("trace") or []) + trace_acc,
    }


async def _stream_draft(llm: LLMClient, user: str) -> tuple[str, list[dict]]:
    """Stream the draft token by token, emitting each delta via the custom
    stream writer (consumed as SSE `token` events); returns (draft, trace)."""
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None
    start = time.perf_counter()
    parts: list[str] = []
    async for token in llm.stream("strong", SYNTHESIZER_SYSTEM, user):
        parts.append(token)
        if writer is not None:
            writer({"node": "synthesizer", "text": token})
    return "".join(parts).strip(), [
        {
            "node": "synthesizer",
            "model": llm.model_for("strong"),
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "input_tokens": 0,
            "output_tokens": len(parts),  # ~one token per delta
        }
    ]
