"""Synthesizer node: grounded draft with [n] citations, or a refusal."""

from __future__ import annotations

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
    trace_acc: list = []
    result = await llm.complete(
        "strong", SYNTHESIZER_SYSTEM, user, trace=trace_acc, node="synthesizer"
    )
    log.info("synthesizer_done", draft_len=len(result.text))
    return {
        "draft": result.text,
        "refused": False,
        "trace": (state.get("trace") or []) + trace_acc,
    }
