"""Research node: run the hybrid pipeline per subquery, merge, dedupe."""

from __future__ import annotations

from dataclasses import asdict

from app.agents.state import OmniAgentState
from app.core.logging import get_logger
from app.retrieval.pipeline import RetrievalPipeline

log = get_logger(__name__)

MAX_EVIDENCE = 12


def _rank_key(ev: dict) -> float:
    scores = ev.get("scores") or {}
    for key in ("rerank", "rrf"):
        value = scores.get(key)
        if value is not None:
            return float(value)
    return 0.0


async def research_node(state: OmniAgentState, retriever: RetrievalPipeline) -> dict:
    merged: dict[str, dict] = {}
    trace_entries: list[dict] = []

    for subquery in state.get("subqueries") or [state["query"]]:
        evidence, timings = await retriever.retrieve(subquery)
        trace_entries.append(
            {"node": "research", "subquery": subquery, "timings_ms": timings}
        )
        for ev in evidence:
            payload = asdict(ev)
            payload["subquery"] = subquery
            existing = merged.get(ev.chunk_id)
            if existing is None or _rank_key(payload) > _rank_key(existing):
                merged[ev.chunk_id] = payload

    evidence_list = sorted(merged.values(), key=_rank_key, reverse=True)[:MAX_EVIDENCE]
    log.info("research_done", evidence_count=len(evidence_list))
    return {
        "evidence": evidence_list,
        "trace": (state.get("trace") or []) + trace_entries,
    }
