"""Research node: run the hybrid pipeline per subquery, merge, dedupe.

Across reflection rounds, evidence ACCUMULATES rather than being replaced:
prior evidence keeps its position (citation stability -- the draft's [n]
citations must keep pointing at the same text after a reflection round), and
newly retrieved chunks append after it, capped with per-round slack. Without
this, a reflection round could silently renumber the citations.
"""

from __future__ import annotations

from dataclasses import asdict

from app.agents.state import OmniAgentState
from app.core.logging import get_logger
from app.retrieval.pipeline import RetrievalPipeline

log = get_logger(__name__)

MAX_EVIDENCE = 12
# Slack for evidence appended during reflection rounds (prior first).
MAX_EVIDENCE_SLACK = 4


def _rank_key(ev: dict) -> float:
    scores = ev.get("scores") or {}
    for key in ("rerank", "rrf"):
        value = scores.get(key)
        if value is not None:
            return float(value)
    return 0.0


async def research_node(state: OmniAgentState, retriever: RetrievalPipeline) -> dict:
    # Prior round's evidence is the stable prefix of the new list.
    prior: dict[str, dict] = {
        ev["chunk_id"]: dict(ev) for ev in state.get("evidence") or []
    }
    fresh: dict[str, dict] = {}
    trace_entries: list[dict] = []

    for subquery in state.get("subqueries") or [state["query"]]:
        evidence, timings = await retriever.retrieve(subquery)
        trace_entries.append(
            {"node": "research", "subquery": subquery, "timings_ms": timings}
        )
        for ev in evidence:
            payload = asdict(ev)
            payload["subquery"] = subquery
            existing = fresh.get(ev.chunk_id)
            if existing is None or _rank_key(payload) > _rank_key(existing):
                fresh[ev.chunk_id] = payload

    cap = MAX_EVIDENCE if not prior else MAX_EVIDENCE + MAX_EVIDENCE_SLACK
    new_items = sorted(
        (e for cid, e in fresh.items() if cid not in prior),
        key=_rank_key,
        reverse=True,
    )[: max(cap - len(prior), 0)]
    evidence_list = list(prior.values()) + new_items

    log.info(
        "research_done",
        prior=len(prior),
        fresh=len(fresh),
        appended=len(new_items),
        total=len(evidence_list),
    )
    return {
        "evidence": evidence_list,
        "trace": (state.get("trace") or []) + trace_entries,
    }
