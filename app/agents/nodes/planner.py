"""Planner node: query -> 1-4 focused subqueries (multi-query expansion)."""

from __future__ import annotations

from app.agents.prompts import PLANNER_SYSTEM
from app.agents.state import OmniAgentState
from app.core.logging import get_logger
from app.llm.client import LLMClient, parse_json

log = get_logger(__name__)

MAX_SUBQUERIES = 4


async def planner_node(state: OmniAgentState, llm: LLMClient) -> dict:
    feedback = state.get("feedback") or {}
    failed = feedback.get("failed_claims") or []

    user = f"Question: {state['query']}\n"
    if failed:
        user += (
            "These claims lacked evidence in earlier searches; "
            "plan searches that would find evidence for them:\n"
        )
        user += "\n".join(f"- {c}" for c in failed) + "\n"
    user += 'Respond with JSON: {"subqueries": [...]}'

    trace_acc: list = []
    result = await llm.complete(
        "fast", PLANNER_SYSTEM, user, json_mode=True, trace=trace_acc, node="planner"
    )

    data = parse_json(result.text) or {}
    raw = data.get("subqueries") if isinstance(data, dict) else None
    subqueries = [
        q.strip() for q in (raw or []) if isinstance(q, str) and q.strip()
    ]
    # Order-preserving dedupe: small models often repeat subqueries.
    subqueries = list(dict.fromkeys(subqueries))[:MAX_SUBQUERIES]
    if not subqueries:
        subqueries = [state["query"]]  # graceful fallback, always searchable

    update: dict = {
        "subqueries": subqueries,
        "feedback": {},  # consumed; cleared for the next round
        "trace": (state.get("trace") or []) + trace_acc,
    }
    if failed:
        update["reflection_count"] = state.get("reflection_count", 0) + 1
    log.info("planner_done", subqueries=subqueries, reflection=failed != [])
    return update
