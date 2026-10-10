"""Graph state schema (LangGraph TypedDict).

Every node returns a partial update; LangGraph merges. Trace entries are
appended immutably (new list per node) so checkpointed state stays consistent.
"""

from __future__ import annotations

from typing import TypedDict


class OmniAgentState(TypedDict, total=False):
    query: str
    subqueries: list[str]
    # Structured reflection feedback: failed claims + suggested subqueries.
    feedback: dict
    evidence: list[dict]  # serialized Evidence + subquery provenance
    draft: str
    refused: bool
    verdicts: list[dict]  # {claim, supported, evidence_ids, reason}
    unsupported_claims: list[str]
    reflection_count: int
    citations: list[dict]  # {n, title, document_id, chunk_id}
    answer: str
    trace: list[dict]  # per-LLM-call {node, model, latency_ms, tokens}
    error: str
