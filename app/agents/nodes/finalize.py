"""Finalize node: answer + citations from the evidence actually used."""

from __future__ import annotations

from app.agents.state import OmniAgentState


def finalize_node(state: OmniAgentState) -> dict:
    evidence = state.get("evidence") or []
    citations = [
        {
            "n": i,
            "title": ev["title"],
            "document_id": ev["document_id"],
            "chunk_id": ev["chunk_id"],
        }
        for i, ev in enumerate(evidence, 1)
    ]
    return {"answer": state.get("draft", ""), "citations": citations}
