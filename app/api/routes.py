"""API v1: documents, ingest, research (multi-agent graph run)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.agents.graph import get_graph
from app.core.logging import get_logger
from app.db.models import Chunk, Document
from app.db.session import get_sessionmaker
from app.ingestion.pipeline import ingest_directory
from app.llm.client import LLMUnavailableError
from app.observability.metrics import LLM_CALLS, LLM_LATENCY, REFLECTIONS

router = APIRouter()
log = get_logger(__name__)


class IngestRequest(BaseModel):
    path: str = Field(default="data/sample_corpus", description="Directory to ingest")


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    thread_id: str | None = None


@router.get("/documents")
async def list_documents() -> dict:
    sm = get_sessionmaker()
    async with sm() as session:
        rows = (
            await session.execute(
                select(
                    Document.id,
                    Document.title,
                    Document.source_path,
                    func.count(Chunk.id).label("chunks"),
                )
                .outerjoin(Chunk, Chunk.document_id == Document.id)
                .group_by(Document.id)
                .order_by(Document.title)
            )
        ).all()
    return {
        "documents": [
            {"id": r.id, "title": r.title, "source_path": r.source_path, "chunks": r.chunks}
            for r in rows
        ]
    }


@router.post("/ingest")
async def ingest(request: IngestRequest) -> dict:
    try:
        summary = await ingest_directory(request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return summary


@router.post("/research")
async def research(request: ResearchRequest) -> dict:
    graph = await get_graph()
    thread_id = request.thread_id or uuid.uuid4().hex
    state = {"query": request.query, "trace": [], "reflection_count": 0}
    try:
        result = await graph.ainvoke(state, {"configurable": {"thread_id": thread_id}})
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Emit LLM metrics from the graph trace (observability at the edge).
    for entry in result.get("trace") or []:
        if entry.get("node") and entry.get("model"):
            LLM_LATENCY.labels(entry["node"]).observe(entry["latency_ms"] / 1000)
            LLM_CALLS.labels(entry["node"], entry["model"]).inc()
    if result.get("reflection_count"):
        REFLECTIONS.inc(result["reflection_count"])

    log.info(
        "research_done",
        thread_id=thread_id,
        reflections=result.get("reflection_count", 0),
        refused=bool(result.get("refused")),
    )
    return {
        "thread_id": thread_id,
        "query": request.query,
        "subqueries": result.get("subqueries", []),
        "answer": result.get("answer", ""),
        "refused": bool(result.get("refused")),
        "citations": result.get("citations", []),
        "verdicts": result.get("verdicts", []),
        "unsupported_claims": result.get("unsupported_claims", []),
        "reflection_count": result.get("reflection_count", 0),
        "trace": result.get("trace", []),
    }
