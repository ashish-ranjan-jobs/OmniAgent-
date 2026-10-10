"""API v1: documents, ingest, research (sync + SSE streaming)."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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


def _initial_state(query: str) -> dict:
    """Fresh run state. Every mutable key is reset explicitly so that reusing
    a thread_id never leaks a previous query's evidence/verdicts into a new
    one (LangGraph merges new input into the thread's checkpoint state)."""
    return {
        "query": query,
        "subqueries": [],
        "feedback": {},
        "evidence": [],
        "draft": "",
        "refused": False,
        "verdicts": [],
        "unsupported_claims": [],
        "reflection_count": 0,
        "citations": [],
        "answer": "",
        "trace": [],
    }


def _response(thread_id: str, result: dict) -> dict:
    return {
        "thread_id": thread_id,
        "query": result.get("query", ""),
        "subqueries": result.get("subqueries", []),
        "answer": result.get("answer", ""),
        "refused": bool(result.get("refused")),
        "citations": result.get("citations", []),
        "verdicts": result.get("verdicts", []),
        "unsupported_claims": result.get("unsupported_claims", []),
        "reflection_count": result.get("reflection_count", 0),
        "trace": result.get("trace", []),
    }


def _emit_metrics(result: dict) -> None:
    """Emit LLM metrics from the graph trace (observability at the edge)."""
    for entry in result.get("trace") or []:
        if entry.get("node") and entry.get("model"):
            LLM_LATENCY.labels(entry["node"]).observe(entry["latency_ms"] / 1000)
            LLM_CALLS.labels(entry["node"], entry["model"]).inc()
    if result.get("reflection_count"):
        REFLECTIONS.inc(result["reflection_count"])


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
            {
                "id": r.id,
                "title": r.title,
                "source_path": r.source_path,
                "chunks": r.chunks,
            }
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
    try:
        result = await graph.ainvoke(
            _initial_state(request.query),
            {"configurable": {"thread_id": thread_id}},
        )
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    _emit_metrics(result)
    log.info(
        "research_done",
        thread_id=thread_id,
        reflections=result.get("reflection_count", 0),
        refused=bool(result.get("refused")),
    )
    return _response(thread_id, result)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _node_payload(node: str, update: dict) -> dict:
    """Compact per-node progress for the SSE stream; full state in `result`."""
    if node == "planner":
        return {"subqueries": update.get("subqueries", [])}
    if node == "research":
        evidence = update.get("evidence", [])
        titles = [e.get("title") for e in evidence][:8]
        return {"evidence_count": len(evidence), "titles": titles}
    if node == "synthesizer":
        return {
            "refused": bool(update.get("refused")),
            "draft_preview": (update.get("draft") or "")[:200],
        }
    if node == "fact_checker":
        verdicts = update.get("verdicts", [])
        return {
            "claims": len(verdicts),
            "unsupported": len(update.get("unsupported_claims", [])),
            "will_reflect": bool(update.get("feedback")),
        }
    if node == "finalize":
        return {"answer_preview": (update.get("answer") or "")[:200]}
    return {}


@router.post("/research/stream")
async def research_stream(request: ResearchRequest) -> StreamingResponse:
    """Server-Sent Events stream of a research run.

    Events: `node` (one per graph step, compact payload), `result` (the full
    response contract including citations, verdicts and trace), `error` (LLM
    backend failure after retries), `done` (terminal, always sent).
    """
    graph = await get_graph()
    thread_id = request.thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    async def events():
        started = time.perf_counter()
        try:
            async for update in graph.astream(
                _initial_state(request.query), config, stream_mode="updates"
            ):
                for node, payload in update.items():
                    if node == "__start__":
                        continue
                    yield _sse("node", {"node": node, **_node_payload(node, payload)})
            snapshot = await graph.aget_state(config)
            result = dict(snapshot.values)
            _emit_metrics(result)
            yield _sse(
                "result",
                {
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    **_response(thread_id, result),
                },
            )
        except LLMUnavailableError as exc:
            yield _sse("error", {"detail": str(exc)})
        yield _sse("done", {"thread_id": thread_id})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
