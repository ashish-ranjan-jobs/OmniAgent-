"""Hybrid retrieval pipeline: dense + BM25 -> RRF -> cross-encoder rerank.

Every stage is toggleable (via settings, or explicitly for the ablation
harness) so each stage's contribution is measurable on the same golden set.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Chunk, Document
from app.db.session import get_sessionmaker
from app.ingestion.embedder import embed_query
from app.retrieval.dense import dense_search
from app.retrieval.rerank import rerank_scores
from app.retrieval.rrf import fuse
from app.retrieval.sparse import sparse_search

log = get_logger(__name__)


@dataclass
class Evidence:
    chunk_id: str
    document_id: str
    content: str
    title: str
    scores: dict = field(default_factory=dict)


@dataclass
class RetrievalStageFlags:
    """Ablation switches; defaults come from .env."""

    dense: bool = True
    bm25: bool = True
    rerank: bool = True


async def _load_chunk_rows(chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    sm = get_sessionmaker()
    async with sm() as session:
        stmt = (
            select(Chunk.id, Chunk.document_id, Chunk.content, Document.title)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(chunk_ids))
        )
        rows = (await session.execute(stmt)).all()
    by_id = {r.id: r for r in rows}
    return [
        {
            "chunk_id": r.id,
            "document_id": r.document_id,
            "content": r.content,
            "title": r.title,
        }
        for cid in chunk_ids
        if (r := by_id.get(cid)) is not None
    ]


class RetrievalPipeline:
    """Retrieves evidence for a query through configurable hybrid stages."""

    async def retrieve(
        self, query: str, flags: RetrievalStageFlags | None = None
    ) -> tuple[list[Evidence], dict]:
        s = get_settings()
        if flags is None:
            flags = RetrievalStageFlags(
                dense=True, bm25=s.hybrid_enabled, rerank=s.rerank_enabled
            )
        timings: dict[str, float] = {}
        dense_rows: list[dict] = []
        bm25_rows: list[tuple[str, float]] = []

        if flags.dense:
            t0 = time.perf_counter()
            qvec = await asyncio.to_thread(embed_query, query)
            dense_rows = await dense_search(qvec, s.retrieval_top_k)
            timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if flags.bm25:
            t0 = time.perf_counter()
            bm25_rows = await sparse_search(query, s.retrieval_top_k)
            timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        dense_sim = {r["chunk_id"]: 1.0 - r["distance"] for r in dense_rows}
        bm25_score = {cid: score for cid, score in bm25_rows}

        candidates: list[dict] = []
        rrf_score: dict[str, float] = {}
        if flags.dense and flags.bm25:
            rankings = {
                "dense": [r["chunk_id"] for r in dense_rows],
                "bm25": [cid for cid, _ in bm25_rows],
            }
            fused = fuse(rankings, k=s.rrf_k)[: s.fusion_top_n]
            rrf_score = dict(fused)
            candidates = await _load_chunk_rows([cid for cid, _ in fused])
        elif flags.dense:
            candidates = dense_rows
        elif flags.bm25:
            candidates = await _load_chunk_rows([cid for cid, _ in bm25_rows])
        else:
            return [], timings

        if flags.rerank and candidates:
            t0 = time.perf_counter()
            scores = await rerank_scores(query, [c["content"] for c in candidates])
            timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            final = [(candidates[i], scores[i]) for i in order[: s.rerank_top_k]]
        else:
            final = [(c, None) for c in candidates[: s.retrieval_top_k]]

        evidence = [
            Evidence(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                content=c["content"],
                title=c["title"],
                scores={
                    "dense_similarity": round(dense_sim.get(c["chunk_id"], 0.0), 4),
                    "bm25": round(bm25_score.get(c["chunk_id"], 0.0), 4),
                    "rrf": round(rrf_score.get(c["chunk_id"], 0.0), 4),
                    "rerank": None if rs is None else round(rs, 4),
                },
            )
            for c, rs in final
        ]
        log.info(
            "retrieval_complete",
            query=query[:80],
            stages=flags.__dict__,
            evidence_count=len(evidence),
            timings_ms=timings,
        )
        return evidence, timings
