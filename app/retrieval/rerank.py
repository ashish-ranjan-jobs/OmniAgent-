"""Cross-encoder reranking (sentence-transformers, local CPU).

A cross-encoder scores (query, passage) jointly through one transformer --
a stronger relevance signal than bi-encoder cosine similarity -- but costs
~10-50ms per pair on CPU, so we only rerank the top-N fused candidates,
never the whole corpus (latency/quality tradeoff, see ADR-0001).
"""

from __future__ import annotations

import asyncio

from sentence_transformers import CrossEncoder

from app.core.config import get_settings

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(get_settings().reranker_model, max_length=512)
    return _reranker


def _score_pairs(query: str, texts: list[str]) -> list[float]:
    return [float(s) for s in _get_reranker().predict([(query, t) for t in texts])]


async def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Relevance score per (query, text) pair, run off the event loop."""
    if not texts:
        return []
    return await asyncio.to_thread(_score_pairs, query, texts)
