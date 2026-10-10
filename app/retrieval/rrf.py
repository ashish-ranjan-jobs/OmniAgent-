"""Reciprocal Rank Fusion (pure, deterministic, unit-tested).

score(d) = sum over systems s of 1 / (K + rank_s(d)), K default 60.

RRF needs only *ranks*, so dense and sparse scores -- different units and
distributions -- are never linearly mixed (the weighted-sum mistake).
Tunable K: small K trusts top ranks more; large K flattens differences.
"""

from __future__ import annotations


def fuse(rankings: dict[str, list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse per-system ranked id lists into one ranking.

    rankings: {"dense": [id, ...], "bm25": [id, ...]} each in rank order.
    Returns [(id, fused_score)] sorted by score desc, ties broken by id
    (deterministic output for identical inputs -> testable).
    """
    scores: dict[str, float] = {}
    for _, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
