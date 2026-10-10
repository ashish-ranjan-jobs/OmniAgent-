from __future__ import annotations

from app.retrieval.rrf import fuse


def test_fuse_exact_math_and_order():
    rankings = {"dense": ["a", "b", "c"], "bm25": ["b", "d", "a"]}
    fused = fuse(rankings, k=60)
    # b: 1/62 (dense r2) + 1/61 (bm25 r1) = 0.03252 (top)
    # a: 1/61 (dense r1) + 1/63 (bm25 r3) = 0.03206
    # d: 1/62 (bm25 r2); c: 1/63 (dense r3)
    assert [cid for cid, _ in fused] == ["b", "a", "d", "c"]
    assert abs(fused[0][1] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(fused[1][1] - (1 / 61 + 1 / 63)) < 1e-9


def test_fuse_deterministic_tie_break():
    one = fuse({"dense": ["x", "y"], "bm25": ["y", "x"]})
    two = fuse({"bm25": ["y", "x"], "dense": ["x", "y"]})
    assert [cid for cid, _ in one] == [cid for cid, _ in two]


def test_fuse_single_system_preserves_rank():
    fused = fuse({"dense": ["m", "n", "p"]}, k=10)
    assert [cid for cid, _ in fused] == ["m", "n", "p"]


def test_fuse_empty():
    assert fuse({}) == []
