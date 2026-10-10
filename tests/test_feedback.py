from __future__ import annotations

from app.agents.feedback import build_reflection_feedback


def test_all_supported_returns_empty():
    assert build_reflection_feedback([{"claim": "a", "supported": True}]) == {}


def test_failed_claims_generate_structured_suggestions():
    verdicts = [
        {"claim": "x", "supported": True},
        {"claim": "y", "supported": False},
        {"claim": "z", "supported": False},
    ]
    fb = build_reflection_feedback(verdicts)
    assert fb["failed_claims"] == ["y", "z"]
    assert fb["suggested_subqueries"] == [
        "evidence for the claim: y",
        "evidence for the claim: z",
    ]


def test_suggestions_capped():
    verdicts = [{"claim": f"c{i}", "supported": False} for i in range(10)]
    fb = build_reflection_feedback(verdicts)
    assert len(fb["suggested_subqueries"]) == 4
