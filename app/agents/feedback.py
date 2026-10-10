"""Structured reflection feedback (pure, deterministic, unit-tested).

Reflection is NOT "try again": the fact-checker's failed claims are converted
into concrete suggested subqueries, so the next planner round has something
actionable to plan against (ADR-0004).
"""

from __future__ import annotations

MAX_SUGGESTED_SUBQUERIES = 4


def build_reflection_feedback(verdicts: list[dict]) -> dict:
    """Build feedback from fact-check verdicts; empty when all claims pass."""
    failed = [v for v in verdicts if not v.get("supported", False)]
    if not failed:
        return {}
    suggested = [f"evidence for the claim: {v.get('claim', '')}" for v in failed]
    return {
        "failed_claims": [v.get("claim", "") for v in failed],
        "suggested_subqueries": suggested[:MAX_SUGGESTED_SUBQUERIES],
    }
