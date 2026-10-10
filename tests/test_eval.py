"""Unit tests for the eval harness pure functions (no LLM, no DB)."""

from __future__ import annotations

from app.eval.runner import aggregate_triad, reflection_delta, write_reports


def _scored(grounded=1.0, ctx=1.0, ans=1.0, refl=0):
    return {
        "query": "q",
        "context_relevance": ctx,
        "groundedness": grounded,
        "answer_relevance": ans,
        "reflections": refl,
        "refused": False,
        "latency_ms": 1.0,
    }


def _refusal_item(correct=True):
    return {"query": "u", "correct_refusal": correct, "latency_ms": 1.0}


def test_aggregate_triad_mixed():
    agg = aggregate_triad(
        [_scored(grounded=1.0), _scored(grounded=0.0), _scored(grounded=0.5), _refusal_item()]
    )
    assert agg["n_scored"] == 3
    assert agg["groundedness"] == 0.5
    assert agg["hallucination_rate"] == 0.5
    assert agg["refusal_accuracy"] == 1.0
    assert agg["n_unanswerable"] == 1


def test_aggregate_triad_no_refusals():
    agg = aggregate_triad([_scored()])
    assert agg["refusal_accuracy"] is None
    assert agg["n_unanswerable"] == 0


def test_reflection_delta_positive():
    off = aggregate_triad([_scored(grounded=0.5), _scored(grounded=0.7)])
    on = aggregate_triad([_scored(grounded=0.9), _scored(grounded=0.9)])
    d = reflection_delta(off, on)
    assert d["groundedness"] == {"off": 0.6, "on": 0.9, "delta": 0.3}
    assert d["hallucination_rate"]["off"] == 0.4
    assert d["hallucination_rate"]["on"] == 0.1
    assert d["hallucination_rate"]["reduction_absolute"] == 0.3
    assert d["hallucination_rate"]["reduction_percent"] == 75.0
    assert d["answer_relevance_delta"] == 0.0
    assert d["context_relevance_delta"] == 0.0
    assert d["refusal_accuracy_delta"] is None
    assert d["n_scored"] == 2


def test_reflection_delta_zero_baseline():
    off = aggregate_triad([_scored(grounded=1.0)])
    on = aggregate_triad([_scored(grounded=1.0)])
    d = reflection_delta(off, on)
    assert d["hallucination_rate"]["reduction_absolute"] == 0.0
    assert d["hallucination_rate"]["reduction_percent"] is None


def test_write_reflection_report(tmp_path):
    off = aggregate_triad([_scored(grounded=0.5)])
    on = aggregate_triad([_scored(grounded=1.0)])
    payload = {
        "mode": "reflection",
        "golden_items": 1,
        "max_reflections_off": 0,
        "max_reflections_on": 2,
        "avg_reflections_on": 1.5,
        "aggregate_reflection_off": off,
        "aggregate_reflection_on": on,
        "delta": reflection_delta(off, on),
    }
    json_path, md_path = write_reports("reflection", payload, out_dir=str(tmp_path))
    assert json_path.exists() and md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "## Reflection off" in md
    assert "## Reflection on" in md
    assert "## Delta (on - off)" in md
    assert "hallucination rate: 0.5 -> 0.0" in md
    assert "relative 100.0%" in md


def test_aggregate_triad_error_items_excluded():
    agg = aggregate_triad(
        [_scored(grounded=1.0), {"query": "q", "error": "LLM backend failed after retries"}]
    )
    assert agg["n_scored"] == 1
    assert agg["n_errors"] == 1
    assert agg["groundedness"] == 1.0
    assert agg["hallucination_rate"] == 0.0
