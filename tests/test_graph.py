"""Graph-level tests: the reflection loop must always terminate.

Adversarial worst case: the fact-checker fails every claim on every round.
The MAX_REFLECTIONS budget must bound the planner <- fact_checker cycle and
force finalize (ADR-0004), no matter what the LLM returns. All deps are
fakes -> no Postgres/Ollama required.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_graph, route_after_fact_checker
from app.llm.client import LLMResult
from app.retrieval.pipeline import Evidence

FAIL_ALL = (
    '{"verdicts": [{"claim": "c1", "supported": false, '
    '"evidence_ids": [], "reason": "not stated in evidence"}]}'
)
PASS_ALL = (
    '{"verdicts": [{"claim": "c1", "supported": true, '
    '"evidence_ids": [1], "reason": "stated in evidence [1]"}]}'
)
ROUND = ["planner", "synthesizer", "fact_checker"]
CONFIG = {"configurable": {"thread_id": "graph-termination-test"}}


class ScriptedLLM:
    """Fake LLM whose fact-checker verdicts follow a per-round script.

    The last script entry repeats once the sequence is exhausted, so an
    always-fail script fails on every round: the loop's worst case.
    """

    def __init__(self, fact_scripts: list[str]):
        self.fact_scripts = fact_scripts
        self.fact_calls = 0
        self.calls: list[str] = []

    async def complete(self, role, system, user, *, json_mode=False, trace=None, node=""):
        self.calls.append(node)
        if trace is not None:
            trace.append(
                {
                    "node": node,
                    "model": "fake",
                    "latency_ms": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
            )
        if node == "planner":
            return LLMResult(
                text='{"subqueries": ["q1", "q2"]}', model="fake", latency_ms=1
            )
        if node == "fact_checker":
            script = self.fact_scripts[min(self.fact_calls, len(self.fact_scripts) - 1)]
            self.fact_calls += 1
            return LLMResult(text=script, model="fake", latency_ms=1)
        return LLMResult(text="Draft [1].", model="fake", latency_ms=1)


class FakeRetriever:
    """Deterministic retriever: same evidence set for any query/subquery."""

    def __init__(self, titles: list[str]):
        self.titles = titles

    async def retrieve(self, query, flags=None):
        evidence = [
            Evidence(
                chunk_id=f"c{i}",
                document_id=f"d{i}",
                content=f"content about {t}",
                title=t,
                scores={"rerank": 1.0 - i * 0.1},
            )
            for i, t in enumerate(self.titles)
        ]
        return evidence, {"fake_ms": 1}


def make_graph(llm: ScriptedLLM, *, max_reflections: int, titles: list[str] | None = None):
    return build_graph(
        llm,
        FakeRetriever(titles or ["rrf", "bm25", "rerank"]),
        checkpointer=MemorySaver(),
        max_reflections=max_reflections,
        refusal_min_evidence=3,
    )


async def run(graph):
    return await graph.ainvoke(
        {"query": "what is rrf", "trace": [], "reflection_count": 0}, CONFIG
    )


async def test_terminates_when_fact_checker_fails_every_round():
    """Worst case: every round fails every claim. Budget 2 -> exactly three
    planner rounds (initial + two reflections), then finalize: the graph
    returns instead of looping forever."""
    llm = ScriptedLLM([FAIL_ALL])
    result = await run(make_graph(llm, max_reflections=2))
    assert llm.calls == ROUND * 3
    assert result["reflection_count"] == 2
    assert result["unsupported_claims"]
    assert result["answer"] == "Draft [1]."
    assert len(result["citations"]) == 3


async def test_exits_early_once_claims_become_supported():
    """Round 1 fails -> one reflection; round 2 passes -> finalize without
    spending the remaining budget."""
    llm = ScriptedLLM([FAIL_ALL, PASS_ALL])
    result = await run(make_graph(llm, max_reflections=3))
    assert llm.calls == ROUND * 2
    assert result["reflection_count"] == 1
    assert result["unsupported_claims"] == []
    assert result["answer"] == "Draft [1]."


async def test_zero_budget_never_loops():
    """max_reflections=0 disables the cycle outright: a single planner round,
    failed claims flagged at finalize."""
    llm = ScriptedLLM([FAIL_ALL])
    result = await run(make_graph(llm, max_reflections=0))
    assert llm.calls == ROUND
    assert result["reflection_count"] == 0
    assert result["unsupported_claims"]


async def test_refusal_short_circuits_to_finalize():
    """Thin evidence -> refusal bypasses the synthesizer LLM call, the
    fact-checker, and the reflection cycle entirely."""
    llm = ScriptedLLM([PASS_ALL])
    graph = make_graph(llm, max_reflections=2, titles=["rrf"])
    result = await run(graph)
    assert llm.calls == ["planner"]
    assert result["refused"] is True
    assert result["reflection_count"] == 0
    assert "insufficient" in result["answer"].lower()


def test_route_after_fact_checker_matrix():
    """The conditional edge alone decides termination: refused always
    finalizes; failed claims route to planner only while budget remains."""
    assert (
        route_after_fact_checker(
            {"refused": True, "feedback": {"failed_claims": ["x"]}, "reflection_count": 0},
            2,
        )
        == "finalize"
    )
    assert (
        route_after_fact_checker(
            {"feedback": {"failed_claims": ["x"]}, "reflection_count": 0}, 2
        )
        == "planner"
    )
    assert (
        route_after_fact_checker(
            {"feedback": {"failed_claims": ["x"]}, "reflection_count": 2}, 2
        )
        == "finalize"
    )
    assert route_after_fact_checker({"feedback": {}, "reflection_count": 0}, 2) == "finalize"
    assert route_after_fact_checker({}, 2) == "finalize"
