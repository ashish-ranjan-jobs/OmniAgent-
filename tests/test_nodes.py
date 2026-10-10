from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_graph
from app.agents.nodes.planner import planner_node
from app.agents.nodes.research import research_node
from app.llm.client import LLMResult
from app.retrieval.pipeline import Evidence


class FakeLLM:
    def __init__(
        self,
        planner_text='{"subqueries": ["q1", "q2"]}',
        synth="Draft [1].",
        fact='{"verdicts": []}',
    ):
        self.planner_text = planner_text
        self.synth = synth
        self.fact = fact
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
            return LLMResult(text=self.planner_text, model="fake", latency_ms=1)
        if node == "fact_checker":
            return LLMResult(text=self.fact, model="fake", latency_ms=1)
        return LLMResult(text=self.synth, model="fake", latency_ms=1)


class FakeRetriever:
    def __init__(self, titles):
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


CONFIG = {"configurable": {"thread_id": "test-thread"}}


async def test_planner_falls_back_on_bad_json():
    llm = FakeLLM(planner_text="totally not json")
    update = await planner_node({"query": "what is rrf"}, llm)
    assert update["subqueries"] == ["what is rrf"]


async def test_refusal_on_thin_evidence():
    llm = FakeLLM()
    retriever = FakeRetriever(["rrf"])  # 1 evidence < threshold 3
    graph = build_graph(llm, retriever, checkpointer=MemorySaver(), refusal_min_evidence=3)
    result = await graph.ainvoke(
        {"query": "what is rrf", "trace": [], "reflection_count": 0}, CONFIG
    )
    assert result["refused"] is True
    assert "insufficient" in result["answer"].lower()
    assert result["verdicts"] == []  # fact checker skipped on refusal


async def test_all_supported_no_reflection():
    llm = FakeLLM(
        fact=(
            '{"verdicts": [{"claim": "c", "supported": true, '
            '"evidence_ids": [1], "reason": "stated"}]}'
        )
    )
    retriever = FakeRetriever(["rrf", "bm25", "hybrid-retrieval"])
    graph = build_graph(llm, retriever, checkpointer=MemorySaver(), refusal_min_evidence=3)
    result = await graph.ainvoke(
        {"query": "what is rrf", "trace": [], "reflection_count": 0}, CONFIG
    )
    assert result["reflection_count"] == 0
    assert result["unsupported_claims"] == []
    assert result["answer"] == "Draft [1]."
    assert len(result["citations"]) == 3


async def test_reflection_loop_bounded_and_terminates():
    llm = FakeLLM(
        fact=(
            '{"verdicts": [{"claim": "c", "supported": false, '
            '"evidence_ids": [], "reason": "not stated"}]}'
        )
    )
    retriever = FakeRetriever(["rrf", "bm25", "hybrid-retrieval"])
    graph = build_graph(
        llm,
        retriever,
        checkpointer=MemorySaver(),
        max_reflections=1,
        refusal_min_evidence=3,
    )
    result = await graph.ainvoke(
        {"query": "what is rrf", "trace": [], "reflection_count": 0}, CONFIG
    )
    # Budget 1: exactly one reflection round, then finalize even though claims fail.
    assert result["reflection_count"] == 1
    assert result["unsupported_claims"]
    planner_calls = [c for c in llm.calls if c == "planner"]
    assert len(planner_calls) == 2  # initial + one reflection round


async def test_planner_dedupes_subqueries():
    llm = FakeLLM(planner_text='{"subqueries": ["bm25 k1", "bm25 k1", "bm25 b"]}')
    update = await planner_node({"query": "bm25 parameters"}, llm)
    assert update["subqueries"] == ["bm25 k1", "bm25 b"]


async def test_research_accumulates_evidence_across_rounds():
    """Reflection rounds must keep prior evidence first (citation stability:
    the draft's [n] citations must keep pointing at the same text) and append
    newly retrieved chunks after it, deduped by chunk_id."""
    retriever = FakeRetriever(["rrf", "bm25", "hnsw", "cross-encoder-rerank"])
    prior = [
        {
            "chunk_id": "c0",
            "document_id": "d0",
            "content": "content about rrf",
            "title": "rrf",
            "scores": {"rerank": 0.9},
            "subquery": "q",
        }
    ]
    state = {"query": "q", "subqueries": ["q"], "evidence": prior, "trace": []}
    update = await research_node(state, retriever)
    evidence = update["evidence"]
    ids = [e["chunk_id"] for e in evidence]
    # Prior evidence is first and untouched; new evidence appended; deduped.
    assert ids[0] == "c0"
    assert ids == ["c0", "c1", "c2", "c3"]
    assert len(set(ids)) == len(ids)
    # Trace appends without mutating the original list.
    assert update["trace"][0]["node"] == "research"
