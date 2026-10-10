"""Token-level streaming: synthesizer deltas surface as custom stream events."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_graph
from app.llm.client import LLMResult
from app.retrieval.pipeline import Evidence

PASS_ALL = (
    '{"verdicts": [{"claim": "c", "supported": true, '
    '"evidence_ids": [1], "reason": "stated"}]}'
)


class StreamingLLM:
    """complete() for planner/fact-checker; stream() for the synthesizer."""

    def __init__(self, fact=PASS_ALL):
        self.fact = fact
        self.calls: list[str] = []
        self.streamed = False

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
            return LLMResult(text='{"subqueries": ["q1"]}', model="fake", latency_ms=1)
        if node == "fact_checker":
            return LLMResult(text=self.fact, model="fake", latency_ms=1)
        return LLMResult(text="Draft [1].", model="fake", latency_ms=1)

    async def stream(self, role, system, user):
        self.streamed = True
        for token in ["Hel", "lo ", "world", "."]:
            yield token

    def model_for(self, role):
        return "fake"


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


def make_graph(llm):
    return build_graph(
        llm,
        FakeRetriever(["rrf", "bm25", "rerank"]),
        checkpointer=MemorySaver(),
        refusal_min_evidence=3,
    )


async def test_synthesizer_tokens_stream_as_custom_events():
    llm = StreamingLLM()
    graph = make_graph(llm)
    config = {"configurable": {"thread_id": "token-stream-on"}}
    tokens, node_events = [], []
    async for mode, payload in graph.astream(
        {"query": "q", "trace": [], "reflection_count": 0, "stream_tokens": True},
        config,
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            tokens.append(payload["text"])
        else:
            node_events.extend(k for k in payload if k != "__start__")
    assert llm.streamed is True
    assert tokens == ["Hel", "lo ", "world", "."]
    assert "planner" in node_events and "finalize" in node_events
    snapshot = await graph.aget_state(config)
    assert snapshot.values["draft"] == "Hello world."
    assert snapshot.values["refused"] is False


async def test_streaming_off_keeps_complete_path():
    llm = StreamingLLM()
    graph = make_graph(llm)
    result = await graph.ainvoke(
        {"query": "q", "trace": [], "reflection_count": 0},  # no stream_tokens
        {"configurable": {"thread_id": "token-stream-off"}},
    )
    assert llm.streamed is False
    assert result["draft"] == "Draft [1]."
