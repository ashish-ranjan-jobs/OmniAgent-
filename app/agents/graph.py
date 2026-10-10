"""Graph assembly: planner -> research -> synthesizer -> fact-check -> loop.

The cyclical edge is fact_checker -> planner, taken only when:
  (a) some claims failed verification, AND
  (b) the reflection budget (MAX_REFLECTIONS) is not exhausted.
Otherwise the graph finalizes. Termination is therefore bounded and provable
(ADR-0004). Deps (LLM client, retriever) are injected -> fully fakeable tests.
"""

from __future__ import annotations

import functools

from langgraph.graph import END, START, StateGraph

from app.agents.nodes.fact_checker import fact_checker_node
from app.agents.nodes.finalize import finalize_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.research import research_node
from app.agents.nodes.synthesizer import synthesizer_node
from app.agents.state import OmniAgentState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.client import LLMClient
from app.retrieval.pipeline import RetrievalPipeline

log = get_logger(__name__)


def route_after_fact_checker(state: OmniAgentState, max_reflections: int) -> str:
    """Conditional edge: reflect (re-plan) or finalize."""
    if state.get("refused"):
        return "finalize"
    feedback = state.get("feedback") or {}
    if feedback.get("failed_claims") and state.get("reflection_count", 0) < max_reflections:
        return "planner"
    return "finalize"


def build_graph(
    llm: LLMClient,
    retriever: RetrievalPipeline,
    checkpointer=None,
    max_reflections: int | None = None,
    refusal_min_evidence: int | None = None,
):
    """Compile the agent graph with injected dependencies."""
    s = get_settings()
    max_r = s.max_reflections if max_reflections is None else max_reflections
    min_ev = s.refusal_min_evidence if refusal_min_evidence is None else refusal_min_evidence

    planner = functools.partial(planner_node, llm=llm)
    research = functools.partial(research_node, retriever=retriever)
    synthesizer = functools.partial(synthesizer_node, llm=llm, refusal_min_evidence=min_ev)
    fact_checker = functools.partial(fact_checker_node, llm=llm)
    route = functools.partial(route_after_fact_checker, max_reflections=max_r)

    graph = StateGraph(OmniAgentState)
    graph.add_node("planner", planner)
    graph.add_node("research", research)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("fact_checker", fact_checker)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "synthesizer")
    graph.add_edge("synthesizer", "fact_checker")
    graph.add_conditional_edges(
        "fact_checker",
        route,
        {"planner": "planner", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    compiled = graph.compile(checkpointer=checkpointer)
    log.info("graph_built", max_reflections=max_r, refusal_min_evidence=min_ev)
    return compiled


_graph = None


async def get_graph():
    """Singleton compiled graph with the Postgres checkpointer."""
    global _graph
    if _graph is None:
        from app.agents.checkpointer import get_checkpointer

        checkpointer = await get_checkpointer()
        _graph = build_graph(LLMClient(), RetrievalPipeline(), checkpointer=checkpointer)
    return _graph
