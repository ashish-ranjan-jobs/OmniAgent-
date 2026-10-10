"""Prometheus metrics, emitted at the API edge from the graph trace."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

HTTP_LATENCY = Histogram(
    "omniagent_http_request_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)
LLM_CALLS = Counter(
    "omniagent_llm_calls_total",
    "LLM calls by node and model",
    ["node", "model"],
)
LLM_LATENCY = Histogram(
    "omniagent_llm_latency_seconds",
    "LLM call latency by node",
    ["node"],
)
REFLECTIONS = Counter(
    "omniagent_reflections_total",
    "Reflection loops taken (fact-checker -> planner)",
)

__all__ = [
    "CONTENT_TYPE_LATEST",
    "HTTP_LATENCY",
    "LLM_CALLS",
    "LLM_LATENCY",
    "REFLECTIONS",
    "generate_latest",
]
