# ADR-0002: First-party LLM client instead of a framework integration

Date: 2026-10-10 | Status: accepted

## Context
The system talks to an OpenAI-compatible endpoint (Ollama today; vLLM or a
commercial API later). Framework integrations (e.g. langchain ChatModel
subclasses) hide retry policy, timeouts, model routing, and JSON parsing.

## Decision
Own a thin client (`app/llm/client.py`): explicit tenacity retries on
connection/timeout/5xx, one budgeted timeout, role-based model routing
("fast" for planner/verdicts, "strong" for synthesis), a tolerant JSON parser
(fence stripping, first-JSON-value extraction), and a graceful one-shot
fallback when a backend rejects json_object mode.

## Consequences
+ Every behavior above is unit-testable and visible in code review.
+ Swapping backends is a .env change (LLM_BASE_URL / LLM_MODEL / FAST_MODEL).
+ Trace entries (model, latency, tokens) are produced at the client, so
  observability and Prometheus emission are uniform across nodes.
- We re-implement ~100 lines a framework would provide; accepted for control.
