# ADR-0006: SSE streaming at node granularity before token-level streaming

Date: 2026-10-10 | Status: accepted

## Context
Research runs take minutes on a CPU 3B backend; clients need progress before
completion. Two streaming granularities were possible: per-token (the
synthesizer's generation streamed token by token) or per-node (one event when
each graph step completes).

## Decision
Ship node-granular SSE first: `POST /api/v1/research/stream` emits a `node`
event per completed graph step (via `graph.astream(stream_mode="updates")`),
including every reflection round as it happens, then a `result` event with the
full response contract (citations, verdicts, trace) and a terminal `done`
event. An `error` event carries backend failure after retries, since the HTTP
status is fixed once streaming begins.

## Consequences
+ Works with ANY OpenAI-compatible backend, including ones whose chat
  completions we consume without token streaming; no client changes when the
  backend changes.
+ Reflection progress is visible, which is the interesting part of this graph.
- Token-level "typing" feel requires the LLM client to stream completions and
  a token-bus through the synthesizer node -- the documented upgrade path
  (client stream=True + an async queue into the node and the SSE layer).
