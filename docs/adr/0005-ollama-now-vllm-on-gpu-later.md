# ADR-0005: Ollama for local dev; vLLM as the GPU upgrade path

Date: 2026-10-10 | Status: accepted

## Context
The current host has no GPU and 7.8 GB RAM. vLLM requires a CUDA/ROCm GPU;
Llama 3.1 8B at Q4 alone would not coexist with Postgres, the embedder and the
reranker in that budget.

## Decision
Run Ollama with llama3.2:3b as the local backend through the OpenAI-compatible
endpoint (ADR-0002 client). Design keeps vLLM as a drop-in: same API contract,
so pointing LLM_BASE_URL at a vLLM server (Llama 3.1 8B / Mistral 7B,
PagedAttention, continuous batching) is a configuration change, validated on a
GPU host when one is available. Benchmarks will be reported with hardware
stated; no throughput/latency numbers are claimed until measured.

## Consequences
+ Development is fully local today; nothing about the architecture changes
  for the GPU upgrade.
+ Model dir (OLLAMA_MODELS) and HF caches live under /opt/omniagent for
  one-command teardown.
- The resume claim "vLLM with PagedAttention and continuous batching" is only
  honest after a GPU run with measured numbers; until then the system runs
  Ollama and says so.
