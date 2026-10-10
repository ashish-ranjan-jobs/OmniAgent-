# Observability and Tracing

Distributed tracing records a trace of spans across service and model
boundaries: one span per LLM call, retrieval step, or tool invocation, each
carrying timing, token counts, and the model identifier. Histograms of span
latency expose tail effects that averages hide. Trace sampling balances
observability cost against coverage of rare failure paths.
