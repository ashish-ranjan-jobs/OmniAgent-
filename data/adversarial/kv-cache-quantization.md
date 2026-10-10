# KV Cache Quantization

The key-value cache grows with sequence length and is a dominant memory cost in long-context serving. Quantizing cached keys and values to 8-bit (or 4-bit) formats roughly halves (or quarters) cache bytes per token at small quality loss, directly raising the sequences per accelerator. Per-layer and per-head cache budgets, plus selective retention policies for stale tokens, push the same budget further.
