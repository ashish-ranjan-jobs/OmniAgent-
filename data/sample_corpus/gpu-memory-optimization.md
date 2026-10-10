# GPU Memory Optimization

Serving throughput is usually memory-bandwidth bound, not compute bound.
Techniques: KV cache offloading to cheaper tiers, weight quantization to
reduce reads, larger batches to amortize weight loads, and flash attention
kernels that avoid materializing the full attention matrix. The KV cache
grows with sequence length and batch size and is the first thing to overflow
on long-context workloads.
