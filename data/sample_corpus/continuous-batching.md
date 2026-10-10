# Continuous Batching

Static (request-level) batching groups requests, runs them to completion,
then admits the next batch: short requests stall behind long ones and GPU
compute sits idle. Continuous (iteration-level) batching treats every decode
step as a scheduling opportunity: finished sequences are evicted and queued
requests are admitted immediately, at each iteration.

Combined with PagedAttention's paged KV cache, continuous batching keeps GPU
utilization high under mixed-length workloads, and it is the core scheduling
idea behind vLLM's serving throughput. The tradeoff is scheduler complexity:
the runtime must handle preemption, page allocation, and sequence state per
step instead of per request.
