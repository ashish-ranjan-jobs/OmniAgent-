# PagedAttention in vLLM

vLLM's PagedAttention manages the key-value (KV) cache the way an operating
system manages virtual memory: instead of one contiguous buffer per sequence,
the cache is split into fixed-size pages (blocks) that are allocated and
mapped on demand. Sequences no longer need over-reserved contiguous memory,
which eliminates most internal and external fragmentation.

Reported effects: KV cache waste drops to under roughly 4 percent (versus
60-80 percent waste with naive contiguous allocation in worst-case settings),
which means bigger effective batches on the same GPU and 2-4x throughput
improvement on latency-sensitive serving workloads. PagedAttention also
enables prefix sharing between sequences that begin with the same tokens.
