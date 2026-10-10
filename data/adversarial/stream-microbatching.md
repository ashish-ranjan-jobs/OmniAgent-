# Micro-Batching in Stream Processors

Stream processors (Spark Structured Streaming, Flink) balance latency and throughput with micro-batches: events are grouped into small time-sliced batches, executed, then the next slice runs. Pure continuous operators avoid the batch boundary but complicate scheduling and state management. Event-time semantics, watermarks, and exactly-once checkpoints make results correct despite late or replayed events.
