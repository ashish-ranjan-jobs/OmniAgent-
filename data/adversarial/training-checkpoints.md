# Checkpointing in Distributed Training

Distributed training jobs snapshot model shards, optimizer state, and dataloader position so a pre-empted job resumes without restarting from zero. Sharded state (per-rank files or a consolidated store) keeps snapshot volume proportional to per-worker state, asynchronous uploads overlap with training steps, and rolling retention bounds storage. Frequency trades I/O overhead against lost progress on failure.
