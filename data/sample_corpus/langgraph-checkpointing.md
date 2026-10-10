# LangGraph Checkpointing

LangGraph persists graph state through a checkpointer abstraction keyed by
thread_id. AsyncPostgresSaver stores checkpoints (and per-node writes) in
Postgres tables that the framework itself owns and creates via setup().

Why it matters architecturally: multi-turn agents resume from the persisted
state instead of resending the entire conversation history into model
context every turn, which bounds token cost per turn; the same mechanism
enables time-travel debugging (inspect state at any node boundary) and
human-in-the-loop interrupts (pause the graph, gather approval, resume).
Checkpoints are additive writes, which keeps replay deterministic given the
same node outputs.
