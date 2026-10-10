# LoRA Fine-Tuning

LoRA (Low-Rank Adaptation) freezes base model parameters and injects small
trainable low-rank matrices into attention and feed-forward projections.
Trainable parameters drop from billions to millions, which slashes optimizer
state and gradient memory. The adapted deltas can be merged back into the
base weights at inference time for zero added latency.
