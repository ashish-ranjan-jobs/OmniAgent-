# Distilling Rerankers

Production rerankers are distilled: a large cross-encoder teacher annotates query-passage pairs, and tiny students are trained to reproduce its scores. The students approximate teacher ordering at a fraction of the latency and memory, which matters when reranking sits on the serving path with tight latency budgets. Distillation data quality bounds the student: noisy teacher labels teach the student the teacher's mistakes.
