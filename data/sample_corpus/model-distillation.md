# Model Distillation

Distillation trains a small student model to reproduce a larger teacher
model's outputs, including soft probability distributions that carry more
signal per example than hard labels. Distilled students approximate teacher
quality at a fraction of the latency and memory, which is why latency-bound
services often serve a distilled ranker or classifier while reserving the
teacher for offline scoring.
