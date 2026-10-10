# Dataset Distillation

Dataset distillation synthesizes a tiny set of synthetic samples that trains a model to similar accuracy as the original set: gradient-matching methods optimize synthetic images so their training trajectory approximates the real data's. Where knowledge distillation compresses a teacher model, dataset distillation compresses the dataset itself. Uses include continual-learning memory budgets and cheap hyperparameter probes.
