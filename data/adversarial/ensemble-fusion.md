# Ensembling and Score Fusion

Ensembles combine multiple models: averaging or voting over predictions, stacking a meta-learner over base outputs, or bagging perturbed training sets. Score fusion requires care: raw outputs live on different scales (probabilities, margins, ranks), so practitioners z-score normalize per model, learn blend weights on held-out data, or stack. Fusion helps when member errors are decorrelated; correlated members add cost without adding signal.
