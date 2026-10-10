# Rank Fusion in Recommenders

Recommender systems blend multiple candidate sources: collaborative filtering scores, popularity ranks, freshness signals. Rank-based fusion is popular here because the blended scores live on incomparable scales and train-time calibration is expensive. Practitioners blend top-k lists per user slot and re-sort by the blended standing. Offline replay evaluation (uniform replay, propensity weighting) decides whether the blend beats the strongest single source before an A/B test sees it.
