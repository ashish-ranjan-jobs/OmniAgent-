# The Attention Mechanism

Attention computes a weighted sum over value vectors, with weights from a
compatibility function between a query and each key. Scaled dot-product
attention divides by the square root of the key dimension to keep gradients
stable. Multi-head attention runs several such subspaces in parallel so the
model can track different relation types at once.
